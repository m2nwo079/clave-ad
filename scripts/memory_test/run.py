"""T4 memory test: heaviest input, full-length generation, per-device pass check."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from transformers import LogitsProcessor, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.vlm.model import build_inputs, load_model


def git_commit() -> str:
    # Kaggle runs a git archive with no .git, so the push script writes GIT_COMMIT
    stamp = Path("GIT_COMMIT")
    if stamp.exists():
        return stamp.read_text().strip()
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class NonFiniteFlag(LogitsProcessor):
    """Records whether any step produced NaN or inf logits (fp16 overflow check)."""

    def __init__(self) -> None:
        self.seen = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.seen = self.seen or not bool(torch.isfinite(scores).all())
        return scores


def synthetic_image(width: int, height: int, generator: torch.Generator) -> Image.Image:
    pixels = torch.randint(0, 256, (height, width, 3), dtype=torch.uint8, generator=generator)
    return Image.fromarray(pixels.numpy(), "RGB")


def truncated_output(tokenizer: Any, num_tokens: int) -> str:
    """Synthetic previous output cut at num_tokens, like a generation that hit max_new_tokens."""
    unit = '{"is_defective": true, "defect_type": "Different Colour Spot", "location_cells": ["r1c3"], "rationale": "'
    ids = tokenizer(unit * num_tokens, add_special_tokens=False)["input_ids"][:num_tokens]
    return tokenizer.decode(ids)


def device_memory() -> list[dict]:
    rows = []
    for d in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(d)
        rows.append({
            "device": d,
            "name": torch.cuda.get_device_name(d),
            "total_memory": torch.cuda.get_device_properties(d).total_memory,
            "max_memory_reserved": torch.cuda.max_memory_reserved(d),
            "max_memory_allocated": torch.cuda.max_memory_allocated(d),
            "mem_get_info_used": total - free,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/memory_test/qwen3_5_9b_1gpu.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])

    test = cfg["test_input"]
    gen = cfg["generation"]
    images = [synthetic_image(test["image_width"], test["image_height"], generator) for _ in range(test["num_images"])]

    oom = False
    error = None
    run = {}
    try:
        model, processor = load_model(cfg["model"])
        # devices actually holding parameters; hf_device_map is not set for a single-device load
        run["device_map"] = sorted({str(p.device) for p in model.parameters()})
        run["processor_size"] = dict(processor.image_processor.size)
        previous = truncated_output(processor.tokenizer, test["previous_output_tokens"])
        run["previous_output_tokens"] = len(processor.tokenizer(previous, add_special_tokens=False)["input_ids"])
        prompt = test["prompt"] + test["retry_template"].format(previous_output=previous, error=test["retry_error"])
        inputs = build_inputs(processor, images, prompt, gen["enable_thinking"], gen["add_vision_id"])
        # image_grid_thw counts patches before the 2x2 merge
        merge = processor.image_processor.merge_size ** 2
        run["input_tokens"] = int(inputs["input_ids"].shape[1])
        run["image_tokens"] = [int(g.prod()) // merge for g in inputs["image_grid_thw"]]
        inputs = inputs.to(model.device)
        # warm-up on the same inputs so CUDA and kernel setup stay out of the timings
        model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=gen["do_sample"], eos_token_id=gen["eos_token_ids"])
        nonfinite = NonFiniteFlag()
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=gen["max_new_tokens"],
            min_new_tokens=gen["min_new_tokens"],
            do_sample=gen["do_sample"],
            eos_token_id=gen["eos_token_ids"],
            logits_processor=LogitsProcessorList([nonfinite]),
        )
        torch.cuda.synchronize()
        run["generate_seconds"] = time.perf_counter() - start
        new_ids = output[0, inputs["input_ids"].shape[1]:]
        run["new_tokens"] = int(new_ids.shape[0])
        run["full_length"] = run["new_tokens"] == gen["min_new_tokens"]
        run["nonfinite_logits"] = nonfinite.seen
        # fp16 sanity check only; not a ground-truth metric
        run["generated_text"] = processor.decode(new_ids, skip_special_tokens=False)
        # prefill only: one new token
        start = time.perf_counter()
        model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=gen["do_sample"], eos_token_id=gen["eos_token_ids"])
        torch.cuda.synchronize()
        run["prefill_seconds"] = time.perf_counter() - start
    except Exception as e:
        oom = isinstance(e, torch.cuda.OutOfMemoryError)
        # type and first line only; full messages can carry local paths
        lines = str(e).splitlines()
        error = f"{type(e).__name__}: {lines[0][:300] if lines else ''}"
        traceback.print_exc()

    devices = device_memory()
    threshold = cfg["pass_threshold"]["reserved_ratio"]
    for row in devices:
        row["reserved_ratio"] = row["max_memory_reserved"] / row["total_memory"]
        row["pass"] = error is None and row["reserved_ratio"] <= threshold

    created_at = datetime.now(timezone.utc)
    run_id = f"memory_test_{created_at:%Y%m%dT%H%M%SZ}"
    result = {
        "run_id": run_id,
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "dataset": "synthetic",
        "seed": cfg["seed"],
        "created_at": created_at.isoformat(),
        "metrics": {
            "pass": bool(devices) and all(row["pass"] for row in devices),
            "oom": oom,
            "error": error,
            "devices": devices,
            **run,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": package_version("transformers"),
            "bitsandbytes": package_version("bitsandbytes"),
            "accelerate": package_version("accelerate"),
            "torchvision": package_version("torchvision"),
            "pillow": package_version("pillow"),
            "pyyaml": package_version("pyyaml"),
            "platform": platform.platform(),
        },
    }
    out = Path(cfg["output_dir"]) / f"{run_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"pass={result['metrics']['pass']} oom={oom} -> {out}")
    if error is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
