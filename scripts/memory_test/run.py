"""T4 memory test: heaviest input, full-length generation, per-device pass check."""

import argparse
import hashlib
import json
import platform
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
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
from clave_ad.vlm.template import template_kwargs
from sizes import heaviest, load_sizes


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


def chat_tokens(processor: Any, gen: dict, images: list[Image.Image]) -> int:
    """Input tokens of a one-word user turn with these images."""
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": "x"}]
    text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, **template_kwargs(gen))
    return int(processor(text=[text], images=images or None, return_tensors="pt")["input_ids"].shape[1])


def sdpa_recorder(counts: Counter) -> Any:
    """torch scaled_dot_product_attention that counts the kernel torch selects for each call.

    torch._fused_sdp_choice is the selection the CUDA dispatch itself runs on the same arguments; calls are keyed by
    head counts, head size and whether a mask was given.
    """
    original = torch.nn.functional.scaled_dot_product_attention
    names = {int(b): name for name, b in torch.nn.attention.SDPBackend.__members__.items()}

    def wrapped(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor | None = None,
                dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None,
                enable_gqa: bool = False) -> torch.Tensor:
        choice = torch._fused_sdp_choice(query, key, value, attn_mask, dropout_p, is_causal, scale=scale,
                                         enable_gqa=enable_gqa)
        counts[f"q{query.shape[1]} kv{key.shape[1]} d{query.shape[-1]} mask={attn_mask is not None} "
               f"gqa={enable_gqa}: {names.get(choice, choice)}"] += 1
        return original(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                        scale=scale, enable_gqa=enable_gqa)

    return wrapped


def time_limit(signum: int, frame: Any) -> None:
    raise TimeoutError("memory test time limit reached")


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
    if "vlm" in cfg:
        # model and generation of a VLM config; the test keeps its own generation keys (min_new_tokens)
        vlm = yaml.safe_load(Path(cfg["vlm"]).read_text())
        cfg["model"] = vlm["model"]
        cfg["generation"] = {**vlm["generation"], **cfg.get("generation", {})}
    torch.manual_seed(cfg["seed"])
    generator = torch.Generator().manual_seed(cfg["seed"])

    test = cfg["test_input"]
    gen = cfg["generation"]

    oom = False
    error = None
    run = {}
    start_time = time.monotonic()
    if "time_limit_minutes" in cfg:
        # the alarm raises in the next Python step, so a long generation stops between two tokens
        signal.signal(signal.SIGALRM, time_limit)
        signal.alarm(int(cfg["time_limit_minutes"] * 60))

    def log(stage: str) -> None:
        # the last stage reached goes into the result, so a stopped test shows where it was
        run["last_stage"] = stage
        print(f"memory test {time.monotonic() - start_time:.0f}s: {stage}", flush=True)

    try:
        log("loading model")
        model, processor = load_model(cfg["model"])
        run["after_load"] = {"memory_allocated": torch.cuda.memory_allocated(0),
                             "memory_reserved": torch.cuda.memory_reserved(0)}
        log(f"model loaded: allocated {run['after_load']['memory_allocated'] / 2**30:.2f} GiB, "
            f"reserved {run['after_load']['memory_reserved'] / 2**30:.2f} GiB")
        # devices actually holding parameters; hf_device_map is not set for a single-device load
        run["device_map"] = sorted({str(p.device) for p in model.parameters()})
        size = getattr(processor.image_processor, "size", None)
        run["processor_size"] = dict(size) if size else None
        run["max_soft_tokens"] = getattr(processor.image_processor, "max_soft_tokens", None)
        # end-of-turn ids as decoded strings, to check the generation_config values
        run["eos_tokens"] = {str(i): processor.tokenizer.decode([i]) for i in gen["eos_token_ids"]}
        log(f"eos tokens: {run['eos_tokens']}")
        if test.get("image_size") == "heaviest_from_sizes_file":
            # the dataset image size with the most image tokens in this processor; the sizes were read locally from
            # the image headers into one committed result file (scripts/memory_test/sizes.py)
            log("size candidates: start")
            size_start = time.monotonic()
            matches = sorted(Path().glob(test["sizes_file"]))
            if len(matches) != 1:
                raise ValueError(f"expected one sizes file for {test['sizes_file']}, found {len(matches)}")
            run["sizes_file"] = {"path": str(matches[0]), "sha256": hashlib.sha256(matches[0].read_bytes()).hexdigest()}
            sizes = load_sizes(json.loads(matches[0].read_text()))
            base = chat_tokens(processor, gen, [])
            (width, height), run["size_candidates"] = heaviest(
                sizes, lambda w, h: chat_tokens(processor, gen, [Image.new("RGB", (w, h))]) - base)
            run["chosen_size"] = f"{width}x{height}"
            run["size_seconds"] = time.monotonic() - size_start
            log(f"size candidates: {len(sizes)} sizes in {run['size_seconds']:.0f}s; chosen {run['chosen_size']} with "
                f"{run['size_candidates'][run['chosen_size']]['image_tokens']} image tokens")
        else:
            width, height = test["image_width"], test["image_height"]
        images = [synthetic_image(width, height, generator) for _ in range(test["num_images"])]
        # Linear layers per top-level module and type, to check which parts stayed out of 4-bit
        run["linear_layers"] = dict(Counter(f"{'.'.join(n.split('.')[:2])}:{type(m).__name__}"
                                            for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)))
        previous = truncated_output(processor.tokenizer, test["previous_output_tokens"])
        run["previous_output_tokens"] = len(processor.tokenizer(previous, add_special_tokens=False)["input_ids"])
        prompt = test["prompt"] + test["retry_template"].format(previous_output=previous, error=test["retry_error"])
        inputs = build_inputs(processor, images, prompt, gen)
        # start of the rendered chat (synthetic prompt), to check that no default system prompt is inserted
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": "x"}]
        run["template_head"] = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                             add_generation_prompt=True, **template_kwargs(gen))[:300]
        run["input_tokens"] = int(inputs["input_ids"].shape[1])
        if "image_grid_thw" in inputs:
            # Qwen: image_grid_thw counts patches before the 2x2 merge
            merge = processor.image_processor.merge_size ** 2
            run["image_tokens"] = [int(g.prod()) // merge for g in inputs["image_grid_thw"]]
        inputs = inputs.to(model.device)
        log(f"inputs built: {run['input_tokens']} input tokens; warm-up generation")
        # warm-up on the same inputs so CUDA and kernel setup stay out of the timings; its prefill also records the
        # sdpa kernel of every attention call (vision encoder and language model)
        backends = Counter()
        run["sdpa_backends"] = backends
        original_sdpa = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = sdpa_recorder(backends)
        try:
            model.generate(**inputs, max_new_tokens=1, min_new_tokens=1, do_sample=gen["do_sample"],
                           eos_token_id=gen["eos_token_ids"])
        finally:
            torch.nn.functional.scaled_dot_product_attention = original_sdpa
            log(f"sdpa kernels: {dict(backends)}")
        nonfinite = NonFiniteFlag()
        torch.cuda.synchronize()
        log("generation start")
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
        log(f"generation end: {int(output.shape[1] - inputs['input_ids'].shape[1])} new tokens in "
            f"{run['generate_seconds']:.0f}s")
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
        log(f"prefill {run['prefill_seconds']:.1f}s")
    except Exception as e:
        oom = isinstance(e, torch.cuda.OutOfMemoryError)
        # type and first line only; full messages can carry local paths
        lines = str(e).splitlines()
        error = f"{type(e).__name__}: {lines[0][:300] if lines else ''}"
        traceback.print_exc()
    signal.alarm(0)
    timed_out = error is not None and error.startswith("TimeoutError")
    run["seconds"] = time.monotonic() - start_time

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
            "timed_out": timed_out,
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
