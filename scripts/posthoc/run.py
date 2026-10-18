"""Post-hoc c4 experiment worker for one GPU (configs/posthoc/c4_cause.yaml): E1 and E2 per target image.

E1 is the B0 request on the original image with grid lines, with the main-run format retries; E2 is the layout
question on the original image without retries. Outcomes are cached per request; the shard file holds records
without raw text (the raw text stays in the cache bundle).
"""

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path
from string import Template

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.data.datasets import load_image
from clave_ad.requery.inputs import image_sha256
from clave_ad.runinfo import file_sha256
from clave_ad.vlm.cache_key import request_key
from clave_ad.vlm.chat import NonFiniteHook, kernel_implementations, render, request
from clave_ad.vlm.model import load_model
from clave_ad.vlm.parse import parse_answer, parse_layout
from clave_ad.vlm.records import summarize
from grid import draw_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/posthoc/c4_cause.yaml")
    parser.add_argument("--targets", required=True, help="results/posthoc/targets_*.json")
    parser.add_argument("--data-root", required=True, help="folder holding <dataset>/<path> images")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main_run = yaml.safe_load(Path(cfg["main_run"]).read_text())
    vlm = yaml.safe_load(Path(main_run["vlm"]).read_text())
    rules = yaml.safe_load(Path(main_run["rules"]).read_text())
    types = yaml.safe_load(Path(vlm["data"]["type_names"]).read_text())
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if alloc_conf != vlm["runtime"]["pytorch_cuda_alloc_conf"]:
        sys.exit(f"PYTORCH_CUDA_ALLOC_CONF is {alloc_conf!r}; set it from the config before torch starts")
    prompt_dir = Path(vlm["prompts"]["dir"])
    b0_template = (prompt_dir / vlm["prompts"]["b0"]).read_text()
    retry_template = (prompt_dir / vlm["prompts"]["retry"]).read_text()
    output_rules = (prompt_dir / vlm["prompts"]["output_rules"]).read_text().strip()
    e1, e2 = cfg["conditions"]["e1"], cfg["conditions"]["e2"]
    layout_template = (prompt_dir / e2["prompt"]).read_text()
    gen = {**vlm["generation"], "max_retries": rules["format"]["max_retries"]}
    gen_e2 = {**vlm["generation"], "max_retries": e2["max_retries"]}
    targets = json.loads(Path(args.targets).read_text())["images"]
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    model, processor = load_model(vlm["model"])
    hook = NonFiniteHook(model)
    device = f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    torch.cuda.reset_peak_memory_stats()

    def cached(kind: str, images: list, hashes: list[str], prompt: str, parse, g: dict,
               retry: str) -> tuple[dict | None, str]:
        """Cached outcome or a new generation and its cache key, as in scripts/main_run/run.py.

        The outcome is None when out of memory.
        """
        key = request_key(vlm["model"]["revision"], g, kind, hashes, prompt, retry)
        path = cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text()), key
        try:
            outcome = request(model, processor, images, prompt, parse, g, retry, hook)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            oom.append([kind, row["dataset"], row["path"]])
            return None, key
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(outcome))
        tmp.rename(path)
        return outcome, key

    records, oom = [], []
    for index, row in enumerate(targets):
        if index % args.num_shards != args.shard:
            continue
        d, c, p = row["dataset"], row["category"], row["path"]
        if file_sha256(data_root / d / p) != row["sha256"]:
            sys.exit(f"image does not match the targets file: {d}/{p}")
        original = load_image(data_root / d, p)
        names = list(types[d]["categories"][c])
        values = {"category": c.replace("_", " "), "candidates": ", ".join(names), "output_rules": output_rules}
        # $error is filled per retry inside request(), as in the main run
        retry = Template(retry_template).safe_substitute(candidates=values["candidates"], output_rules=output_rules)

        gridded = draw_grid(original, e1["grid"]["size"], e1["grid"]["width_share"], e1["grid"]["min_px"],
                            e1["grid"]["border_px"])
        grid_sha = image_sha256(gridded)
        out, key = cached("e1", [gridded], [grid_sha], render(b0_template, **values),
                          partial(parse_answer, candidates=names), gen, retry)
        if out is not None:
            records.append({**summarize("e1", row, out, device), "role": row["role"], "inputs_sha256": [grid_sha],
                            "cache_key": key})

        # E2 has no retry, so the retry text is empty and never used
        out, key = cached("e2", [original], [row["sha256"]], render(layout_template, category=values["category"]),
                          parse_layout, gen_e2, "")
        if out is not None:
            records.append({**summarize("e2", row, out, device), "role": row["role"],
                            "inputs_sha256": [row["sha256"]], "cache_key": key})

    shard = {"shard": args.shard, "device": device, "gpu": torch.cuda.get_device_name(0),
             "kernels": kernel_implementations(), "oom": oom,
             "max_memory_reserved": torch.cuda.max_memory_reserved(0),
             "max_memory_allocated": torch.cuda.max_memory_allocated(0),
             "total_memory": torch.cuda.get_device_properties(0).total_memory,
             "pytorch_cuda_alloc_conf": alloc_conf, "records": records}
    out_dir = Path(cfg["output_dir"]) / "posthoc_shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"shard_{args.shard}.json").write_text(json.dumps(shard) + "\n")
    print(f"shard {args.shard}: {len(records)} records, oom {len(oom)}")


if __name__ == "__main__":
    main()
