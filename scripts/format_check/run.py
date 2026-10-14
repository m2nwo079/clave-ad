"""Format check worker for one GPU: the configured request kinds (B0, B1, B3 or B2, re-query) on its share of images.

Each request is cached under cache_dir by a hash of everything that defines it, so a rerun skips finished work.
Raw outputs stay in the cache; the shard record file holds statuses, token counts, times and output hashes.
"""

import argparse
import hashlib
import json
import os
import sys
from functools import partial
from pathlib import Path
from string import Template

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.data.datasets import load_image
from clave_ad.data.maps import downscale, jet_image, zoom_box
from clave_ad.vlm.chat import NonFiniteHook, kernel_implementations, render, request
from clave_ad.vlm.model import load_model
from clave_ad.vlm.parse import has_control_char, has_think_block, parse_answer, parse_check, unwrap


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def image_sha256(image: Image.Image) -> str:
    """Hash of a derived input image (B2 map, half-size original, crop), for the cache key and the record."""
    return hashlib.sha256(f"{image.mode}{image.size}".encode() + image.tobytes()).hexdigest()


def summarize(kind: str, image: dict, outcome: dict, device: str, **flags: bool) -> dict:
    """Shard record without raw text; the final output is kept as a hash for cross-device comparison."""
    attempts = [{k: a[k] for k in ("input_tokens", "new_tokens", "seconds", "truncated", "nonfinite", "error")}
                for a in outcome["attempts"]]
    final_text = outcome["attempts"][-1]["text"] if outcome["attempts"] else ""
    return {
        "kind": kind, "dataset": image["dataset"], "category": image["category"], "path": image["path"],
        "device": device, "ok": outcome["parsed"] is not None, "final_error": outcome["error"],
        "retries": len(outcome["attempts"]) - (0 if flags.get("forced_retry") else 1),
        "output_sha256": sha256(final_text), "attempts": attempts,
        # tolerated wrapper and the think-tag violation, counted separately
        "fenced": unwrap(final_text)[1], "control_char": has_control_char(final_text),
        "think_block": has_think_block(final_text), **flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/format_check/qwen3_5_9b.yaml")
    parser.add_argument("--data-root", required=True, help="folder holding samples.json and <dataset>/<path> images")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    types = yaml.safe_load(Path(cfg["data"]["type_names"]).read_text())
    gen, checks = cfg["generation"], cfg["checks"]
    prompt_dir = Path(cfg["prompts"]["dir"])
    templates = {k: (prompt_dir / v).read_text() for k, v in cfg["prompts"].items() if k not in ("dir", "output_rules")}
    kinds = cfg["kinds"]
    output_rules = (prompt_dir / cfg["prompts"]["output_rules"]).read_text().strip()
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if alloc_conf != cfg["runtime"]["pytorch_cuda_alloc_conf"]:
        sys.exit(f"PYTORCH_CUDA_ALLOC_CONF is {alloc_conf!r}; set it from the config before torch starts")
    data_root = Path(args.data_root)
    rows = json.loads((data_root / "samples.json").read_text())["images"]
    samples = [r for r in rows if r["role"] == "sample"]
    references = {(r["dataset"], r["category"]): r for r in rows if r["role"] == "reference"}
    detector = None
    if {"b2", "requery"} & set(kinds):
        result = json.loads(Path(cfg["detector"]["result"]).read_text())
        maps_index = json.loads((data_root / "maps" / "maps.json").read_text())
        if maps_index["detector_run_id"] != result["run_id"]:
            sys.exit("packed maps come from another detector run; rerun pack_maps.py")
        map_keys = {(r["dataset"], r["path"]): r["cache_key"] for r in result["records"]}
        scales = result["metrics"]["b2_scale"]
        detector = {"result": cfg["detector"]["result"], "b2_scale": scales, "maps_sha256": maps_index["sha256"]}

    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cfg["output_dir"]) / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(cfg["model"])
    hook = NonFiniteHook(model)
    device = f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    torch.cuda.reset_peak_memory_stats()

    def originals(*image_rows: dict) -> list[tuple[Image.Image, str]]:
        return [(load_image(data_root / r["dataset"], r["path"]), r["sha256"]) for r in image_rows]

    def map_of(row: dict) -> tuple[np.ndarray, str]:
        key = map_keys[(row["dataset"], row["path"])]
        path = data_root / "maps" / f"{key}.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != detector["maps_sha256"][key]:
            sys.exit(f"map does not match maps.json: {row['dataset']} {row['path']}")
        with np.load(path) as entry:
            return entry["anomaly_map"], digest

    def derived(kind: str, row: dict) -> tuple[list[tuple[Image.Image, str]], dict]:
        """Input images of B2 or the re-query, and the record fields that describe them."""
        [(original, original_sha)] = originals(row)
        anomaly_map, map_sha = map_of(row)
        if kind == "b2":
            scale = scales[row["dataset"]]
            jet = jet_image(anomaly_map, original.size, scale["low"], scale["high"])
            inputs = [(original, original_sha), (jet, image_sha256(jet))]
            return inputs, {"map_sha256": map_sha, "inputs_sha256": [h for _, h in inputs]}
        rq = cfg["requery"]
        box, shift = zoom_box(anomaly_map, original.size, rq["top_fraction"], rq["box_scale"], rq["grid"])
        half, crop = downscale(original, rq["downscale"]), original.crop(box)
        inputs = [(half, image_sha256(half)), (crop, image_sha256(crop))]
        return inputs, {"map_sha256": map_sha, "inputs_sha256": [h for _, h in inputs], "crop_box": list(box),
                        "crop_size": list(crop.size), "crop_shift": list(shift)}

    def cached(kind: str, inputs: list[tuple[Image.Image, str]], prompt: str, parse, retry: str,
               forced: str | None = None) -> dict:
        # everything that changes the request goes into the key
        key = sha256(json.dumps({"revision": cfg["model"]["revision"], "generation": gen, "kind": kind,
                                 "images": [h for _, h in inputs], "prompt": prompt,
                                 "retry": retry, "forced": forced}, sort_keys=True))
        path = cache_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        outcome = request(model, processor, [image for image, _ in inputs], prompt, parse, gen, retry, hook, forced)
        path.write_text(json.dumps(outcome))
        return outcome

    records = []
    for index, image in enumerate(samples):
        mine = index % args.num_shards == args.shard
        consistency = index < checks["device_consistency_images"]
        if not (mine or consistency):
            continue
        names = list(types[image["dataset"]]["categories"][image["category"]])
        # category names such as "metal_nut" are shown with spaces, like the defect types
        values = {"category": image["category"].replace("_", " "), "candidates": ", ".join(names),
                  "output_rules": output_rules}
        # $error is filled per retry inside request()
        retry = Template(templates["retry"]).safe_substitute(candidates=values["candidates"], output_rules=output_rules)
        answer_parse = partial(parse_answer, candidates=names)
        forced_check = mine and index < checks["forced_retry_images"]
        if "b0" in kinds:
            b0_prompt = render(templates["b0"], **values)
            b0 = cached("b0", originals(image), b0_prompt, answer_parse, retry)
            records.append(summarize("b0", image, b0, device, consistency=consistency and not mine))
            if forced_check:
                forced = cached("b0", originals(image), b0_prompt, answer_parse, retry, checks["broken_output"])
                records.append(summarize("b0", image, forced, device, forced_retry=True))
        if mine and "b1" in kinds:
            reference = references[(image["dataset"], image["category"])]
            b1 = cached("b1", originals(reference, image), render(templates["b1"], **values), answer_parse, retry)
            records.append(summarize("b1", image, b1, device))
        if mine and "b3" in kinds and b0["parsed"] is not None:
            b3_prompt = render(templates["b3"], category=values["category"], answer=json.dumps(b0["parsed"]))
            b3 = cached("b3", originals(image), b3_prompt, partial(parse_check, answer=b0["parsed"]), retry)
            records.append(summarize("b3", image, b3, device))
        # B2 and the re-query run on every sample image (the re-query is forced, normal images included)
        for kind in ("b2", "requery"):
            if kind not in kinds:
                continue
            inputs, fields = derived(kind, image)
            prompt = render(templates[kind], **values)
            outcome = cached(kind, inputs, prompt, answer_parse, retry)
            records.append({**summarize(kind, image, outcome, device, consistency=consistency and not mine), **fields})
            if forced_check:
                outcome = cached(kind, inputs, prompt, answer_parse, retry, checks["broken_output"])
                records.append({**summarize(kind, image, outcome, device, forced_retry=True), **fields})

    shard = {
        "shard": args.shard, "device": device, "gpu": torch.cuda.get_device_name(0),
        "kernels": kernel_implementations(),
        "max_memory_reserved": torch.cuda.max_memory_reserved(0),
        "max_memory_allocated": torch.cuda.max_memory_allocated(0),
        "pytorch_cuda_alloc_conf": alloc_conf,
        "prompts": {k: v for k, v in cfg["prompts"].items() if k != "dir"},
        "kinds": kinds,
        "detector": detector,
        "total_memory": torch.cuda.get_device_properties(0).total_memory,
        "records": records,
    }
    (out_dir / f"shard_{args.shard}.json").write_text(json.dumps(shard, indent=2) + "\n")
    print(f"shard {args.shard}: {len(records)} records")


if __name__ == "__main__":
    main()
