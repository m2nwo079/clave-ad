"""VLM worker of the main run for one GPU: the request kinds of the configured session on its share of test images.

Requests are cached as in the format check, so a rerun or a later session skips finished work. B3 and the P1
re-query use the B0 answer; when B0 is not part of the session it must already be in the cache (never generated).
The worker stops before a new image once its time budget is spent, and stops at once when the same request runs
out of GPU memory again after an earlier session did (design doc 13.4).
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from string import Template

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.data.datasets import load_image
from clave_ad.requery.inputs import b2_inputs, requery_inputs
from clave_ad.runinfo import file_sha256
from clave_ad.verifiers.verify import verify
from clave_ad.vlm.cache_key import request_key
from clave_ad.vlm.chat import NonFiniteHook, kernel_implementations, render, request
from clave_ad.vlm.model import load_model
from clave_ad.vlm.parse import parse_answer, parse_check
from clave_ad.vlm.records import summarize


def load_detector(root: Path, cfg: dict, main_seed: int) -> dict:
    """Detector outputs attached to the kernel, checked against the committed detector result and run.yaml."""
    result = json.loads(Path(cfg["detector"]["result"]).read_text())
    if result["run_id"] != cfg["detector"]["run_id"]:
        sys.exit("detector result does not match run.yaml run_id")
    if result["files"]["maps.json"] != cfg["detector"]["maps_json_sha256"]:
        sys.exit("detector result maps.json hash does not match run.yaml")
    for name, digest in result["files"].items():
        if file_sha256(root / name) != digest:
            sys.exit(f"attached detector output {name} does not match {result['run_id']}")
    with np.load(root / "embeddings.npz") as e:
        embeddings = dict(zip(e["names"].tolist(), e["embeddings"]))
    prototypes = defaultdict(dict)
    with np.load(root / "prototypes.npz") as p:
        for key, vector in zip(p["keys"].tolist(), p["vectors"]):
            seed, name = key.split("|")
            dataset, category, kind = name.split("/", 2)
            if seed == str(main_seed):
                prototypes[(dataset, category)][kind] = vector
    return {"features": json.loads((root / "features.json").read_text()), "embeddings": embeddings,
            "prototypes": prototypes, "maps": json.loads((root / "maps.json").read_text())["sha256"],
            "thresholds": {k: v["threshold"] for k, v in result["metrics"]["thresholds"].items()},
            "provenance": {"result": cfg["detector"]["result"], "run_id": result["run_id"],
                           "kernel": cfg["detector"]["kernel"], "kernel_version": cfg["detector"]["kernel_version"],
                           "files": result["files"]}}


def session_images(split: dict, session: dict) -> list[dict]:
    """Test images of the session's datasets; a smoke session keeps evenly spaced images of a few categories."""
    rows = [r for r in split["vlm_images"] if r["dataset"] in session["kinds"]]
    if "categories" not in session:
        return rows
    picked = []
    for dataset, categories in session["categories"].items():
        for c in categories:
            group = sorted((r for r in rows if r["dataset"] == dataset and r["category"] == c), key=lambda r: r["path"])
            picked += group[::max(1, len(group) // session["per_category"])][:session["per_category"]]
    return picked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main_run/run.yaml")
    parser.add_argument("--data-root", required=True, help="folder holding <dataset>/<path> images")
    parser.add_argument("--detector-root", required=True, help="attached detector output holding maps.json")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    vlm = yaml.safe_load(Path(cfg["vlm"]).read_text())
    rules = yaml.safe_load(Path(cfg["rules"]).read_text())
    split = json.loads(Path(cfg["split_result"]).read_text())
    types = yaml.safe_load(Path(vlm["data"]["type_names"]).read_text())
    session = cfg["sessions"][cfg["session"]]
    gen = {**vlm["generation"], "max_retries": rules["format"]["max_retries"]}
    prompt_dir = Path(vlm["prompts"]["dir"])
    templates = {k: (prompt_dir / v).read_text() for k, v in vlm["prompts"].items() if k not in ("dir", "output_rules")}
    output_rules = (prompt_dir / vlm["prompts"]["output_rules"]).read_text().strip()
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if alloc_conf != vlm["runtime"]["pytorch_cuda_alloc_conf"]:
        sys.exit(f"PYTORCH_CUDA_ALLOC_CONF is {alloc_conf!r}; set it from the config before torch starts")
    deadline = time.monotonic() + cfg["time_budget_hours"] * 3600

    data_root, main_seed = Path(args.data_root), rules["type"]["library_main_seed"]
    det = load_detector(Path(args.detector_root), cfg, main_seed)
    scales = json.loads(Path(vlm["detector"]["result"]).read_text())["metrics"]["b2_scale"]
    references = {(r["dataset"], r["category"]): r for r in json.loads(Path(cfg["references"]).read_text())["images"]
                  if r["role"] == "reference"}
    library = {(r["dataset"], r["path"]) for r in split["library"][str(main_seed)]}
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    oom_file = cache_dir / "_oom.jsonl"
    earlier_oom = {tuple(json.loads(line)) for line in oom_file.read_text().splitlines()} if oom_file.exists() else set()

    model, processor = load_model(vlm["model"])
    hook = NonFiniteHook(model)
    device = f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    torch.cuda.reset_peak_memory_stats()
    oom, missing, stop = [], [], {"repeat_oom": False}
    # requests generated in this session and requests restored from the cache, per kind and dataset
    sources = defaultdict(lambda: {"new": 0, "restored": 0})

    def cached(kind: str, name: str, hashes: list[str], load, prompt: str, parse, retry: str,
               may_generate: bool = True) -> dict | None:
        """Cached outcome, a new generation, or None (not generated, or out of memory)."""
        key = request_key(vlm["model"]["revision"], gen, kind, hashes, prompt, retry)
        path = cache_dir / f"{key}.json"
        group = f"{kind}/{name.split('/')[0]}"
        if path.exists():
            sources[group]["restored"] += 1
            return json.loads(path.read_text())
        if not may_generate:
            missing.append([kind, name])
            return None
        try:
            outcome = request(model, processor, load(), prompt, parse, gen, retry, hook)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            oom.append([kind, name])
            stop["repeat_oom"] |= (kind, name) in earlier_oom
            return None
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(outcome))
        tmp.rename(path)
        sources[group]["new"] += 1
        return outcome

    def map_of(name: str) -> np.ndarray:
        key = det["features"][name]["cache_key"]
        path = Path(args.detector_root) / "maps" / f"{key}.npz"
        if file_sha256(path) != det["maps"][key]:
            sys.exit(f"map does not match maps.json: {name}")
        with np.load(path) as entry:
            return entry["anomaly_map"]

    records, complete, assigned = [], True, 0
    for index, row in enumerate(session_images(split, session)):
        if index % args.num_shards != args.shard:
            continue
        assigned += 1
        if stop["repeat_oom"]:
            break
        if time.monotonic() > deadline:
            complete = False
            continue
        d, c, p = row["dataset"], row["category"], row["path"]
        name, kinds = f"{d}/{p}", session["kinds"][d]
        names = list(types[d]["categories"][c])
        values = {"category": c.replace("_", " "), "candidates": ", ".join(names), "output_rules": output_rules}
        # $error is filled per retry inside request()
        retry = Template(templates["retry"]).safe_substitute(candidates=values["candidates"], output_rules=output_rules)
        answer_parse = partial(parse_answer, candidates=names)
        original = {}

        def load_original() -> Image.Image:
            if "image" not in original:
                if file_sha256(data_root / d / p) != row["sha256"]:
                    sys.exit(f"image does not match the split result: {name}")
                original["image"] = load_image(data_root / d, p)
            return original["image"]

        b0 = None
        if {"b0", "b3", "p1"} & set(kinds):
            b0 = cached("b0", name, [row["sha256"]], lambda: [load_original()], render(templates["b0"], **values),
                        answer_parse, retry, may_generate="b0" in kinds)
            if b0 is not None and "b0" in kinds:
                records.append(summarize("b0", row, b0, device))
        if "b3" in kinds and b0 is not None and b0["parsed"] is not None:
            b3 = cached("b3", name, [row["sha256"]], lambda: [load_original()],
                        render(templates["b3"], category=values["category"], answer=json.dumps(b0["parsed"])),
                        partial(parse_check, answer=b0["parsed"]), retry)
            if b3 is not None:
                records.append(summarize("b3", row, b3, device))
        if "b1" in kinds:
            ref = references[(d, c)]
            b1 = cached("b1", name, [ref["sha256"], row["sha256"]],
                        lambda: [load_image(data_root / d, ref["path"]), load_original()],
                        render(templates["b1"], **values), answer_parse, retry)
            if b1 is not None:
                records.append(summarize("b1", row, b1, device))
        if "b2" in kinds:
            inputs, fields = b2_inputs(load_original(), row["sha256"], map_of(name), scales[d])
            b2 = cached("b2", name, [h for _, h in inputs], lambda: [i for i, _ in inputs],
                        render(templates["b2"], **values), answer_parse, retry)
            if b2 is not None:
                records.append({**summarize("b2", row, b2, device), **fields})
        if "p1" in kinds:
            rec = {"kind": "p1", "dataset": d, "category": c, "path": p, "device": device}
            if (d, p) in library:
                rec["skipped"] = "main_seed_library"
            elif b0 is None:
                rec["skipped"] = "missing_b0"
            elif b0["parsed"] is None:
                rec["skipped"] = "unparsed_b0"
            else:
                feat = det["features"][name]

                def check(answer: dict) -> dict:
                    return verify(answer, feat["score"], det["thresholds"][f"{d}/{c}"], feat["cell_shares"],
                                  det["embeddings"][name], det["prototypes"][(d, c)], names,
                                  rules["location"]["min_share_in_claimed"])

                rec["p0"] = check(b0["parsed"])
                rec["requeried"] = rec["p0"]["failed"]
                if rec["requeried"]:
                    inputs, fields = requery_inputs(load_original(), map_of(name), rules["requery"])
                    rq = cached("requery", name, [h for _, h in inputs], lambda: [i for i, _ in inputs],
                                render(templates["requery"], **values), answer_parse, retry)
                    if rq is not None:
                        records.append({**summarize("requery", row, rq, device), **fields})
                        rec["requery_unparsed"] = rq["parsed"] is None
                        if rq["parsed"] is not None:
                            rec["p1"] = check(rq["parsed"])
            records.append(rec)

    shard = {
        "shard": args.shard, "session": cfg["session"], "kinds": session["kinds"], "device": device,
        "gpu": torch.cuda.get_device_name(0), "kernels": kernel_implementations(), "complete": complete,
        "assigned_images": assigned, "repeat_oom": stop["repeat_oom"], "oom": oom, "missing": missing,
        "request_sources": dict(sources),
        "max_memory_reserved": torch.cuda.max_memory_reserved(0),
        "max_memory_allocated": torch.cuda.max_memory_allocated(0),
        "total_memory": torch.cuda.get_device_properties(0).total_memory,
        "pytorch_cuda_alloc_conf": alloc_conf,
        "prompts": {k: v for k, v in vlm["prompts"].items() if k != "dir"},
        "detector": det["provenance"], "records": records,
    }
    out = Path(cfg["output_dir"]) / "vlm_shards"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"shard_{args.shard}.json").write_text(json.dumps(shard) + "\n")
    print(f"shard {args.shard}: {assigned} images, {len(records)} records, complete={complete}")
    sys.exit(4 if stop["repeat_oom"] else 0)


if __name__ == "__main__":
    main()
