"""Merge detector shards: per-category existence thresholds, per-seed type prototypes, and the files the VLM run
and the local evaluation read (maps.json, features.json, embeddings.npz, prototypes.npz), with their hashes.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.runinfo import environment, file_sha256, header


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main_run/detector.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    det_path = Path(cfg["detector_config"])
    det = yaml.safe_load(det_path.read_text())["detector"]
    rules = yaml.safe_load(Path(cfg["rules"]).read_text())
    split = json.loads(Path(cfg["split_result"]).read_text())
    cache_dir, out_dir = Path(cfg["cache_dir"]), Path(cfg["output_dir"])
    shards = [json.loads(p.read_text()) for p in sorted((out_dir / "detector_shards").glob("shard_*.json"))]
    records = [r for s in shards for r in s["records"]]
    complete = bool(shards) and all(s["complete"] for s in shards)

    # existence threshold per category from train normal scores
    train = defaultdict(list)
    for r in records:
        if r["kind"] == "train":
            train[(r["dataset"], r["category"])].append(r["score"])
    percentile = rules["existence"]["threshold_percentile"]
    thresholds = {f"{d}/{c}": {"threshold": float(np.percentile(v, percentile)), "images": len(v)}
                  for (d, c), v in sorted(train.items())}

    # test image features, maps and embeddings, keyed by "<dataset>/<path>"
    tests = sorted((r for r in records if r["kind"] == "test"), key=lambda r: (r["dataset"], r["path"]))
    feats, embeddings, maps = {}, [], {}
    for r in tests:
        name = f"{r['dataset']}/{r['path']}"
        feats[name] = {"cache_key": r["cache_key"], "category": r["category"]}
        feats[name] |= json.loads((cache_dir / "features" / f"{r['cache_key']}.json").read_text())
        npz = cache_dir / "maps" / f"{r['cache_key']}.npz"
        maps[r["cache_key"]] = file_sha256(npz)
        with np.load(npz) as entry:
            embeddings.append(entry["embedding"])
    names = list(feats)
    matrix = np.stack(embeddings) if embeddings else np.zeros((0, 0), dtype=np.float32)
    index = {n: i for i, n in enumerate(names)}

    # per seed: prototype = normalised mean of the normalised library embeddings of each type
    prototypes, counts = {}, {}
    for seed, rows in split["library"].items():
        groups = defaultdict(list)
        for r in rows:
            name = f"{r['dataset']}/{r['path']}"
            if name in index:
                groups[f"{r['dataset']}/{r['category']}/{r['type']}"].append(matrix[index[name]])
        for t, vectors in sorted(groups.items()):
            mean = np.mean(vectors, axis=0)
            prototypes[f"{seed}|{t}"] = mean / np.linalg.norm(mean)
            counts[f"{seed}|{t}"] = len(vectors)

    files = {"maps.json": {"sha256": maps}, "features.json": feats}
    for name, content in files.items():
        (cache_dir / name).write_text(json.dumps(content, indent=1) + "\n")
    np.savez(cache_dir / "embeddings.npz", names=np.array(names), embeddings=matrix)
    # "<seed>|<dataset>/<category>/<type>" keys beside a matrix, since npz member names should stay plain
    np.savez(cache_dir / "prototypes.npz", keys=np.array(list(prototypes)),
             vectors=np.stack(list(prototypes.values())) if prototypes else np.zeros((0, 0), dtype=np.float32))

    timed = [r["seconds"] for r in records if r["seconds"] is not None]
    result = header("detector", config_path, "mvtec_ad+visa", cfg["seed"])
    result["detector"] = {
        "config": str(det_path), "repo": det["repo"], "commit": det["commit"], "checkpoints": det["checkpoints"],
        "settings": det["settings"], "loaded": {d: v for s in shards for d, v in s["loaded"].items()},
        "import_workarounds": shards[0]["import_workarounds"] if shards else None,
        "embedding_dtype": next((s["embedding_dtype"] for s in shards if s["embedding_dtype"]), None),
        "split_result": cfg["split_result"],
    }
    result["files"] = {name: file_sha256(cache_dir / name)
                       for name in ("maps.json", "features.json", "embeddings.npz", "prototypes.npz")}
    result["metrics"] = {
        "complete": complete,
        "images": {k: sum(r["kind"] == k for r in records) for k in ("train", "test")},
        "expected": {"train": len(split["train_normals"]), "test": len(split["vlm_images"])},
        "nonfinite": sum(not r["finite"] for r in records),
        "computed": len(timed),
        "seconds_per_image": sum(timed) / len(timed) if timed else None,
        "max_memory_allocated_gb": [s["max_memory_allocated"] / 1e9 for s in shards],
        "thresholds": thresholds,
        "prototype_images": counts,
    }
    result["environment"] = environment()
    out = out_dir / f"{result['run_id']}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{len(records)} images (complete={complete}) -> {out}")
    sys.exit(0 if complete else 3)


if __name__ == "__main__":
    main()
