"""Detector worker for one GPU: AnomalyCLIP scores for train normals, and for test images also the map, the
verification features and the CLIP embedding of the type crop.

Every image is cached under cache_dir by a hash of everything that defines its output, so a rerun skips finished
work. A worker stops cleanly when its time budget runs out; the shard file then says it is incomplete.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.data.datasets import load_image
from clave_ad.data.maps import cell_shares, peak, type_box, zoom_box
from clave_ad.detectors.anomalyclip import AnomalyCLIP, import_official
from clave_ad.detectors.clip_embed import ClipEmbedder
from clave_ad.runinfo import file_sha256


def features(anomaly_map: np.ndarray, size: tuple[int, int], rules: dict) -> dict:
    """Map-derived inputs of location and type verification, B4 and the zoom-in re-query, in original pixels."""
    loc, kind, rq = rules["location"], rules["type"], rules["requery"]
    tbox, tshift = type_box(anomaly_map, size, kind["crop_cells"], loc["grid"])
    zbox, zshift = zoom_box(anomaly_map, size, rq["top_fraction"], rq["box_scale"], rq["grid"])
    return {"size": list(size), "cell_shares": cell_shares(anomaly_map, loc["top_fraction"], loc["grid"]),
            "peak": list(peak(anomaly_map)), "type_box": list(tbox), "type_shift": list(tshift),
            "zoom_box": list(zbox), "zoom_shift": list(zshift)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main_run/detector.yaml")
    parser.add_argument("--data-root", required=True, help="folder holding <dataset>/<path> images")
    parser.add_argument("--repo", required=True, help="AnomalyCLIP checkout at the configured commit")
    parser.add_argument("--clip-dir", required=True, help="download folder for the CLIP backbone")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    det = yaml.safe_load(Path(cfg["detector_config"]).read_text())["detector"]
    rules = yaml.safe_load(Path(cfg["rules"]).read_text())
    split = json.loads(Path(cfg["split_result"]).read_text())
    data_root, repo = Path(args.data_root), Path(args.repo)
    torch.manual_seed(cfg["seed"])
    deadline = time.monotonic() + cfg["time_budget_hours"] * 3600

    for dataset, ckpt in det["checkpoints"].items():
        if file_sha256(repo / ckpt["path"]) != ckpt["sha256"]:
            sys.exit(f"{dataset} checkpoint does not match its configured sha256")
    workarounds = import_official(repo)
    cache_dir = Path(cfg["cache_dir"])
    for sub in ("scores", "maps", "features"):
        (cache_dir / sub).mkdir(parents=True, exist_ok=True)
    # rule values that shape the features and the embedding, so a change gives new cache keys
    feature_rules = {"location": rules["location"], "type": {k: rules["type"][k] for k in ("crop_cells", "embedding")},
                     "requery": rules["requery"]}

    rows = [{**r, "kind": "train"} for r in split["train_normals"]] + [{**r, "kind": "test"} for r in split["vlm_images"]]
    mine = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    records, loaded, complete, embedder = [], {}, True, None
    torch.cuda.reset_peak_memory_stats()
    for dataset, ckpt in det["checkpoints"].items():
        model = None
        for row in (r for r in mine if r["dataset"] == dataset):
            if time.monotonic() > deadline:
                complete = False
                break
            test = row["kind"] == "test"
            spec = {"commit": det["commit"], "checkpoint": ckpt["sha256"], "settings": det["settings"],
                    "image": row["sha256"], "map": test, "features": feature_rules if test else None}
            key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
            done = (cache_dir / "features" / f"{key}.json") if test else (cache_dir / "scores" / f"{key}.json")
            seconds = None
            if not done.exists():
                image_path = data_root / dataset / row["path"]
                if file_sha256(image_path) != row["sha256"]:
                    sys.exit(f"image does not match the split result: {dataset} {row['path']}")
                if model is None:
                    model = AnomalyCLIP(repo / ckpt["path"], det["settings"], "cuda", Path(args.clip_dir))
                    loaded[dataset] = {"model_dtype": model.dtype, "checkpoint_torch_load": model.torch_load}
                start = time.perf_counter()
                image = load_image(data_root / dataset, row["path"])
                score, anomaly_map = model(image)
                entry = {"score": score, "finite": bool(np.isfinite(score) and np.isfinite(anomaly_map).all())}
                if test:
                    if embedder is None:
                        embedder = ClipEmbedder(rules["type"]["embedding"], "cuda", Path(args.clip_dir))
                    entry |= features(anomaly_map, image.size, rules)
                    embedding = embedder(image.crop(tuple(entry["type_box"])))
                    entry["finite"] &= bool(np.isfinite(embedding).all())
                    # write then rename, so an interrupted run never leaves a partial cache entry
                    tmp = cache_dir / "maps" / f"{key}.tmp.npz"
                    np.savez(tmp, anomaly_map=anomaly_map.astype(np.float32), embedding=embedding.astype(np.float32))
                    tmp.rename(cache_dir / "maps" / f"{key}.npz")
                seconds = time.perf_counter() - start
                tmp = done.with_suffix(".tmp")
                tmp.write_text(json.dumps(entry))
                tmp.rename(done)
            entry = json.loads(done.read_text())
            records.append({k: row[k] for k in ("dataset", "category", "path", "sha256", "kind")}
                           | {"cache_key": key, "seconds": seconds, "score": entry["score"], "finite": entry["finite"]})
        if not complete:
            break

    shard = {
        "shard": args.shard, "complete": complete, "assigned": len(mine), "gpu": torch.cuda.get_device_name(0),
        "loaded": loaded, "import_workarounds": workarounds,
        "embedding_dtype": embedder.dtype if embedder else None,
        "max_memory_allocated": torch.cuda.max_memory_allocated(0), "records": records,
    }
    out = Path(cfg["output_dir"]) / "detector_shards"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"shard_{args.shard}.json").write_text(json.dumps(shard) + "\n")
    print(f"shard {args.shard}: {len(records)}/{len(mine)} images, complete={complete}")


if __name__ == "__main__":
    main()
