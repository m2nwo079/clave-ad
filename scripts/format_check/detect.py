"""AnomalyCLIP score and anomaly map for every format check image, and the fixed B2 colour scale.

Each image is cached under cache_dir by a hash of everything that defines its output, so a rerun skips finished work.
Maps are stored at the detector resolution; resizing to the original size is left to the code that uses them.
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
from clave_ad.detectors.anomalyclip import AnomalyCLIP, import_official
from clave_ad.runinfo import environment, file_sha256, header

PERCENTILES = (1, 50, 99)


def summary(values: np.ndarray) -> dict:
    return {"min": float(values.min()), **{f"p{q}": float(np.percentile(values, q)) for q in PERCENTILES},
            "max": float(values.max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/format_check/anomalyclip.yaml")
    parser.add_argument("--data-root", required=True, help="folder holding samples.json and <dataset>/<path> images")
    parser.add_argument("--repo", required=True, help="AnomalyCLIP checkout at the configured commit")
    parser.add_argument("--clip-dir", required=True, help="download folder for the CLIP backbone")
    args = parser.parse_args()
    config_path, data_root, repo = Path(args.config), Path(args.data_root), Path(args.repo)
    cfg = yaml.safe_load(config_path.read_text())
    det, settings = cfg["detector"], cfg["detector"]["settings"]
    torch.manual_seed(cfg["seed"])

    for dataset, ckpt in det["checkpoints"].items():
        if file_sha256(repo / ckpt["path"]) != ckpt["sha256"]:
            sys.exit(f"{dataset} checkpoint does not match its configured sha256")
    workarounds = import_official(repo)
    rows = json.loads((data_root / "samples.json").read_text())["images"]
    if any("split" not in r for r in rows):
        sys.exit("samples.json has no split field; redraw the sample and upload the new data version")
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    records, maps, seconds, loaded = [], [], [], {}
    for dataset, ckpt in det["checkpoints"].items():
        model = None
        for row in (r for r in rows if r["dataset"] == dataset):
            image_path = data_root / dataset / row["path"]
            if file_sha256(image_path) != row["sha256"]:
                sys.exit(f"image does not match samples.json: {dataset} {row['path']}")
            key = hashlib.sha256(json.dumps({"commit": det["commit"], "checkpoint": ckpt["sha256"],
                                             "settings": settings, "image": row["sha256"]},
                                            sort_keys=True).encode()).hexdigest()
            path = cache_dir / f"{key}.npz"
            cached = path.exists()
            if not cached:
                if model is None:
                    model = AnomalyCLIP(repo / ckpt["path"], settings, "cuda", Path(args.clip_dir))
                    loaded[dataset] = {"model_dtype": model.dtype, "checkpoint_torch_load": model.torch_load}
                start = time.perf_counter()
                score, anomaly_map = model(load_image(data_root / dataset, row["path"]))
                seconds.append(time.perf_counter() - start)
                # write then rename, so an interrupted run never leaves a partial cache entry
                tmp = cache_dir / f"{key}.tmp.npz"
                np.savez(tmp, anomaly_map=anomaly_map.astype(np.float32), score=np.float32(score))
                tmp.rename(path)
            with np.load(path) as entry:
                anomaly_map, score = entry["anomaly_map"], float(entry["score"])
            maps.append(anomaly_map)
            records.append({**{k: row[k] for k in ("dataset", "category", "role", "split", "path", "sha256")},
                            "cache_key": key, "cached": cached, "score": score,
                            "finite": bool(np.isfinite(anomaly_map).all() and np.isfinite(score)),
                            "map": summary(anomaly_map)})

    # one fixed scale per dataset (each uses its own checkpoint), from the pooled pixels of its train normal sample maps
    scale, b2_scale = cfg["b2_scale"], {}
    for dataset in det["checkpoints"]:
        normal = [m.ravel() for m, r in zip(maps, records)
                  if r["dataset"] == dataset and r["role"] == "sample" and r["split"] == "train"]
        pixels = np.concatenate(normal)
        low, high = np.percentile(pixels, [scale["low_percentile"], scale["high_percentile"]])
        b2_scale[dataset] = {
            "source": "train normal sample images", "images": len(normal),
            "low_percentile": scale["low_percentile"], "high_percentile": scale["high_percentile"],
            "low": float(low), "high": float(high), "high_end_pixel_fraction": float((pixels >= high).mean()),
        }
    groups = sorted({(r["dataset"], r["split"]) for r in records})

    result = header("anomalyclip", config_path, "mvtec_ad+visa", cfg["seed"])
    result["detector"] = {
        "repo": det["repo"], "commit": det["commit"], "settings": settings, "checkpoints": det["checkpoints"],
        "checkpoint_sha256_verified": True, "loaded": loaded, "import_workarounds": workarounds,
        "clip_file_sha256": {p.name: file_sha256(p) for p in sorted(Path(args.clip_dir).glob("*.pt"))},
    }
    result["metrics"] = {
        "images": len(records),
        "per_dataset": {d: sum(r["dataset"] == d for r in records) for d in det["checkpoints"]},
        "computed": len(seconds),
        "nonfinite": sum(not r["finite"] for r in records),
        "seconds_per_image": sum(seconds) / len(seconds) if seconds else None,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
        "map_values": {f"{d}/{s}": summary(np.concatenate([m.ravel() for m, r in zip(maps, records)
                                                          if (r["dataset"], r["split"]) == (d, s)]))
                       for d, s in groups},
        "b2_scale": b2_scale,
    }
    result["environment"] = environment()
    result["records"] = records
    out = Path(cfg["output_dir"]) / f"{result['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(records)} images ({len(seconds)} computed) -> {out}")


if __name__ == "__main__":
    main()
