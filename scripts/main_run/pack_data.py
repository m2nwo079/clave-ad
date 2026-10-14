"""Pack the main run images for the private Kaggle data dataset as hard links: train normals and the VLM test images.

Only image files listed in the split result are linked, so no ground-truth mask or label file is packed. The image
list and hashes stay in the split result (in the code dataset); images.json only names it and marks the data root.
The packed files are hard links to data/, so they are never written to.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.runinfo import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main_run/split.yaml")
    parser.add_argument("--detector-config", default="configs/main_run/detector.yaml", help="names the split result")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data = yaml.safe_load(Path(cfg["vlm"]).read_text())["data"]
    roots = {"mvtec_ad": Path(data["mvtec_root"]), "visa": Path(data["visa_root"])}
    split_path = Path(yaml.safe_load(Path(args.detector_config).read_text())["split_result"])
    split = json.loads(split_path.read_text())

    pack = Path(cfg["kaggle_data_dir"])
    # removes only the earlier links; data/ keeps its files
    shutil.rmtree(pack, ignore_errors=True)
    rows = split["train_normals"] + split["vlm_images"]
    for r in rows:
        source, target = roots[r["dataset"]] / r["path"], pack / r["dataset"] / r["path"]
        if file_sha256(source) != r["sha256"]:
            sys.exit(f"image does not match the split result: {r['dataset']} {r['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    stamp = {"split_result": str(split_path), "split_run_id": split["run_id"], "images": len(rows)}
    (pack / "images.json").write_text(json.dumps(stamp, indent=2) + "\n")
    print(f"{len(rows)} images linked -> {pack}")


if __name__ == "__main__":
    main()
