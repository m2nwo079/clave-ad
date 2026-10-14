"""Copy the anomaly maps of the format check sample images into the Kaggle data folder, with their SHA-256 list.

Run after sample.py, which clears the data folder. The worker checks every map against maps.json before use.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.runinfo import file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/format_check/qwen3_5_9b.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    detector_result = Path(cfg["detector"]["result"])
    detector = json.loads(detector_result.read_text())
    cache_dir = Path(yaml.safe_load(Path(detector["config_path"]).read_text())["cache_dir"])

    pack = Path(cfg["kaggle_data_dir"]) / "maps"
    shutil.rmtree(pack, ignore_errors=True)
    pack.mkdir(parents=True)
    maps = {}
    for r in detector["records"]:
        if r["role"] != "sample":
            continue
        source = cache_dir / f"{r['cache_key']}.npz"
        shutil.copy2(source, pack / source.name)
        maps[r["cache_key"]] = file_sha256(source)
    index = {"detector_result": str(detector_result), "detector_run_id": detector["run_id"], "sha256": maps}
    (pack / "maps.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"{len(maps)} maps -> {pack}")


if __name__ == "__main__":
    main()
