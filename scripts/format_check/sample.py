"""Draw the format check sample and B1 references locally, record the list, and pack those images for Kaggle."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.library.split import MaskValueError, check_type_names, format_check_sample
from clave_ad.runinfo import environment, file_sha256, header


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/format_check/qwen3_5_9b.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    types = yaml.safe_load(Path(cfg["data"]["type_names"]).read_text())

    problems = check_type_names(cfg, types)
    if problems:
        sys.exit("type_names.yaml does not match the data:\n" + "\n".join(problems))

    roots = {"mvtec_ad": Path(cfg["data"]["mvtec_root"]), "visa": Path(cfg["data"]["visa_root"])}
    try:
        rows = format_check_sample(cfg, types)
    except MaskValueError as e:
        # stop before drawing; only the kinds of unexpected values are reported
        sys.exit(f"stopped, no sample drawn: {e}")
    for row in rows:
        row["sha256"] = file_sha256(roots[row["dataset"]] / row["path"])

    result = header("samples", config_path, "mvtec_ad+visa", cfg["sampling"]["sample_seed"])
    samples = [r for r in rows if r["role"] == "sample"]
    result["metrics"] = {
        "sample_images": len(samples),
        "reference_images": len(rows) - len(samples),
        "per_dataset": {d: sum(r["dataset"] == d for r in samples) for d in roots},
    }
    result["environment"] = environment()
    result["images"] = rows
    out = Path(cfg["output_dir"]) / f"{result['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")

    # private Kaggle dataset: only the listed images, under <dataset>/<path>, plus the list itself
    pack = Path(cfg["kaggle_data_dir"])
    shutil.rmtree(pack, ignore_errors=True)
    for row in rows:
        target = pack / row["dataset"] / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(roots[row["dataset"]] / row["path"], target)
    shutil.copy2(out, pack / "samples.json")
    print(f"{len(samples)} samples, {len(rows) - len(samples)} references -> {out}, packed in {pack}")


if __name__ == "__main__":
    main()
