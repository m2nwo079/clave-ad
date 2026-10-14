"""Example library split (k and seeds from rules.yaml) and the image lists of the main run, recorded as a result file.

Format check sample defects are left out of the library pools and of every image list but the train normals.
Small types (A7), combined and Other are judged in eval/, so the lists here carry paths only.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.data.datasets import mvtec_categories, mvtec_train_normals, visa_train_normals
from clave_ad.library.split import (MaskValueError, check_visa_class_ids, library_split, single_type_pools,
                                    test_images)
from clave_ad.runinfo import environment, file_sha256, header


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main_run/split.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    rules = yaml.safe_load(Path(cfg["rules"]).read_text())["type"]
    data = yaml.safe_load(Path(cfg["vlm"]).read_text())["data"]
    types = yaml.safe_load(Path(data["type_names"]).read_text())
    class_ids = yaml.safe_load(Path(cfg["visa_class_ids"]).read_text())
    roots = {"mvtec_ad": Path(data["mvtec_root"]), "visa": Path(data["visa_root"])}

    problems = check_visa_class_ids(types, class_ids)
    if problems:
        sys.exit("visa_class_ids.yaml does not match type_names.yaml:\n" + "\n".join(problems))
    samples = json.loads(Path(cfg["format_check_samples"]).read_text())["images"]
    exclude = {(r["dataset"], r["path"]) for r in samples if r["role"] == "sample" and r["split"] == "test"}
    try:
        pools = single_type_pools(data, types, class_ids, exclude)
    except MaskValueError as e:
        # stop before drawing; only the kinds of unexpected values are reported
        sys.exit(f"stopped, no library drawn: {e}")
    library = library_split(pools, rules["library_per_type"], rules["library_seeds"])

    vlm_images = [r for r in test_images(data) if (r["dataset"], r["path"]) not in exclude]
    train = [{"dataset": "mvtec_ad", "category": c, "path": p}
             for c in mvtec_categories(roots["mvtec_ad"]) for p in mvtec_train_normals(roots["mvtec_ad"], c)]
    train += [{"dataset": "visa", "category": c, "path": p} for c in sorted(class_ids["categories"])
              for p in visa_train_normals(Path(data["visa_split_csv"]), c)]
    for r in vlm_images + train:
        r["sha256"] = file_sha256(roots[r["dataset"]] / r["path"])
    hashes = {(r["dataset"], r["path"]): r["sha256"] for r in vlm_images}
    for rows in library.values():
        for r in rows:
            r["sha256"] = hashes[(r["dataset"], r["path"])]

    result = header("library", config_path, "mvtec_ad+visa", rules["library_main_seed"])
    datasets = list(roots)
    result["metrics"] = {
        "vlm_images": {d: sum(r["dataset"] == d for r in vlm_images) for d in datasets},
        "train_normals": {d: sum(r["dataset"] == d for r in train) for d in datasets},
        "excluded_format_check": {d: sum(e[0] == d for e in exclude) for d in datasets},
        "types": {d: sum(key[0] == d for key in pools) for d in datasets},
        "single_type_candidates": {f"{d}/{c}/{t}": len(p) for (d, c, t), p in sorted(pools.items())},
        "library_images": {str(s): {d: sum(r["dataset"] == d for r in rows) for d in datasets}
                           for s, rows in library.items()},
        "types_below_per_type": sorted(f"{d}/{c}/{t}" for (d, c, t), p in pools.items()
                                       if len(p) < rules["library_per_type"]),
    }
    result["environment"] = environment()
    result["library"] = {str(s): rows for s, rows in library.items()}
    # per seed: the test images left after removing that seed's library
    taken = {s: {(x["dataset"], x["path"]) for x in rows} for s, rows in library.items()}
    result["eval_images"] = {str(s): [f"{r['dataset']}/{r['path']}" for r in vlm_images
                                      if (r["dataset"], r["path"]) not in taken[s]] for s in library}
    result["vlm_images"] = vlm_images
    result["train_normals"] = train
    out = Path(cfg["output_dir"]) / f"{result['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{len(vlm_images)} test images, {len(train)} train normals, library "
          f"{ {s: len(rows) for s, rows in library.items()} } -> {out}")


if __name__ == "__main__":
    main()
