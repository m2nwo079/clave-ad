"""Target images of the post-hoc c4 experiment (configs/posthoc/c4_cause.yaml), from format records only.

Violation images: any record with the location_cells range error as its final error. Controls: per category as many
main-seed evaluation images without a violation, in path order and evenly spaced. No ground truth is read.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.runinfo import environment, header


def evenly_spaced(paths: list[str], n: int) -> list[str]:
    # same selection as the smoke session in scripts/main_run/run.py session_images
    group = sorted(paths)
    return group[::max(1, len(group) // n)][:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/posthoc/c4_cause.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main_run = yaml.safe_load(Path(cfg["main_run"]).read_text())
    split = json.loads(Path(main_run["split_result"]).read_text())
    rows = {(r["dataset"], r["path"]): r for r in split["vlm_images"]}

    kinds = defaultdict(set)
    for path in cfg["sessions"]:
        for r in json.loads(Path(path).read_text())["records"]:
            if r["kind"] != "p1" and not r["ok"] and r["final_error"] == cfg["violation_error"]:
                kinds[(r["dataset"], r["path"])].add(r["kind"])

    per_category = defaultdict(int)
    for d, p in kinds:
        per_category[(d, rows[(d, p)]["category"])] += 1
    pool = defaultdict(list)
    for name in split["eval_images"]["0"]:
        d, p = name.split("/", 1)
        if (d, rows[(d, p)]["category"]) in per_category and (d, p) not in kinds:
            pool[(d, rows[(d, p)]["category"])].append(p)

    images = [{"role": "violation", **rows[k], "kinds": sorted(v)} for k, v in sorted(kinds.items())]
    for (d, c), n in sorted(per_category.items()):
        picked = evenly_spaced(pool[(d, c)], n)
        if len(picked) < n:
            sys.exit(f"not enough control images in {d}/{c}")
        images += [{"role": "control", **rows[(d, p)]} for p in picked]

    result = header("targets", Path(args.config), "+".join(sorted({d for d, _ in kinds})), 0)
    result["metrics"] = {role: {f"{d}/{c}": sum(1 for i in images if i["role"] == role and i["dataset"] == d
                                                and i["category"] == c) for d, c in sorted(per_category)}
                         for role in ("violation", "control")}
    result["environment"] = environment()
    result["images"] = images
    out = Path(cfg["output_dir"]) / f"{result['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"{len(images)} images -> {out}")


if __name__ == "__main__":
    main()
