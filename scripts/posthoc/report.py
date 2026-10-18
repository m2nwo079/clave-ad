"""Local report of the post-hoc c4 experiment (configs/posthoc/c4_cause.yaml), apart from the main results.

Reads the kernel shards and cache bundle (E1, E2), the main-run cache (E3 and the B0 answers of the controls), and
ground truth through clave_ad.eval.claims only (E3). The result holds counts and format records, no answer content.
Run with --self-check to test the helpers on made-up data only.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "analysis")]
from answers import Requests, load
from clave_ad.eval.claims import GroundTruth, cell_boxes, judge
from clave_ad.runinfo import environment, file_sha256, header
from clave_ad.vlm.parse import has_think_block, parse_answer, unwrap
from clave_ad.vlm.records import compliance
from grid import boundaries

CELL_NUMBERS = re.compile(r"r(\d+)c(\d+)")


def first_output_class(error: str | None, parsed: dict | None, violation_error: str) -> str:
    """E1 first output: cleared (normal or in-grid answer), out of grid, or another format error."""
    if parsed is not None:
        return "cleared_normal" if not parsed["is_defective"] else "cleared_in_grid"
    return "out_of_grid" if error == violation_error else "other_error"


def fix_cells(text: str, candidates: list[str], clip: list[int]) -> dict | None:
    """E3: clip row and column numbers to the grid and drop repeats; None when the answer still fails to parse.

    Only the location_cells range is repaired, so an answer with any other error stays unparsed.
    """
    if has_think_block(text):
        return None
    try:
        obj = json.loads(unwrap(text)[0], strict=False)
    except json.JSONDecodeError:
        return None
    cells = obj.get("location_cells") if isinstance(obj, dict) else None
    if not isinstance(cells, list) or not all(isinstance(c, str) and CELL_NUMBERS.fullmatch(c) for c in cells):
        return None
    low, high = clip
    fixed = []
    for c in cells:
        r, k = (min(max(int(n), low), high) for n in CELL_NUMBERS.fullmatch(c).groups())
        if f"r{r}c{k}" not in fixed:
            fixed.append(f"r{r}c{k}")
    return parse_answer(json.dumps({**obj, "location_cells": fixed}), candidates)[0]


def grid_matches(size: tuple[int, int], grid: int) -> bool:
    """The drawn line positions are the inner edges of eval/claims.py cell_boxes."""
    boxes = cell_boxes(size, grid)
    xs = [boxes[f"r1c{j + 1}"][0] for j in range(1, grid)]
    ys = [boxes[f"r{i + 1}c1"][1] for i in range(1, grid)]
    return xs == boundaries(size[0], grid) and ys == boundaries(size[1], grid)


def layout_counts(answers: list[dict | None], min_columns: int) -> dict:
    """E2 counts only: parsed answers, the column distribution and the share with at least min_columns."""
    parsed = [a for a in answers if a is not None]
    wide = sum(a["columns"] >= min_columns for a in parsed)
    return {"n": len(answers), "parsed": len(parsed), "columns": dict(sorted(Counter(a["columns"] for a in parsed)
                                                                             .items())),
            "columns_at_least_min": wide, "share": wide / len(parsed) if parsed else None}


def changes(e1: dict, b0: dict) -> dict[str, bool]:
    """E1 against the main-run B0 answer of a control image: judgement, claimed cell set and type."""
    return {"is_defective": e1["is_defective"] != b0["is_defective"],
            "location_cells": set(e1["location_cells"]) != set(b0["location_cells"]),
            "defect_type": e1["defect_type"] != b0["defect_type"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/posthoc/c4_cause.yaml")
    parser.add_argument("--targets", help="results/posthoc/targets_*.json")
    parser.add_argument("--download", help="folder the kernel output was downloaded to")
    parser.add_argument("--visa-class-ids", default="configs/main_run/visa_class_ids.yaml")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    cfg = yaml.safe_load(Path(args.config).read_text())
    main_run = yaml.safe_load(Path(cfg["main_run"]).read_text())
    vlm = yaml.safe_load(Path(main_run["vlm"]).read_text())
    rules = yaml.safe_load(Path(main_run["rules"]).read_text())
    e1_cfg, e3_cfg, hyp = cfg["conditions"]["e1"], cfg["conditions"]["e3"], cfg["hypotheses"]
    targets = json.loads(Path(args.targets).read_text())["images"]
    download = Path(args.download)
    shards = sorted((download / cfg["output_dir"] / "posthoc_shards").glob("shard_*.json"))
    records = [r for s in shards for r in json.loads(s.read_text())["records"]]
    bundle_path = download / "cache" / "posthoc_bundle" / "cache.jsonl"
    with open(bundle_path) as f:
        bundle = {item["key"]: item["entry"] for item in map(json.loads, f)}
    requests = Requests(main_run)
    main_answers, _, _ = load(cfg["sessions"], requests)

    # grid lines against eval/claims.py cell boxes, on every target image size (image headers only)
    roots = {"mvtec_ad": Path(vlm["data"]["mvtec_root"]), "visa": Path(vlm["data"]["visa_root"])}
    sizes = {Image.open(roots[t["dataset"]] / t["path"]).size for t in targets}
    grid_check = all(grid_matches(s, e1_cfg["grid"]["size"]) for s in sizes)

    by_kind = defaultdict(list)
    for r in records:
        by_kind[(r["kind"], r["role"])].append(r)
    metrics = {"grid_matches_claims_cell_boxes": grid_check, "e1": {}, "e2": {}}
    for role in ("violation", "control"):
        e1 = by_kind[("e1", role)]
        first = Counter()
        final = Counter()
        changed, compared, skipped = Counter(), 0, 0
        for r in e1:
            entry = bundle[r["cache_key"]]
            attempt = entry["attempts"][0]
            parsed_first = entry["parsed"] if len(entry["attempts"]) == 1 else None
            first[first_output_class(attempt["error"], parsed_first, cfg["violation_error"])] += 1
            final["parsed" if entry["parsed"] is not None else
                  "out_of_grid" if entry["error"] == cfg["violation_error"] else "other_error"] += 1
            if role == "control" and entry["parsed"] is not None:
                b0 = main_answers.get(("b0", r["dataset"], r["path"]))
                if b0 is None or b0["parsed"] is None:
                    skipped += 1
                    continue
                compared += 1
                changed.update(k for k, v in changes(entry["parsed"], b0["parsed"]).items() if v)
        cleared = first["cleared_normal"] + first["cleared_in_grid"]
        metrics["e1"][role] = {"n": len(e1), "first_output": dict(first), "cleared": cleared,
                               "cleared_share": cleared / len(e1) if e1 else None,
                               "final_output_reference": dict(final), "compliance": compliance(e1)}
        if role == "control":
            metrics["e1"][role]["changes_vs_main_b0"] = {"compared": compared, "b0_unparsed_skipped": skipped,
                                                         **{k: changed[k] for k in cfg["reporting"]
                                                            ["e1_control_changes"]}}
        e2 = by_kind[("e2", role)]
        metrics["e2"][role] = {**layout_counts([bundle[r["cache_key"]]["parsed"] for r in e2],
                                               hyp["h1_layout"]["min_columns"]), "compliance": compliance(e2)}

    v1, v2, c2 = metrics["e1"]["violation"], metrics["e2"]["violation"], metrics["e2"]["control"]
    metrics["hypotheses"] = {
        "h2_grid": {"value": v1["cleared_share"],
                    "holds": v1["cleared_share"] is not None and v1["cleared_share"] > hyp["h2_grid"]["above"]},
        "h1_layout": {"value": v2["share"], "control": c2["share"],
                      "holds": v2["share"] is not None and v2["share"] > hyp["h1_layout"]["above"]
                      and (not hyp["h1_layout"]["above_control"]
                           or (c2["share"] is not None and v2["share"] > c2["share"]))}}

    # E3: repair the last output of each violation record and judge it against ground truth (eval/ only)
    gt = GroundTruth(vlm["data"], requests.types, yaml.safe_load(Path(args.visa_class_ids).read_text()),
                     rules["location"]["grid"], rules["evaluation"]["gt_cell_min_pixels"])
    e3 = defaultdict(Counter)
    for (kind, d, p), a in sorted(main_answers.items()):
        r = a["record"]
        if r["ok"] or r["final_error"] != cfg["violation_error"]:
            continue
        b0 = main_answers.get(("b0", d, p))
        text = requests.entry(requests.key(r, b0["parsed"] if b0 else None))["attempts"][-1]["text"]
        fixed = fix_cells(text, requests.candidates(d, r["category"]), e3_cfg["clip"])
        e3[kind]["records"] += 1
        if fixed is not None:
            e3[kind]["fixable"] += 1
            verdict = judge(fixed, gt.truth(d, p, r["category"]), rules["evaluation"]["location_correct_share"])
            e3[kind]["full_faithful_after_fix"] += verdict["full"]
    metrics["e3"] = {k: dict(v) for k, v in sorted(e3.items())}

    result = header("posthoc", Path(args.config), "visa", 0)
    result["provenance"] = {"targets": {"path": args.targets, "sha256": file_sha256(Path(args.targets))},
                            "bundle": {"sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
                                       "entries": len(bundle)},
                            "shards": [{"shard": json.loads(s.read_text())["shard"], "sha256": file_sha256(s)}
                                       for s in shards]}
    result["metrics"] = metrics
    result["environment"] = environment()
    # format records without raw text or answer content (design doc 9.2)
    result["records"] = [{k: v for k, v in r.items() if k != "cache_key"} for r in records]
    out = Path(cfg["output_dir"]) / f"{result['run_id']}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"-> {out}")


def _self_check() -> None:
    err = '"location_cells" must be a list of cells from "r1c1" to "r3c3"'
    assert first_output_class(None, {"is_defective": False}, err) == "cleared_normal"
    assert first_output_class(None, {"is_defective": True}, err) == "cleared_in_grid"
    assert first_output_class(err, None, err) == "out_of_grid"
    assert first_output_class("not a single JSON object", None, err) == "other_error"

    cands = ["chip", "crack"]
    base = {"is_defective": True, "defect_type": "chip", "confidence": 0.7, "rationale": "x"}
    # r2c4 becomes r2c3, which repeats and is dropped; r4c1 is clipped too
    out = fix_cells(json.dumps({**base, "location_cells": ["r2c3", "r2c4", "r4c1"]}), cands, [1, 3])
    assert out["location_cells"] == ["r2c3", "r3c1"]
    # a fenced answer is repaired like the parser reads it
    assert fix_cells("```json\n" + json.dumps({**base, "location_cells": ["r1c4"]}) + "\n```", cands, [1, 3])
    # other errors are not repaired: unknown type, non-cell string, think tags, broken JSON
    assert fix_cells(json.dumps({**base, "defect_type": "dent", "location_cells": ["r1c4"]}), cands, [1, 3]) is None
    assert fix_cells(json.dumps({**base, "location_cells": ["top left"]}), cands, [1, 3]) is None
    assert fix_cells("<think></think>" + json.dumps({**base, "location_cells": ["r1c4"]}), cands, [1, 3]) is None
    assert fix_cells('{"is_defective": true,', cands, [1, 3]) is None

    # grid lines match the evaluation cell boxes, including sizes not divisible by 3
    assert all(grid_matches(s, 3) for s in [(1284, 1168), (1000, 1000), (1500, 1000), (7, 11)])

    counts = layout_counts([{"columns": 4}, {"columns": 2}, None, {"columns": 5}], 4)
    assert counts == {"n": 4, "parsed": 3, "columns": {2: 1, 4: 1, 5: 1}, "columns_at_least_min": 2, "share": 2 / 3}
    assert layout_counts([None], 4)["share"] is None

    b0 = {"is_defective": True, "defect_type": "chip", "location_cells": ["r1c1", "r2c2"]}
    assert changes({**b0, "location_cells": ["r2c2", "r1c1"]}, b0) == {
        "is_defective": False, "location_cells": False, "defect_type": False}
    assert changes({"is_defective": False, "defect_type": None, "location_cells": []}, b0) == {
        "is_defective": True, "location_cells": True, "defect_type": True}

    # E3 judging of a repaired answer on made-up ground truth (no files)
    fixed = fix_cells(json.dumps({**base, "location_cells": ["r1c4"]}), cands, [1, 3])
    assert judge(fixed, {"defective": True, "cells": {"r1c3"}, "types": {"chip"}}, 0.5)["full"] is True
    assert judge(fixed, {"defective": False, "cells": set(), "types": set()}, 0.5)["full"] is False
    print("posthoc report self-check ok")


if __name__ == "__main__":
    main()
