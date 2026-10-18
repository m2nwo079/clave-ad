"""Ground-truth evaluation of the main run (design doc 12, 13, 13.1; configs/analysis/eval.yaml).

Answers come from the request cache, verifier inputs from the local detector output, and ground truth only through
clave_ad.eval.claims. Writes one result file with aggregates only (no per-image ground truth).
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from answers import Requests, load
from clave_ad.baselines.b4 import b4_answer
from clave_ad.eval.claims import GroundTruth, judge
from clave_ad.eval.stats import auroc, bootstrap, resample_indices
from clave_ad.runinfo import environment, file_sha256, header
from clave_ad.verifiers.verify import verify

METHODS = {"mvtec_ad": ["b0", "b1", "b2", "b3", "b4", "p0", "p1", "p1_secondary"],
           "visa": ["b0", "b2", "b4", "p0", "p1", "p1_secondary"]}
SIGNAL_METHODS = ("b0", "b1", "b2", "b3", "p0", "p1", "p1_secondary")


def rate(hits: int, n: int) -> dict:
    return {"correct": hits, "n": n, "rate": hits / n if n else None}


def b3_score(check: dict) -> float:
    """A1 on the self-check: share of true among the claims present (null is an absent claim)."""
    present = [v for v in check.values() if v is not None]
    return sum(present) / len(present)


def ablated_score(claims: dict, removed: list[str]) -> float | None:
    """Verification score without the claims of the removed tools; None when no verifiable claim is left."""
    kept = [v for k, v in claims.items() if k not in removed and isinstance(v, bool)]
    return sum(kept) / len(kept) if kept else None


def status(claims: dict) -> str:
    """Design doc 10.5 over verifiable claims (eval.yaml explanation_status)."""
    checked = [v for v in claims.values() if isinstance(v, bool)]
    return "all_verified" if all(checked) else "withheld" if not any(checked) else "partially_verified"


class Detector:
    def __init__(self, root: Path, main_run: dict, rules: dict) -> None:
        result = json.loads(Path(main_run["detector"]["result"]).read_text())
        for name in ("features.json", "embeddings.npz", "prototypes.npz"):
            if file_sha256(root / name) != result["files"][name]:
                sys.exit(f"local detector output {name} does not match {result['run_id']}")
        self.features = json.loads((root / "features.json").read_text())
        with np.load(root / "embeddings.npz") as e:
            self.embeddings = dict(zip(e["names"].tolist(), e["embeddings"]))
        self.prototypes = defaultdict(lambda: defaultdict(dict))
        with np.load(root / "prototypes.npz") as p:
            for key, vector in zip(p["keys"].tolist(), p["vectors"]):
                seed, name = key.split("|")
                dataset, category, kind = name.split("/", 2)
                self.prototypes[int(seed)][(dataset, category)][kind] = vector
        self.thresholds = {k: v["threshold"] for k, v in result["metrics"]["thresholds"].items()}
        self.seconds_per_image = result["metrics"].get("seconds_per_image")
        self.min_share = rules["location"]["min_share_in_claimed"]

    def verify(self, d: str, c: str, path: str, answer: dict, names: list[str], seed: int) -> dict:
        f, name = self.features[f"{d}/{path}"], f"{d}/{path}"
        return verify(answer, f["score"], self.thresholds[f"{d}/{c}"], f["cell_shares"], self.embeddings[name],
                      self.prototypes[seed][(d, c)], names, self.min_share)

    def b4(self, d: str, c: str, path: str, names: list[str], share: float) -> dict:
        f, name = self.features[f"{d}/{path}"], f"{d}/{path}"
        return b4_answer(f["score"], self.thresholds[f"{d}/{c}"], f["cell_shares"], self.embeddings[name],
                         self.prototypes[0][(d, c)], names, share)


class Table:
    """Per-image arrays of one dataset's main set: full faithfulness, parse state and signal score per method."""

    def __init__(self, rows: list[dict], methods: list[str]) -> None:
        self.rows, self.methods = rows, methods
        self.category = np.array([r["category"] for r in rows])
        self.small = np.array([r["small"] for r in rows], bool)
        self.defective = np.array([r["gt"]["defective"] for r in rows], bool)
        self.full = {m: np.array([bool(r["m"][m]["judge"] and r["m"][m]["judge"]["full"]) for r in rows])
                     for m in methods}
        self.score = {m: np.array([np.nan if r["m"][m].get("score") is None else r["m"][m]["score"] for r in rows])
                      for m in methods}

    def full_rate(self, m: str, idx: np.ndarray) -> float | None:
        keep = idx[~self.small[idx]]
        return float(self.full[m][keep].mean()) if len(keep) else None

    def auroc(self, m: str, idx: np.ndarray, fill: float | None = None,
              score: np.ndarray | None = None) -> float | None:
        """Main analysis over answers with a score; with fill, a missing score takes that value (secondary)."""
        s = self.score[m] if score is None else score
        keep = idx[~self.small[idx]]
        if fill is not None:
            return auroc(np.nan_to_num(s[keep], nan=fill), self.full[m][keep])
        keep = keep[~np.isnan(s[keep])]
        return auroc(s[keep], self.full[m][keep])


def pair_auroc(t: Table, a: str, b: str, indices: list[np.ndarray] | None, fill: float, level: float) -> dict:
    """AUROC of method a minus method b, each with its own signal and its own answer's full faithfulness.

    Main analysis: images where both answers have a score, with the images left out counted. Secondary analysis:
    missing scores set to fill over all images (eval.yaml comparisons, p0_vs_b3).
    """
    both = ~np.isnan(t.score[a]) & ~np.isnan(t.score[b])
    usable = ~t.small

    def values(idx: np.ndarray, main: bool) -> tuple:
        if main:
            keep = idx[both[idx] & usable[idx]]
            x, y = t.auroc(a, keep), t.auroc(b, keep)
        else:
            x, y = t.auroc(a, idx, fill=fill), t.auroc(b, idx, fill=fill)
        return x, y, (x - y if x is not None and y is not None else None)

    allidx = np.arange(len(t.rows))
    out = {}
    for label, main in (("main", True), ("secondary_fill", False)):
        x, y, diff = values(allidx, main)
        out[label] = {a: {"value": x}, b: {"value": y}, "diff": {"value": diff}}
    out["main"]["n_pairs"] = int((both & usable).sum())
    out["main"]["excluded"] = {f"{a}_missing": int((np.isnan(t.score[a]) & usable).sum()),
                               f"{b}_missing_only": int((~np.isnan(t.score[a]) & np.isnan(t.score[b])
                                                         & usable).sum())}
    if indices is not None:
        for key, pos in ((a, 0), (b, 1), ("diff", 2)):
            out["main"][key].update(bootstrap(lambda i: values(i, True)[pos], indices, level))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis/run.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    ev = yaml.safe_load(Path(cfg["eval"]).read_text())
    main_run = yaml.safe_load(Path(cfg["main_run"]).read_text())
    rules = yaml.safe_load(Path(ev["rules"]).read_text())
    vlm = yaml.safe_load(Path(main_run["vlm"]).read_text())
    split = json.loads(Path(main_run["split_result"]).read_text())
    requests = Requests(main_run)
    answers, p1_records, _ = load(cfg["sessions"], requests)
    det = Detector(Path(cfg["detector_dir"]), main_run, rules)
    evr, bs = rules["evaluation"], rules["evaluation"]["bootstrap"]
    gt_source = GroundTruth(vlm["data"], requests.types, yaml.safe_load(Path(cfg["visa_class_ids"]).read_text()),
                            rules["location"]["grid"], evr["gt_cell_min_pixels"])
    share = evr["location_correct_share"]

    # ground truth of every test image without the format check samples (A7 counts library images too)
    images = {(r["dataset"], r["path"]): r for r in split["vlm_images"]}
    gts = {k: gt_source.truth(k[0], k[1], r["category"]) for k, r in images.items()}
    type_images = Counter((d, images[(d, p)]["category"], t) for (d, p), g in gts.items() for t in g["types"])
    small_limit = rules["type"]["library_per_type"] + evr["small_type_margin"]
    small_types = {k for k, n in type_images.items() if n < small_limit}

    def answer_of(kind: str, d: str, p: str) -> dict | None:
        a = answers.get((kind, d, p))
        return a["parsed"] if a else None

    def method_entries(d: str, p: str, c: str, gt: dict, seed: int) -> dict:
        names = requests.candidates(d, c)
        out = {}
        b0 = answer_of("b0", d, p)
        p0 = det.verify(d, c, p, b0, names, seed) if b0 else None
        for kind in ("b0", "b1", "b2"):
            if kind in METHODS[d]:
                a = answer_of(kind, d, p)
                out[kind] = {"answer": a, "score": a["confidence"] if a else None}
                if a and kind != "b0":
                    out[kind]["verifier_score"] = det.verify(d, c, p, a, names, seed)["score"]
        if "b3" in METHODS[d]:
            check = answer_of("b3", d, p)
            out["b3"] = {"answer": b0, "score": b3_score(check) if b0 and check else None}
        out["b4"] = {"answer": det.b4(d, c, p, names, ev["b4"]["location_cumulative_share"])}
        out["p0"] = {"answer": b0, "score": p0["score"] if p0 else None, "verifier": p0}
        rec = p1_records.get((d, p), {})
        rq = answer_of("requery", d, p) if rec.get("requeried") else None
        p1 = det.verify(d, c, p, rq, names, seed) if rq is not None else None
        # the main-seed scores must match what the kernel recorded
        if seed == 0 and (rec.get("p0", {}).get("score") != (p0 and p0["score"])
                          or rec.get("p1", {}).get("score") != (p1 and p1["score"])):
            sys.exit(f"verification score differs from the session record: {d}/{p}")
        if p1 is not None:
            out["p1"] = {"answer": rq, "score": p1["score"], "verifier": p1, "requeried": True, "kept": False}
            adopt = p1["score"] > p0["score"]
            out["p1_secondary"] = {"answer": rq if adopt else b0, "score": max(p1["score"], p0["score"]),
                                   "verifier": p1 if adopt else p0}
        else:
            # not re-queried, or the re-query never parsed and the original answer is kept
            requeried = bool(rec.get("requeried"))
            out["p1"] = {**out["p0"], "requeried": requeried, "kept": requeried}
            out["p1_secondary"] = dict(out["p0"])
        for m in out.values():
            m["judge"] = judge(m["answer"], gt, share) if m["answer"] else None
        return out

    def main_rows(d: str, seed: int) -> tuple[list[dict], dict]:
        """Main set rows of a seed: evaluation images without combined and Other images (reported apart)."""
        rows, apart = [], defaultdict(list)
        for name in split["eval_images"][str(seed)]:
            dd, p = name.split("/", 1)
            if dd != d:
                continue
            c, gt = images[(d, p)]["category"], gts[(d, p)]
            row = {"path": p, "category": c, "gt": gt, "m": method_entries(d, p, c, gt, seed),
                   "small": any((d, c, t) in small_types for t in gt["types"])}
            (apart[gt["separate"]] if gt["separate"] else rows).append(row)
        return rows, apart

    fill = ev["unparsed"]["secondary_fill"]

    def a2(rows: list[dict], indices: list[np.ndarray] | None) -> dict:
        """A2: P0 minus the B3 self-check (p0_vs_b3); judged only when intervals are computed (seed 0)."""
        out = pair_auroc(Table(rows, ["p0", "b3"]), "p0", "b3", indices, fill, bs["interval"])
        if indices is not None:
            ci1, ci2 = out["main"]["p0"]["ci"], out["main"]["diff"]["ci"]
            out["main"]["holds"] = bool(ci1 and ci2 and ci1[0] > 0.5 and ci2[0] > 0)
        return out

    result = header("eval", Path(args.config), "mvtec_ad+visa", bs["seed"])
    result["provenance"] = provenance(cfg, ev, main_run)
    datasets, separate, unverifiable, ablation, requery, cost, fmt, comparisons = {}, {}, {}, {}, {}, {}, {}, {}
    library_miss, small_report, a2_result = {}, {}, {}
    for d in METHODS:
        rows, apart = main_rows(d, 0)
        t = Table(rows, METHODS[d])
        indices = resample_indices(t.category, bs["resamples"], bs["seed"])
        allidx = np.arange(len(rows))
        defective_rows = [r for r in rows if r["gt"]["defective"]]
        datasets[d] = {"images": len(rows), "small_type_images": int(t.small.sum()),
                       "multi_type_ratio": rate(sum(len(r["gt"]["types"]) > 1 for r in defective_rows),
                                                len(defective_rows))}
        for m in METHODS[d]:
            entries = [r["m"][m] for r in rows]
            parsed = [e for e in entries if e["answer"]]
            said = [(r, r["m"][m]) for r in rows if r["m"][m]["answer"] and r["m"][m]["answer"]["is_defective"]]
            tp = [(r, e) for r, e in said if r["gt"]["defective"]]
            tp_typed = [(r, e) for r, e in tp if not r["small"]]
            block = {
                "unparsed": len(entries) - len(parsed),
                "full_faithful": {"rate": t.full_rate(m, allidx), "n": int((~t.small).sum()),
                                  **bootstrap(lambda i: t.full_rate(m, i), indices, bs["interval"])},
                "claims": {"existence": rate(sum(e["judge"]["existence"] for e in parsed), len(parsed)),
                           "location": rate(sum(e["judge"]["location"] for _, e in tp), len(tp)),
                           "type": rate(sum(e["judge"]["type"] for _, e in tp_typed), len(tp_typed))},
                "sq1_2": {"n": len(tp_typed), "wrong": sum(not e["judge"]["full"] for _, e in tp_typed)},
                "normal_called_defective": sum(not r["gt"]["defective"] for r, _ in said),
                "cells": {"mean": float(np.mean([len(e["answer"]["location_cells"]) for _, e in said]))
                          if said else None, "empty": sum(not e["answer"]["location_cells"] for _, e in said)},
                "per_category": {c: {"full_faithful": t.full_rate(m, np.flatnonzero(t.category == c)),
                                     "auroc": t.auroc(m, np.flatnonzero(t.category == c))
                                     if m in SIGNAL_METHODS else None} for c in sorted(set(t.category))},
            }
            block["sq1_2"]["rate"] = block["sq1_2"]["wrong"] / block["sq1_2"]["n"] if block["sq1_2"]["n"] else None
            if m in SIGNAL_METHODS:
                block["auroc"] = {
                    "signal": ev["auroc_signal"]["main"][m.split("_")[0]], "main": t.auroc(m, allidx),
                    "excluded": int((np.isnan(t.score[m]) & ~t.small).sum()),
                    **bootstrap(lambda i: t.auroc(m, i), indices, bs["interval"]),
                    "secondary_fill": t.auroc(m, allidx, fill=fill),
                    "by_class": {"normal": t.auroc(m, np.flatnonzero(~t.defective)),
                                 "defective": t.auroc(m, np.flatnonzero(t.defective))}}
            if m in ev["auroc_signal"]["secondary"]:
                vs = np.array([np.nan if e.get("verifier_score") is None else e["verifier_score"] for e in entries])
                block["verifier_auroc"] = {"main": t.auroc(m, allidx, score=vs),
                                           "secondary_fill": t.auroc(m, allidx, fill=fill, score=vs)}
            if m.startswith("p"):
                verified = [(r, r["m"][m]) for r in rows if r["m"][m].get("verifier") and not r["small"]]
                passed = [e for _, e in verified if e["verifier"]["all_pass"]]
                block["all_pass"] = {"main": rate(len(passed), len(verified)),
                                     "secondary": rate(len(passed), int((~t.small).sum())),
                                     "full_rate_among_pass": rate(sum(e["judge"]["full"] for e in passed),
                                                                  len(passed))}
                groups = defaultdict(list)
                for _, e in verified:
                    groups[status(e["verifier"]["claims"])].append(e["judge"]["full"])
                block["explanation_status"] = {s: {"n": len(v), "full_rate": sum(v) / len(v)}
                                               for s, v in sorted(groups.items())}
            datasets[d][m] = block

        # SQ2.2: P0 score with the claims of one tool removed; delta against the full score on the images left
        ablation[d] = {}
        for name, removed in {**ev["ablation"]["main"], **ev["ablation"]["secondary"]}.items():
            scores = np.array([np.nan if s is None else s for s in
                               (ablated_score(r["m"]["p0"]["verifier"]["claims"], removed)
                                if r["m"]["p0"]["verifier"] else None for r in rows)])
            left = np.flatnonzero(~np.isnan(scores))
            value, full_same = t.auroc("p0", left, score=scores), t.auroc("p0", left)
            excluded = int((~np.isnan(t.score["p0"]) & np.isnan(scores) & ~t.small).sum())
            ablation[d][name] = {"removed": removed, "auroc": value, "auroc_full_same_images": full_same,
                                 "excluded": excluded,
                                 "delta": value - full_same if value is not None and full_same is not None else None}

        # A8: paired differences between methods over the same resamples
        comparisons[d] = {"full_faithful": {}, "auroc": {}}
        for a, b in ev["comparisons"]["full_faithful"]:
            if a in METHODS[d] and b in METHODS[d]:
                def diff(i: np.ndarray, a: str = a, b: str = b) -> float | None:
                    x, y = t.full_rate(a, i), t.full_rate(b, i)
                    return x - y if x is not None and y is not None else None
                comparisons[d]["full_faithful"][f"{a}-{b}"] = {"value": diff(allidx),
                                                               **bootstrap(diff, indices, bs["interval"])}
        for a, b in ev["comparisons"]["auroc"]:
            # p0 minus b3 is A2, reported under a2
            if (a, b) != ("p0", "b3") and a in METHODS[d] and b in METHODS[d]:
                comparisons[d]["auroc"][f"{a}-{b}"] = pair_auroc(t, a, b, indices, fill, bs["interval"])

        # SQ2.3: re-query before and after, main and secondary final answers
        requeried = [r for r in rows if r["m"]["p1"]["requeried"] and not r["small"]]
        requery[d] = {"requeried": len(requeried), "unparsed_kept": sum(r["m"]["p1"]["kept"] for r in requeried)}
        for m in ("p1", "p1_secondary"):
            before = [r["m"]["p0"]["judge"]["full"] for r in requeried]
            after = [r["m"][m]["judge"]["full"] for r in requeried]
            requery[d][m] = {"before_full": rate(sum(before), len(before)), "after_full": rate(sum(after), len(after)),
                             "fixed": sum(not b and a for b, a in zip(before, after)),
                             "broken": sum(b and not a for b, a in zip(before, after))}

        # design doc 8: combined and Other images, existence and location only
        separate[d] = {}
        for flag, apart_rows in apart.items():
            separate[d][flag] = {}
            for m in METHODS[d]:
                parsed = [r["m"][m] for r in apart_rows if r["m"][m]["answer"]]
                tp = [e for e in parsed if e["answer"]["is_defective"]]
                separate[d][flag][m] = {"images": len(apart_rows),
                                        "existence": rate(sum(e["judge"]["existence"] for e in parsed), len(parsed)),
                                        "location": rate(sum(e["judge"]["location"] for e in tp), len(tp))}

        # A7: small types, with their type and full faithfulness counted apart
        small_rows = [r for r in rows if r["small"]]
        small_report[d] = {"types": sorted(f"{c}/{ty}" for dd, c, ty in small_types if dd == d),
                           "images": len(small_rows), "methods": {}}
        for m in METHODS[d]:
            parsed = [r["m"][m] for r in small_rows if r["m"][m]["answer"]]
            typed = [e for e in parsed if e["answer"]["is_defective"] and e["judge"]["type"] is not None]
            small_report[d]["methods"][m] = {"type": rate(sum(e["judge"]["type"] for e in typed), len(typed)),
                                             "full_faithful": rate(sum(r["m"][m]["judge"]["full"] for r in small_rows
                                                                       if r["m"][m]["answer"]), len(small_rows))}

        # design doc 13: types without library images
        missing = {(c, ty) for c in sorted(set(t.category)) for ty in requests.candidates(d, c)
                   if ty not in det.prototypes[0][(d, c)]}
        keep = np.array([not any((r["category"], ty) in missing for ty in r["gt"]["types"]) for r in rows])
        unverifiable[d] = {
            "types": sorted(f"{c}/{ty}" for c, ty in missing),
            "claims": sum(1 for r in rows if r["m"]["p0"].get("verifier") and r["m"]["p0"]["verifier"]["unverifiable"]),
            "gt_images_per_type": {f"{c}/{ty}": sum(1 for r in rows if r["category"] == c and ty in r["gt"]["types"])
                                   for c, ty in sorted(missing)},
            "b4_cannot_answer": len(missing),
            "secondary_excluding_types": {m: {"full_faithful": t.full_rate(m, np.flatnonzero(keep)),
                                              "auroc": t.auroc(m, np.flatnonzero(keep))
                                              if m in SIGNAL_METHODS else None} for m in METHODS[d]}}

        # A4: library crops that hold no mask pixel, per library seed
        library_miss[d] = {}
        for seed, lib in split["library"].items():
            lib_rows = [r for r in lib if r["dataset"] == d]
            missed = sum(gt_source.pixels_in_box(d, r["path"], det.features[f"{d}/{r['path']}"]["type_box"]) == 0
                         for r in lib_rows)
            library_miss[d][seed] = {"missed": missed, "n": len(lib_rows),
                                     "rate": missed / len(lib_rows) if lib_rows else None}

        # SQ2.4 cost and design doc 13 format compliance, over every request of the dataset
        cost[d], fmt[d] = {}, {}
        for kind in ("b0", "b1", "b2", "b3", "requery"):
            recs = [a["record"] for (k, dd, _), a in answers.items() if k == kind and dd == d]
            if not recs:
                continue
            seconds = sum(x["seconds"] for r in recs for x in r["attempts"])
            cost[d][kind] = {"requests": len(recs), "calls": sum(len(r["attempts"]) for r in recs),
                             "vlm_seconds": seconds, "vlm_seconds_per_request": seconds / len(recs)}
            bad = [r for r in recs if not r["ok"]]
            fmt[d][kind] = {"requests": len(recs), "violations": len(bad), "rate": len(bad) / len(recs),
                            "truncated_share": sum(r["attempts"][-1]["truncated"] for r in bad) / len(bad)
                            if bad else None}
        cost[d]["detector_seconds_per_image"] = (det.seconds_per_image if det.seconds_per_image is not None
                                                 else ev["cost_time"]["missing"])

        if d == "mvtec_ad":
            a2_result = {"seed_0": a2(rows, indices)}
            for seed in ev["seeds_range"]["seeds"]:
                if seed != 0:
                    a2_result[f"seed_{seed}"] = a2(main_rows(d, seed)[0], None)
            values = {k: [a2_result[f"seed_{s}"]["main"][k]["value"] for s in ev["seeds_range"]["seeds"]]
                      for k in ("p0", "b3", "diff")}
            a2_result["range"] = {k: [min(v), max(v)] if None not in v else None for k, v in values.items()}

    result["metrics"] = {"datasets": datasets, "a2": a2_result, "comparisons": comparisons, "ablation": ablation,
                         "requery": requery, "cost": cost, "format": fmt, "unverifiable": unverifiable,
                         "separate": separate, "small_types": small_report, "library_crop_miss": library_miss}
    result["environment"] = environment()
    out = Path(cfg["output_dir"]) / f"{result['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"-> {out}")


def provenance(cfg: dict, ev: dict, main_run: dict) -> dict:
    bundle = Path(ev["provenance"]["cache_bundle"])
    with open(bundle) as f:
        entries = sum(1 for _ in f)
    inputs = [*cfg["sessions"], main_run["split_result"], main_run["references"], main_run["detector"]["result"]]
    return {"cache_bundle": {"path": str(bundle), "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                             "entries": entries},
            "inputs": {p: file_sha256(Path(p)) for p in inputs},
            "eval_config": {"path": cfg["eval"], "sha256": file_sha256(Path(cfg["eval"]))},
            "rules": {"path": ev["rules"], "sha256": file_sha256(Path(ev["rules"]))}}


if __name__ == "__main__":
    main()
