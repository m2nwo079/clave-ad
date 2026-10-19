"""Ground-truth evaluation of the extension (configs/extension/plan.yaml hypotheses h5_1, h5_2, h8, h9).

Reuses scripts/analysis/evaluate.py (unchanged) and the handling rules of configs/analysis/eval.yaml. Modes:
  --check      no ground truth: cache keys of X1 to X4, parse states, P0 and P1 scores against the X4 records
  --reproduce  the main-run blocks rebuilt with the functions below must equal the main evaluation result file
  (default)    one result file in output_dir with aggregates only (no per-image ground truth)
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from string import Template

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts" / "analysis")]
from answers import Requests, load
from clave_ad.runinfo import environment, file_sha256, header
from clave_ad.vlm.cache_key import request_key
from clave_ad.verifiers.verify import verify

RANGE_ERROR = '"location_cells" must be a list of cells from "r1c1" to "r3c3"'
GLM = {"mvtec_ad": ["b0", "b3", "p0", "p1", "p1_secondary"], "visa": ["b0", "p0", "p1", "p1_secondary"]}
GRID = ["b0", "p0", "b0_grid", "p0_grid"]
SIGNAL = ("b0", "b3", "p0", "p1", "p1_secondary", "b0_grid", "p0_grid")


class GridRequests(Requests):
    """Adds the cache key of B0-grid records: the B0 prompt on the gridded image (scripts/main_run/run.py)."""

    def key(self, record: dict, b0_answer: dict | None = None) -> str:
        if record["kind"] != "b0_grid":
            return super().key(record, b0_answer)
        d, c = record["dataset"], record["category"]
        names = ", ".join(self.candidates(d, c))
        values = {"category": c.replace("_", " "), "candidates": names, "output_rules": self.output_rules}
        retry = Template(self.templates["retry"]).safe_substitute(candidates=names, output_rules=self.output_rules)
        prompt = Template(self.templates["b0"]).substitute(values)
        return request_key(self.revision, self.gen, "b0_grid", record["inputs_sha256"], prompt, retry)


def grid_answers(requests: GridRequests, path: str) -> tuple[dict, set[str]]:
    out, keys = {}, set()
    for r in json.loads(Path(path).read_text())["records"]:
        key = requests.key(r)
        keys.add(key)
        out[(r["kind"], r["dataset"], r["path"])] = {"record": r, "parsed": requests.entry(key)["parsed"]}
    return out, keys


def sources(cfg: dict, main_run: dict) -> tuple:
    """GLM answers and P1 records of X2 to X4, B0-grid answers of X1, request helpers, and every cache key used."""
    glm = Requests({**main_run, "vlm": cfg["new_model_vlm"], "cache_dir": cfg["cache_dir"]})
    grid = GridRequests({**main_run, "cache_dir": cfg["cache_dir"]})
    s = cfg["sessions"]
    answers, p1, keys = load([s["x2"], s["x3"], s["x4"]], glm)
    gridded, grid_keys = grid_answers(grid, s["x1"])
    return glm, answers, p1, gridded, keys | grid_keys


def check(cfg: dict, main_run: dict, rules: dict) -> None:
    """No ground truth: every request resolves, parse states agree, and seed-0 P0 and P1 scores match X4."""
    glm, answers, p1, gridded, keys = sources(cfg, main_run)
    cached = {p.stem for p in Path(cfg["cache_dir"]).glob("*.json")}
    with open(cfg["cache_bundle"]) as f:
        bundle = {json.loads(line)["key"] for line in f}
    disagree = sum((a["parsed"] is not None) != a["record"]["ok"] for a in {**answers, **gridded}.values())
    # seed-0 verifier inputs as in evaluate.Detector, rebuilt here because evaluate.py imports clave_ad.eval
    result = json.loads(Path(main_run["detector"]["result"]).read_text())
    root = Path(cfg["detector_dir"])
    for name in ("features.json", "embeddings.npz", "prototypes.npz"):
        if file_sha256(root / name) != result["files"][name]:
            sys.exit(f"local detector output {name} does not match {result['run_id']}")
    features = json.loads((root / "features.json").read_text())
    with np.load(root / "embeddings.npz") as e:
        embeddings = dict(zip(e["names"].tolist(), e["embeddings"]))
    prototypes = defaultdict(dict)
    with np.load(root / "prototypes.npz") as pz:
        for key, vector in zip(pz["keys"].tolist(), pz["vectors"]):
            seed, name = key.split("|")
            dataset, category, kind = name.split("/", 2)
            if seed == str(rules["type"]["library_main_seed"]):
                prototypes[(dataset, category)][kind] = vector
    thresholds = {k: v["threshold"] for k, v in result["metrics"]["thresholds"].items()}

    def score(d: str, c: str, p: str, answer: dict) -> dict:
        f = features[f"{d}/{p}"]
        return verify(answer, f["score"], thresholds[f"{d}/{c}"], f["cell_shares"], embeddings[f"{d}/{p}"],
                      prototypes[(d, c)], glm.candidates(d, c), rules["location"]["min_share_in_claimed"])

    counts = Counter()
    for (d, p), rec in p1.items():
        if rec.get("skipped"):
            counts["skipped"] += 1
            continue
        v0 = score(d, rec["category"], p, answers[("b0", d, p)]["parsed"])
        counts["p0_same" if v0["score"] == rec["p0"]["score"] and v0["failed"] == rec["requeried"] else "p0_diff"] += 1
        if "p1" in rec:
            v1 = score(d, rec["category"], p, answers[("requery", d, p)]["parsed"])
            counts["p1_same" if v1["score"] == rec["p1"]["score"] else "p1_diff"] += 1
    imported = sorted(m for m in sys.modules if m.startswith("clave_ad.eval"))
    print(f"{len(keys)} request keys, {len(cached)} cache entries, {len(bundle)} bundle keys, "
          f"missing {len(keys - cached)}, unused {len(cached - keys)}, bundle differs {len(bundle ^ cached)}")
    print(f"parse state disagreements {disagree}; P0/P1 against X4: {dict(sorted(counts.items()))}")
    print(f"clave_ad.eval modules imported: {imported}")
    good = keys == cached == bundle and not disagree and not counts["p0_diff"] and not counts["p1_diff"]
    sys.exit(0 if good and not imported else 1)


def evaluate(cfg: dict, config_path: Path, main_run: dict, rules: dict, reproduce: bool) -> None:
    # ground truth is read only through clave_ad.eval (evaluate.py imports it), so these imports stay here
    from clave_ad.eval.claims import GroundTruth, judge
    from clave_ad.eval.stats import bootstrap, resample_indices
    from evaluate import Detector, Table, b3_score, pair_auroc, rate

    ev = yaml.safe_load(Path(cfg["eval"]).read_text())
    plan = yaml.safe_load(Path(cfg["plan"]).read_text())
    analysis = yaml.safe_load(Path(cfg["analysis_run"]).read_text())
    vlm = yaml.safe_load(Path(main_run["vlm"]).read_text())
    split = json.loads(Path(main_run["split_result"]).read_text())
    det = Detector(Path(cfg["detector_dir"]), main_run, rules)
    evr, bs = rules["evaluation"], rules["evaluation"]["bootstrap"]
    share, fill, level = evr["location_correct_share"], ev["unparsed"]["secondary_fill"], bs["interval"]
    qwen = Requests(main_run)
    qwen_answers, qwen_p1, _ = load(analysis["sessions"], qwen)
    gt_source = GroundTruth(vlm["data"], qwen.types, yaml.safe_load(Path(cfg["visa_class_ids"]).read_text()),
                            rules["location"]["grid"], evr["gt_cell_min_pixels"])
    images = {(r["dataset"], r["path"]): r for r in split["vlm_images"]}
    gts = {k: gt_source.truth(k[0], k[1], r["category"]) for k, r in images.items()}
    type_images = Counter((d, images[(d, p)]["category"], t) for (d, p), g in gts.items() for t in g["types"])
    small_types = {k for k, n in type_images.items() if n < rules["type"]["library_per_type"] + evr["small_type_margin"]}

    def entries(d: str, p: str, c: str, seed: int, answers: dict, p1_records: dict, methods: list[str]) -> dict:
        """Per-method answer, signal score and verifier result, as evaluate.py method_entries builds them."""
        names = qwen.candidates(d, c)

        def parsed(kind: str) -> dict | None:
            a = answers.get((kind, d, p))
            return a["parsed"] if a else None

        b0, check_answer, grid = parsed("b0"), parsed("b3"), parsed("b0_grid")
        p0 = det.verify(d, c, p, b0, names, seed) if b0 else None
        out = {"b0": {"answer": b0, "score": b0["confidence"] if b0 else None},
               "b3": {"answer": b0, "score": b3_score(check_answer) if b0 and check_answer else None},
               "p0": {"answer": b0, "score": p0["score"] if p0 else None, "verifier": p0}}
        rec = p1_records.get((d, p), {})
        rq = parsed("requery") if rec.get("requeried") else None
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
            requeried = bool(rec.get("requeried"))
            out["p1"] = {**out["p0"], "requeried": requeried, "kept": requeried}
            out["p1_secondary"] = dict(out["p0"])
        pg = det.verify(d, c, p, grid, names, seed) if grid else None
        out["b0_grid"] = {"answer": grid, "score": grid["confidence"] if grid else None}
        out["p0_grid"] = {"answer": grid, "score": pg["score"] if pg else None, "verifier": pg}
        out = {m: out[m] for m in methods}
        for m in out.values():
            m["judge"] = judge(m["answer"], gts[(d, p)], share) if m["answer"] else None
        return out

    def main_rows(d: str, seed: int, answers: dict, p1_records: dict, methods: list[str]) -> list[dict]:
        """Main set of a seed: evaluation images without combined and Other images (as evaluate.py main_rows)."""
        rows = []
        for name in split["eval_images"][str(seed)]:
            dd, p = name.split("/", 1)
            gt = gts.get((dd, p))
            if dd != d or gt["separate"]:
                continue
            c = images[(d, p)]["category"]
            rows.append({"path": p, "category": c, "gt": gt, "m": entries(d, p, c, seed, answers, p1_records, methods),
                         "small": any((d, c, t) in small_types for t in gt["types"])})
        return rows

    def claim_rate(rows: list[dict], m: str, claim: str, idx: np.ndarray) -> float | None:
        """Existence over parsed answers, location over defective answers on defective images (own denominators)."""
        picked = [rows[i] for i in idx if rows[i]["m"][m]["answer"]]
        if claim == "location":
            picked = [r for r in picked if r["m"][m]["answer"]["is_defective"] and r["gt"]["defective"]]
        return sum(r["m"][m]["judge"][claim] for r in picked) / len(picked) if picked else None

    def block(t, rows: list[dict], m: str, indices: list[np.ndarray]) -> dict:
        """The evaluate.py per-method block without its descriptive extras."""
        allidx = np.arange(len(rows))
        parsed = [r["m"][m] for r in rows if r["m"][m]["answer"]]
        tp = [r["m"][m] for r in rows if r["m"][m]["answer"] and r["m"][m]["answer"]["is_defective"]
              and r["gt"]["defective"]]
        tp_typed = [r["m"][m] for r in rows if not r["small"] and r["m"][m]["answer"]
                    and r["m"][m]["answer"]["is_defective"] and r["gt"]["defective"]]
        out = {"unparsed": len(rows) - len(parsed),
               "full_faithful": {"rate": t.full_rate(m, allidx), "n": int((~t.small).sum()),
                                 **bootstrap(lambda i: t.full_rate(m, i), indices, level)},
               "claims": {"existence": rate(sum(e["judge"]["existence"] for e in parsed), len(parsed)),
                          "location": rate(sum(e["judge"]["location"] for e in tp), len(tp)),
                          "type": rate(sum(e["judge"]["type"] for e in tp_typed), len(tp_typed))}}
        if m in SIGNAL:
            out["auroc"] = {"main": t.auroc(m, allidx), "excluded": int((np.isnan(t.score[m]) & ~t.small).sum()),
                            **bootstrap(lambda i: t.auroc(m, i), indices, level),
                            "secondary_fill": t.auroc(m, allidx, fill=fill)}
        return out

    def diff(fa, fb, n: int, indices: list[np.ndarray]) -> dict:
        """A8 paired difference: both statistics on the same resample, first minus second."""
        def d_(i: np.ndarray) -> float | None:
            x, y = fa(i), fb(i)
            return x - y if x is not None and y is not None else None
        return {"value": d_(np.arange(n)), **bootstrap(d_, indices, level)}

    def requery_block(rows: list[dict]) -> dict:
        requeried = [r for r in rows if r["m"]["p1"]["requeried"] and not r["small"]]
        out = {"requeried": len(requeried), "unparsed_kept": sum(r["m"]["p1"]["kept"] for r in requeried)}
        for m in ("p1", "p1_secondary"):
            before = [r["m"]["p0"]["judge"]["full"] for r in requeried]
            after = [r["m"][m]["judge"]["full"] for r in requeried]
            out[m] = {"before_full": rate(sum(before), len(before)), "after_full": rate(sum(after), len(after)),
                      "fixed": sum(not b and a for b, a in zip(before, after)),
                      "broken": sum(b and not a for b, a in zip(before, after))}
        return out

    def low(x: dict) -> float | None:
        return x["ci"][0] if x.get("ci") else None

    if reproduce:
        reproduce_main(cfg, qwen_answers, qwen_p1, main_rows, Table, block, pair_auroc, diff, requery_block,
                       claim_rate, resample_indices, bs, fill, level)
        return

    glm, answers, p1, gridded, _ = sources(cfg, main_run)
    hyp = plan["hypotheses"]
    x5, h5_1, h5_2, h8, x9, h9 = {}, {}, {}, {}, {}, {}
    for d in GLM:
        rows = main_rows(d, 0, answers, p1, GLM[d])
        t = Table(rows, GLM[d])
        indices = resample_indices(t.category, bs["resamples"], bs["seed"])
        x5[d] = {"images": len(rows), "small_type_images": int(t.small.sum()),
                 **{m: block(t, rows, m, indices) for m in GLM[d]}}
        out = pair_auroc(t, "p0", "b0", indices, fill, level)
        out["holds"] = bool(low(out["main"]["diff"]) is not None
                            and low(out["main"]["diff"]) > hyp["h5_2"]["p0_minus_b0_confidence_auroc_ci_low_above"])
        h5_2[d] = out
        rq = requery_block(rows)
        main = diff(lambda i: t.full_rate("p1_secondary", i), lambda i: t.full_rate("b0", i), len(rows), indices)
        h8[d] = {"main_selective_minus_b0": main,
                 "secondary_always_minus_b0": diff(lambda i: t.full_rate("p1", i), lambda i: t.full_rate("b0", i),
                                                   len(rows), indices),
                 "requery": rq,
                 "holds": bool(low(main) is not None
                               and low(main) > hyp["h8"]["p1_selective_minus_b0_full_faithful_ci_low_above"]
                               and rq["p1_secondary"]["broken"] < rq["p1_secondary"]["fixed"])}
        if d == "mvtec_ad":
            h = hyp["h5_1"]
            a2 = pair_auroc(Table(rows, ["p0", "b3"]), "p0", "b3", indices, fill, level)
            a2["holds"] = bool(low(a2["main"]["p0"]) is not None and low(a2["main"]["diff"]) is not None
                               and low(a2["main"]["p0"]) > h["p0_auroc_ci_low_above"]
                               and low(a2["main"]["diff"]) > h["p0_minus_b3_auroc_ci_low_above"])
            h5_1 = {"seed_0": a2}
            for seed in h["seeds_range"]:
                seed_rows = main_rows(d, seed, answers, p1, ["p0", "b3"])
                h5_1[f"seed_{seed}"] = pair_auroc(Table(seed_rows, ["p0", "b3"]), "p0", "b3", None, fill, level)
            seeds = [0, *h["seeds_range"]]
            values = {k: [h5_1[f"seed_{s}"]["main"][k]["value"] for s in seeds] for k in ("p0", "b3", "diff")}
            h5_1["range"] = {k: [min(v), max(v)] if None not in v else None for k, v in values.items()}

        # x9: Qwen B0 of the main run against B0-grid of X1 on the same images
        both = {**qwen_answers, **gridded}
        rows = main_rows(d, 0, both, qwen_p1, GRID)
        t = Table(rows, GRID)
        indices = resample_indices(t.category, bs["resamples"], bs["seed"])
        x9[d] = {"images": len(rows), **{m: block(t, rows, m, indices) for m in GRID}}
        h = hyp["h9"]
        main_set = {r["path"] for r in rows}
        range_errors = {}
        for kind in ("b0", "b0_grid"):
            recs = [a["record"] for (k, dd, p), a in both.items() if k == kind and dd == d]
            bad = [r for r in recs if not r["ok"] and r["final_error"] == RANGE_ERROR]
            range_errors[kind] = {"main_set": sum(r["path"] in main_set for r in bad), "all_session_images": len(bad)}
        n = len(rows)
        location = diff(lambda i: claim_rate(rows, "b0_grid", "location", i),
                        lambda i: claim_rate(rows, "b0", "location", i), n, indices)
        existence = diff(lambda i: claim_rate(rows, "b0_grid", "existence", i),
                         lambda i: claim_rate(rows, "b0", "existence", i), n, indices)
        full = diff(lambda i: t.full_rate("b0_grid", i), lambda i: t.full_rate("b0", i), n, indices)
        p0_pair = pair_auroc(t, "p0_grid", "p0", indices, fill, level)

        def above(x: dict, bound: float) -> bool:
            return low(x) is not None and low(x) > bound

        h9[d] = {"out_of_grid": range_errors, "location_correct": location, "existence_correct": existence,
                 "full_faithful": full, "p0_auroc": p0_pair,
                 "holds": {"out_of_grid_fewer": range_errors["b0_grid"]["main_set"] < range_errors["b0"]["main_set"],
                           "location": above(location, h["location_correct_ci_low_above"]),
                           "full_faithful": above(full, h["full_faithful_ci_low_above"]),
                           "existence": above(existence, h["existence_correct_ci_low_above"]),
                           "p0_auroc": above(p0_pair["main"]["diff"], h["p0_auroc_ci_low_above"])}}

    result = header("eval_extension", config_path, "mvtec_ad+visa", bs["seed"])
    with open(cfg["cache_bundle"]) as f:
        entries_in_bundle = sum(1 for _ in f)
    inputs = [*cfg["sessions"].values(), *analysis["sessions"], cfg["plan"], cfg["eval"], cfg["new_model_vlm"],
              main_run["vlm"], ev["rules"], main_run["split_result"], main_run["references"],
              main_run["detector"]["result"]]
    result["provenance"] = {"cache_bundle": {"path": cfg["cache_bundle"], "entries": entries_in_bundle,
                                             "sha256": file_sha256(Path(cfg["cache_bundle"]))},
                            "inputs": {p: file_sha256(Path(p)) for p in inputs}}
    result["metrics"] = {"x5": x5, "h5_1": h5_1, "h5_2": h5_2, "h8": h8, "x9": x9, "h9": h9}
    result["environment"] = environment()
    out = Path(cfg["output_dir"]) / f"{result['run_id']}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"-> {out}")


def reproduce_main(cfg, answers, p1_records, main_rows, Table, block, pair_auroc, diff, requery_block,
                   claim_rate, resample_indices, bs, fill, level) -> None:
    """Rebuild main-run blocks with the functions above and compare them with the main evaluation result file."""
    target = json.loads(Path(cfg["reproduce"]).read_text())["metrics"]
    mismatches = []

    def same(ours, theirs, where: str) -> None:
        if isinstance(ours, dict):
            for k, v in ours.items():
                same(v, theirs.get(k) if isinstance(theirs, dict) else None, f"{where}.{k}")
        elif ours != theirs:
            mismatches.append(f"{where}: {ours} != {theirs}")

    for d, methods in GLM.items():
        rows = main_rows(d, 0, answers, p1_records, methods)
        t = Table(rows, methods)
        indices = resample_indices(t.category, bs["resamples"], bs["seed"])
        for m in methods:
            ours = block(t, rows, m, indices)
            same(ours, target["datasets"][d][m], f"datasets.{d}.{m}")
        same(pair_auroc(t, "p0", "b0", indices, fill, level), target["comparisons"][d]["auroc"]["p0-b0"],
             f"comparisons.{d}.auroc.p0-b0")
        same(diff(lambda i: t.full_rate("p1", i), lambda i: t.full_rate("b0", i), len(rows), indices),
             target["comparisons"][d]["full_faithful"]["p1-b0"], f"comparisons.{d}.full_faithful.p1-b0")
        same(requery_block(rows), target["requery"][d], f"requery.{d}")
        # the own-denominator rates of h9 equal the claim rates of the main blocks on all images
        for m in methods:
            for claim in ("existence", "location"):
                same(claim_rate(rows, m, claim, np.arange(len(rows))), target["datasets"][d][m]["claims"][claim]["rate"],
                     f"claim_rate.{d}.{m}.{claim}")
        if d == "mvtec_ad":
            a2 = pair_auroc(Table(rows, ["p0", "b3"]), "p0", "b3", indices, fill, level)
            same(a2, target["a2"]["seed_0"], "a2.seed_0")
            for seed in (1, 2):
                seed_rows = main_rows(d, seed, answers, p1_records, ["p0", "b3"])
                same(pair_auroc(Table(seed_rows, ["p0", "b3"]), "p0", "b3", None, fill, level),
                     target["a2"][f"seed_{seed}"], f"a2.seed_{seed}")
    print(f"reproduction of {cfg['reproduce']}: {len(mismatches)} mismatches")
    for line in mismatches[:20]:
        print(" ", line)
    sys.exit(1 if mismatches else 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/extension/eval_run.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--reproduce", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main_run = yaml.safe_load(Path(cfg["main_run"]).read_text())
    rules = yaml.safe_load(Path(main_run["rules"]).read_text())
    if args.check:
        check(cfg, main_run, rules)
    else:
        evaluate(cfg, Path(args.config), main_run, rules, args.reproduce)


if __name__ == "__main__":
    main()
