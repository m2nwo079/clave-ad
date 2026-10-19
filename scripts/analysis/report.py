"""Korean result report: tables and figures generated from result files, assembled into one Markdown file.

The human-written source (configs/analysis/report.yaml `source`) holds no result numbers; it places generated tables
and figures with <!-- table:name --> and <!-- figure:name --> markers. Every number in the output comes from the
result files in the config. Figures are aggregates only (no dataset images). Run with --self-check to test the
helpers on made-up data only.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.runinfo import file_sha256, git_commit

DATASETS = {"mvtec_ad": "MVTec AD", "visa": "VisA"}
METHODS = {"b0": "B0", "b1": "B1", "b2": "B2", "b3": "B3", "b4": "B4", "p0": "P0", "p1": "P1",
           "p1_secondary": "P1-sec"}
REQUERY_RULES = {"p1": "re-query answer always", "p1_secondary": "re-query answer if score is higher"}
# extension methods: GLM-4.6V-Flash blocks (x5) and the Qwen B0-grid comparison (x9)
EXT_METHODS = {**METHODS, "b0_grid": "B0-grid", "p0_grid": "P0-grid", "requery": "re-query"}
MARKER = re.compile(r"<!--\s*(table|figure):([a-z0-9_]+)\s*-->")


def f3(x: float | None) -> str:
    return "-" if x is None else f"{x:.3f}"


def ci(c: list[float] | None) -> str:
    return "-" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"


def frac(d: dict) -> str:
    """Rate with its counts, from a {correct, n, rate} dict."""
    return f"{f3(d['rate'])} ({d['correct']}/{d['n']})"


def md(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def number_issues(text: str, blocked: list[str], allowed: list[str]) -> list[str]:
    """Lines of the source that carry a blocked number pattern once the allowed tokens are removed."""
    issues = []
    for i, line in enumerate(text.splitlines(), 1):
        rest = line
        for pattern in allowed:
            rest = re.sub(pattern, "", rest)
        for pattern in blocked:
            issues += [f"line {i}: {m.group()!r} in: {line.strip()}" for m in re.finditer(pattern, rest)]
    return issues


def assemble(source: str, parts: dict[str, str]) -> str:
    """Replace every marker with its generated part; a missing or an unused part stops the report."""
    used, missing = set(), []

    def sub(m: re.Match) -> str:
        key = f"{m.group(1)}:{m.group(2)}"
        if key not in parts:
            missing.append(key)
            return m.group()
        used.add(key)
        return parts[key]

    out = MARKER.sub(sub, source)
    unused = sorted(set(parts) - used)
    if missing or unused:
        sys.exit(f"markers without a generated part: {missing}; generated parts without a marker: {unused}")
    return out


def methods_of(block: dict) -> list[str]:
    return [m for m in METHODS if m in block]


def bar_err(values: list[float], cis: list[list[float]]) -> list[list[float]]:
    return [[v - c[0] for v, c in zip(values, cis)], [c[1] - v for v, c in zip(values, cis)]]


# ---------- tables ----------

def tables(ev: dict, ph: dict, ph_cfg: dict, sessions: list[dict], inputs: dict[str, str], commit: str) -> dict:
    m, t = ev["metrics"], {}
    ds = [d for d in DATASETS if d in m["datasets"]]

    t["provenance"] = md(["Input", "sha256 (first 12)", "run_id", "git_commit"],
                         [[f"`{p}`", s[:12], r.get("run_id", "-"), r.get("git_commit", "-")[:7]]
                          for p, (s, r) in inputs.items()]) + f"\n\nGenerated at git commit `{commit}`."

    t["data"] = md(["Dataset", "Evaluation images", "Small-type images", "Multi-type defective images"],
                   [[DATASETS[d], m["datasets"][d]["images"], m["datasets"][d]["small_type_images"],
                     frac(m["datasets"][d]["multi_type_ratio"])] for d in ds])

    rows = []
    for d in ds:
        for k in methods_of(m["datasets"][d]):
            x = m["datasets"][d][k]
            rows.append([DATASETS[d], METHODS[k], f3(x["full_faithful"]["rate"]), ci(x["full_faithful"]["ci"]),
                         frac(x["claims"]["existence"]), frac(x["claims"]["location"]), frac(x["claims"]["type"]),
                         f"{f3(x['sq1_2']['rate'])} ({x['sq1_2']['wrong']}/{x['sq1_2']['n']})",
                         f3(x["cells"]["mean"]), x["cells"]["empty"], x["normal_called_defective"], x["unparsed"]])
    t["bq1_claims"] = md(["Dataset", "Method", "Full faithful", "95% CI", "Existence", "Location", "Type",
                          "Judgement right, location or type wrong", "Mean cells", "Empty cells",
                          "Normal called defective", "Unparsed"], rows)

    rows = []
    for d in ds:
        for k in methods_of(m["datasets"][d]):
            a = m["datasets"][d][k].get("auroc")
            if a is None:
                continue
            v = m["datasets"][d][k].get("verifier_auroc")
            rows.append([DATASETS[d], METHODS[k], a["signal"], f3(a["main"]), ci(a["ci"]), a["excluded"],
                         f3(a["secondary_fill"]), f3(a["by_class"]["normal"]), f3(a["by_class"]["defective"]),
                         f3(v["main"]) if v else "-"])
    t["auroc"] = md(["Dataset", "Method", "Signal", "AUROC", "95% CI", "Excluded (unparsed)", "AUROC (fill, secondary)",
                     "AUROC on normal images", "AUROC on defective images", "P0 verifier on these answers (secondary)"],
                    rows)

    rows = []
    for d in ds:
        for pair, x in m["comparisons"][d]["full_faithful"].items():
            rows.append([DATASETS[d], "full faithful", pair_label(pair), f3(x["value"]), ci(x["ci"]), "-"])
        for pair, x in m["comparisons"][d]["auroc"].items():
            main = x["main"]
            rows.append([DATASETS[d], "AUROC", pair_label(pair), f3(main["diff"]["value"]), ci(main["diff"]["ci"]),
                         main["n_pairs"]])
    t["comparisons"] = md(["Dataset", "Metric", "Difference", "Value", "95% CI (paired)", "Image pairs"], rows)

    a2, rows = m["a2"], []
    for seed in sorted(k for k in a2 if k.startswith("seed_")):
        x = a2[seed]["main"]
        rows.append([seed.split("_")[1], f3(x["p0"]["value"]), ci(x["p0"].get("ci")), f3(x["b3"]["value"]),
                     ci(x["b3"].get("ci")), f3(x["diff"]["value"]), ci(x["diff"].get("ci")), x["n_pairs"],
                     {True: "holds", False: "does not hold"}.get(x.get("holds"), "-"),
                     f3(a2[seed]["secondary_fill"]["diff"]["value"])])
    rows.append(["range (0-2)", ci(a2["range"]["p0"]), "-", ci(a2["range"]["b3"]), "-", ci(a2["range"]["diff"]),
                 "-", "-", "-", "-"])
    t["a2"] = md(["Library seed", "P0 AUROC", "95% CI", "B3 AUROC", "95% CI", "P0 - B3", "95% CI (paired)",
                  "Image pairs", "A2", "P0 - B3 (fill, secondary)"], rows)

    rows = []
    for d in ds:
        for k in methods_of(m["datasets"][d]):
            x = m["datasets"][d][k]
            if "all_pass" not in x:
                continue
            s = x["explanation_status"]
            rows.append([DATASETS[d], METHODS[k], frac(x["all_pass"]["main"]), frac(x["all_pass"]["secondary"]),
                         frac(x["all_pass"]["full_rate_among_pass"])]
                        + [f"{f3(s[c]['full_rate'])} (n={s[c]['n']})"
                           for c in ("all_verified", "partially_verified", "withheld")])
    t["verification"] = md(["Dataset", "Method", "All pass (parsed answers)", "All pass (all images)",
                            "Full faithful among all pass", "All verified: full faithful",
                            "Partially verified: full faithful", "Withheld: full faithful"], rows)

    t["ablation"] = md(["Dataset", "Variant", "Removed claims", "AUROC", "Full AUROC on same images", "Delta",
                        "Excluded (no claim left)"],
                       [[DATASETS[d], v, ", ".join(x["removed"]), f3(x["auroc"]), f3(x["auroc_full_same_images"]),
                         f3(x["delta"]), x["excluded"]] for d in ds for v, x in m["ablation"][d].items()])

    t["requery"] = md(["Dataset", "Final answer rule", "Re-queried", "Full faithful before", "Full faithful after",
                       "Fixed", "Broken", "Unparsed re-query, original kept"],
                      [[DATASETS[d], REQUERY_RULES[k], m["requery"][d]["requeried"], frac(x["before_full"]),
                        frac(x["after_full"]), x["fixed"], x["broken"], m["requery"][d]["unparsed_kept"]]
                       for d in ds for k, x in m["requery"][d].items() if k in REQUERY_RULES])

    rows = []
    for d in ds:
        for k, x in m["cost"][d].items():
            if isinstance(x, dict):
                rows.append([DATASETS[d], k, x["requests"], x["calls"], f"{x['vlm_seconds']:.0f}",
                             f"{x['vlm_seconds_per_request']:.2f}"])
        rows.append([DATASETS[d], "detector", "-", "-", "-", f"{m['cost'][d]['detector_seconds_per_image']:.2f}"])
    t["cost"] = md(["Dataset", "Request kind", "Requests", "VLM calls (with retries)", "VLM seconds",
                    "Seconds per request (detector: per image)"], rows)

    for name, key in (("per_category_full", "full_faithful"), ("per_category_auroc", "auroc")):
        rows, heads = [], []
        for d in ds:
            ms = [k for k in methods_of(m["datasets"][d])
                  if m["datasets"][d][k].get("per_category")
                  and any(v.get(key) is not None for v in m["datasets"][d][k]["per_category"].values())]
            heads = heads or ms
            cats = sorted(m["datasets"][d][ms[0]]["per_category"])
            rows += [[DATASETS[d], c] + [f3(m["datasets"][d][k]["per_category"][c].get(key)) if k in ms else "-"
                                         for k in heads] for c in cats]
        t[name] = md(["Dataset", "Category"] + [METHODS[k] for k in heads], rows)

    rows = []
    for d in ds:
        for group, block in m["separate"][d].items():
            rows += [[DATASETS[d], group, METHODS[k], x["images"], frac(x["existence"]), frac(x["location"])]
                     for k, x in block.items()]
    t["separate"] = md(["Dataset", "Group", "Method", "Images", "Existence", "Location"], rows)

    rows = []
    for d in ds:
        s = m["small_types"][d]
        if not s["types"]:
            rows.append([DATASETS[d], "none", 0, "-", "-", "-"])
            continue
        rows += [[DATASETS[d], ", ".join(s["types"]), s["images"], METHODS[k], frac(x["type"]),
                  frac(x["full_faithful"])] for k, x in s["methods"].items()]
    t["small_types"] = md(["Dataset", "Small-sample types", "Images", "Method", "Type", "Full faithful"], rows)

    rows = []
    for d in ds:
        u = m["unverifiable"][d]
        rows += [[DATASETS[d], ty, n, u["claims"], u["b4_cannot_answer"]] for ty, n in u["gt_images_per_type"].items()]
        if not u["types"]:
            rows.append([DATASETS[d], "none", 0, u["claims"], u["b4_cannot_answer"]])
    t["unverifiable"] = md(["Dataset", "Type not in library", "Ground-truth images", "Unverifiable claims (dataset)",
                            "B4 cannot answer (dataset)"], rows)
    t["unverifiable_secondary"] = md(
        ["Dataset", "Method", "Full faithful", "AUROC"],
        [[DATASETS[d], METHODS[k], f3(x["full_faithful"]), f3(x["auroc"])]
         for d in ds for k, x in m["unverifiable"][d]["secondary_excluding_types"].items()])

    t["crop_miss"] = md(["Dataset", "Library seed", "Crops missing the defect", "Library crops", "Rate"],
                        [[DATASETS[d], seed, x["missed"], x["n"], f3(x["rate"])]
                         for d in ds for seed, x in m["library_crop_miss"][d].items()])

    rows = []
    for d in ds:
        for k, x in m["format"][d].items():
            c = session_compliance(sessions, f"{k}/{d}")
            rows.append([DATASETS[d], k, x["requests"], c["first_ok"], c["ok_after_retry_1"], c["ok_after_retry_2"],
                         x["violations"], f3(x["rate"]), f3(x["truncated_share"]), c["ok_fenced"],
                         c["ok_control_char"]])
    t["format"] = md(["Dataset", "Request kind", "Requests", "First answer ok", "Ok after retry 1",
                      "Ok after retry 2", "Violations", "Violation rate", "Truncated share of violations",
                      "Fenced (allowed)", "Control characters (allowed)"], rows)

    t.update(posthoc_tables(ph, ph_cfg))
    return t


def pair_label(pair: str) -> str:
    a, b = pair.split("-")
    return f"{METHODS[a]} - {METHODS[b]}"


def session_compliance(sessions: list[dict], key: str) -> dict:
    found = [s["metrics"]["compliance"][key] for s in sessions if key in s["metrics"]["compliance"]]
    if len(found) != 1:
        sys.exit(f"expected one session with compliance {key}, found {len(found)}")
    return found[0]


def posthoc_tables(ph: dict, cfg: dict) -> dict:
    m, t = ph["metrics"], {}
    classes = ("cleared_normal", "cleared_in_grid", "out_of_grid", "other_error")
    t["posthoc_e1"] = md(["Images", "n"] + ["First output: " + c for c in classes]
                         + ["Cleared share", "Final output parsed (reference)"],
                         [[role, x["n"]] + [x["first_output"].get(c, 0) for c in classes]
                          + [f3(x["cleared_share"]), x["final_output_reference"].get("parsed", 0)]
                          for role, x in m["e1"].items()])
    ch = m["e1"]["control"]["changes_vs_main_b0"]
    t["posthoc_e1_changes"] = md(["Changed against main-run B0", "Control images", "Compared", "B0 unparsed, skipped"],
                                 [[k, ch[k], ch["compared"], ch["b0_unparsed_skipped"]]
                                  for k in cfg["reporting"]["e1_control_changes"]])
    min_columns = cfg["hypotheses"]["h1_layout"]["min_columns"]
    t["posthoc_e2"] = md(["Images", "n", "Parsed", f"Columns >= {min_columns}", "Share", "Column distribution"],
                         [[role, x["n"], x["parsed"], x["columns_at_least_min"], f3(x["share"]),
                           ", ".join(f"{c} columns: {n}" for c, n in x["columns"].items())]
                          for role, x in m["e2"].items()])
    h, hc = m["hypotheses"], cfg["hypotheses"]
    verdict = {True: "holds", False: "does not hold"}
    t["posthoc_hypotheses"] = md(
        ["Hypothesis", "Measure", "Violation images", "Control images", "Rule", "Outcome"],
        [["H2 no grid reference", hc["h2_grid"]["measure"], f3(h["h2_grid"]["value"]), "-",
          f"above {hc['h2_grid']['above']}", verdict[h["h2_grid"]["holds"]]],
         ["H1 layout read as columns", hc["h1_layout"]["measure"], f3(h["h1_layout"]["value"]),
          f3(h["h1_layout"]["control"]), f"above {hc['h1_layout']['above']} and above control",
          verdict[h["h1_layout"]["holds"]]],
         ["Grid lines on eval cell boundaries", "grid_matches_claims_cell_boxes", "-", "-", "match",
          "match" if m["grid_matches_claims_cell_boxes"] else "mismatch"]])
    t["posthoc_e3"] = md(["Answer kind", "Violation records", "Fixable by clipping", "Full faithful after fix"],
                         [[k.upper() if k != "requery" else "re-query", x["records"], x["fixable"],
                           x["full_faithful_after_fix"]] for k, x in m["e3"].items()])
    return t


def outcome(holds: bool | None) -> str:
    return {True: "holds", False: "does not hold"}.get(holds, "-")


def extension_tables(ex: dict, plan: dict, sessions: dict[str, dict]) -> dict:
    """Tables of the extension evaluation (GLM-4.6V-Flash x5, h5, h8; Qwen B0-grid x9, h9) and its sessions."""
    m, hyp, t = ex["metrics"], plan["hypotheses"], {}

    def method_rows(block: dict) -> list[list]:
        rows = []
        for d, b in block.items():
            for k in [k for k in EXT_METHODS if k in b]:
                x, a = b[k], b[k].get("auroc", {})
                rows.append([DATASETS[d], EXT_METHODS[k], x["unparsed"], f3(x["full_faithful"]["rate"]),
                             ci(x["full_faithful"]["ci"]), frac(x["claims"]["existence"]),
                             frac(x["claims"]["location"]), frac(x["claims"]["type"]), f3(a.get("main")),
                             ci(a.get("ci")), a.get("excluded", "-"), f3(a.get("secondary_fill"))])
        return rows

    heads = ["Dataset", "Method", "Unparsed", "Full faithful", "95% CI", "Existence", "Location", "Type", "AUROC",
             "95% CI", "Excluded (unparsed)", "AUROC (fill, secondary)"]
    t["ext_x5"] = md(heads, method_rows(m["x5"]))
    t["ext_x9"] = md(heads, method_rows(m["x9"]))

    h1, h2, rows = hyp["h5_1"], hyp["h5_2"], []
    for seed in sorted(k for k in m["h5_1"] if k.startswith("seed_")):
        x, main = m["h5_1"][seed], m["h5_1"][seed]["main"]
        judged = seed == "seed_0"
        rows.append(["H5-1", DATASETS[h1["dataset"]], seed.split("_")[1], f3(main["p0"]["value"]), ci(main["p0"].get("ci")),
                     "B3", f3(main["b3"]["value"]), ci(main["b3"].get("ci")), f3(main["diff"]["value"]),
                     ci(main["diff"].get("ci")), main["n_pairs"],
                     f"P0 CI low > {h1['p0_auroc_ci_low_above']}, P0 - B3 CI low > {h1['p0_minus_b3_auroc_ci_low_above']}"
                     if judged else "range only", outcome(x.get("holds")) if judged else "-",
                     f3(x["secondary_fill"]["diff"]["value"])])
    r = m["h5_1"]["range"]
    rows.append(["H5-1", DATASETS[h1["dataset"]], "range (0-2)", ci(r["p0"]), "-", "B3", ci(r["b3"]), "-", ci(r["diff"]),
                 "-", "-", "-", "-", "-"])
    for d, x in m["h5_2"].items():
        main = x["main"]
        rows.append(["H5-2", DATASETS[d], "0", f3(main["p0"]["value"]), ci(main["p0"]["ci"]), "B0 confidence",
                     f3(main["b0"]["value"]), ci(main["b0"]["ci"]), f3(main["diff"]["value"]), ci(main["diff"]["ci"]),
                     main["n_pairs"], f"P0 - B0 CI low > {h2['p0_minus_b0_confidence_auroc_ci_low_above']}",
                     outcome(x["holds"]), f3(x["secondary_fill"]["diff"]["value"])])
    t["ext_h5"] = md(["Hypothesis", "Dataset", "Library seed", "P0 AUROC", "95% CI", "Compared signal", "AUROC",
                      "95% CI", "Difference", "95% CI (paired)", "Image pairs", "Criterion (plan.yaml)", "Outcome",
                      "Difference (fill, secondary)"], rows)

    h8, rows = hyp["h8"], []
    for d, x in m["h8"].items():
        for key, rule, diff in (("p1_secondary", "re-query answer if score is higher (main)", "main_selective_minus_b0"),
                                ("p1", "re-query answer always (secondary)", "secondary_always_minus_b0")):
            q, main = x["requery"][key], key == "p1_secondary"
            rows.append([DATASETS[d], rule, f3(x[diff]["value"]), ci(x[diff]["ci"]), x["requery"]["requeried"],
                         frac(q["before_full"]), frac(q["after_full"]), q["fixed"], q["broken"],
                         x["requery"]["unparsed_kept"],
                         f"CI low > {h8['p1_selective_minus_b0_full_faithful_ci_low_above']} and broken < fixed"
                         if main else "-", outcome(x["holds"]) if main else "-"])
    t["ext_h8"] = md(["Dataset", "Final answer rule", "Full faithful minus B0", "95% CI (paired)", "Re-queried",
                      "Full faithful before", "Full faithful after", "Fixed", "Broken",
                      "Unparsed re-query, original kept", "Criterion (plan.yaml)", "Outcome"], rows)

    h9, rows = hyp["h9"], []
    for d, x in m["h9"].items():
        oog = x["out_of_grid"]
        rows.append([DATASETS[d], "out-of-grid final answers (main set)", oog["b0_grid"]["main_set"],
                     oog["b0"]["main_set"], "-", "-", "B0-grid count < B0 count", outcome(x["holds"]["out_of_grid_fewer"])])
        rows.append([DATASETS[d], "out-of-grid final answers (all session images, reference)",
                     oog["b0_grid"]["all_session_images"], oog["b0"]["all_session_images"], "-", "-", "no verdict", "-"])
        for key, label, bound in (("location_correct", "location correct rate", "location_correct_ci_low_above"),
                                  ("existence_correct", "existence correct rate", "existence_correct_ci_low_above"),
                                  ("full_faithful", "full faithful rate", "full_faithful_ci_low_above")):
            v = x[key]
            rows.append([DATASETS[d], label + " (difference)", "-", "-", f3(v["value"]), ci(v["ci"]),
                         f"CI low > {h9[bound]}", outcome(x["holds"][key.split("_")[0]] if key != "full_faithful"
                                                          else x["holds"]["full_faithful"])])
        p = x["p0_auroc"]["main"]
        rows.append([DATASETS[d], "P0 AUROC (P0-grid minus P0)", f3(p["p0_grid"]["value"]), f3(p["p0"]["value"]),
                     f3(p["diff"]["value"]), ci(p["diff"]["ci"]), f"CI low > {h9['p0_auroc_ci_low_above']}",
                     outcome(x["holds"]["p0_auroc"])])
    t["ext_h9"] = md(["Dataset", "Measure", "B0-grid", "B0", "Difference", "95% CI (paired)", "Criterion (plan.yaml)",
                      "Outcome"], rows)

    rows, cost = [], []
    for name, s in sessions.items():
        sm = s["metrics"]
        for key, c in sm["compliance"].items():
            if "/" not in key:
                continue
            kind, d = key.split("/")
            rows.append([name, DATASETS[d], EXT_METHODS.get(kind, kind), c["requests"], c["first_ok"],
                         c["ok_after_retry_1"], c["ok_after_retry_2"], c["violations"], c["violations_truncated"],
                         c["violations_think_block"], c["ok_fenced"], c["nonfinite_requests"]])
        cost.append([name, sm["generations"], f"{sm['generation_seconds']:.0f}",
                     f"{sm['generation_seconds'] / sm['generations']:.2f}"])
    t["ext_format"] = md(["Session", "Dataset", "Request kind", "Requests", "First answer ok", "Ok after retry 1",
                          "Ok after retry 2", "Violations", "Truncated violations", "Think-block violations",
                          "Fenced (allowed)", "NaN or inf requests"], rows)
    t["ext_cost"] = md(["Session", "VLM generations (with retries)", "VLM seconds", "Seconds per generation"], cost)
    return t


# ---------- figures ----------

def figures(ev: dict, ph: dict, ex: dict, out: Path, dpi: int) -> dict[str, str]:
    """Write every figure as PNG and return its Markdown image line by name."""
    m = ev["metrics"]
    ds = [d for d in DATASETS if d in m["datasets"]]
    captions = {}

    def save(fig: plt.Figure, name: str, caption: str) -> None:
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=dpi)
        plt.close(fig)
        captions[name] = caption

    fig, axes = plt.subplots(1, len(ds), figsize=(5 * len(ds), 3.6), sharey=True)
    for ax, d in zip(axes, ds):
        ks = methods_of(m["datasets"][d])
        v = [m["datasets"][d][k]["full_faithful"]["rate"] for k in ks]
        c = [m["datasets"][d][k]["full_faithful"]["ci"] for k in ks]
        ax.bar([METHODS[k] for k in ks], v, yerr=bar_err(v, c), capsize=3, color="tab:blue")
        ax.set_title(DATASETS[d])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Full faithful rate (95% CI)")
    save(fig, "full_faithful", "Full faithful rate by method with stratified bootstrap 95% CI")

    fig, axes = plt.subplots(1, len(ds), figsize=(5 * len(ds), 3.6), sharey=True)
    for ax, d in zip(axes, ds):
        ks = methods_of(m["datasets"][d])
        for i, claim in enumerate(("existence", "location", "type")):
            ax.bar([j + (i - 1) * 0.27 for j in range(len(ks))],
                   [m["datasets"][d][k]["claims"][claim]["rate"] for k in ks], width=0.27, label=claim)
        ax.set_xticks(range(len(ks)), [METHODS[k] for k in ks])
        ax.set_title(DATASETS[d])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Correct rate")
    axes[-1].legend(loc="lower right", fontsize=8)
    save(fig, "claims", "Correct rate per claim (location and type: defective images answered as defective)")

    colors = {"confidence": "tab:gray", "self_check_score": "tab:orange", "verification_score": "tab:green"}
    fig, axes = plt.subplots(1, len(ds), figsize=(5 * len(ds), 3.6), sharey=True)
    for ax, d in zip(axes, ds):
        ks = [k for k in methods_of(m["datasets"][d]) if "auroc" in m["datasets"][d][k]]
        a = [m["datasets"][d][k]["auroc"] for k in ks]
        v = [x["main"] for x in a]
        ax.bar([METHODS[k] for k in ks], v, yerr=bar_err(v, [x["ci"] for x in a]), capsize=3,
               color=[colors[x["signal"]] for x in a])
        # chance level of AUROC
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(DATASETS[d])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("AUROC for full faithful (95% CI)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    axes[-1].legend(handles, list(colors), loc="lower right", fontsize=8)
    save(fig, "auroc", "AUROC of each method's own signal: confidence (B0-B2), self-check (B3), verification (P0, P1)")

    a2 = m["a2"]
    seeds = sorted(k for k in a2 if k.startswith("seed_"))
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for i, key in enumerate(("p0", "b3", "diff")):
        x0 = a2[seeds[0]]["main"][key]
        ax.errorbar([i], [x0["value"]], yerr=bar_err([x0["value"]], [x0["ci"]]), fmt="o", color="tab:blue",
                    capsize=4, label="seed 0 (95% CI)" if i == 0 else None)
        ax.scatter([i + 0.15] * (len(seeds) - 1), [a2[s]["main"][key]["value"] for s in seeds[1:]],
                   facecolors="none", edgecolors="tab:red", label="seeds 1, 2" if i == 0 else None)
    # A2 thresholds: AUROC lower bound above 0.5, paired difference lower bound above 0
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(3), ["P0 AUROC", "B3 AUROC", "P0 - B3"])
    ax.legend(fontsize=8)
    save(fig, "a2", "A2 on MVTec AD: P0 and B3 AUROC and their paired difference, library seeds 1 and 2 as points")

    fig, axes = plt.subplots(1, len(ds), figsize=(5 * len(ds), 3.2), sharex=True)
    for ax, d in zip(axes, ds):
        names = list(m["ablation"][d])
        ax.barh(names, [m["ablation"][d][n]["delta"] for n in names], color="tab:purple")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(DATASETS[d])
        ax.invert_yaxis()
    axes[0].set_xlabel("AUROC change against full verification (same images)")
    axes[-1].set_xlabel(axes[0].get_xlabel())
    save(fig, "ablation", "Change in verification AUROC when tools are removed")

    fig, axes = plt.subplots(1, len(ds), figsize=(5 * len(ds), 3.4), sharey=True)
    for ax, d in zip(axes, ds):
        rules = [k for k in m["requery"][d] if k in REQUERY_RULES]
        for i, key in enumerate(("fixed", "broken")):
            ax.bar([j + (i - 0.5) * 0.35 for j in range(len(rules))], [m["requery"][d][k][key] for k in rules],
                   width=0.35, label=key, color=("tab:green", "tab:red")[i])
        ax.set_xticks(range(len(rules)), [METHODS[k] for k in rules])
        ax.set_title(DATASETS[d])
    axes[0].set_ylabel("Re-queried images")
    axes[-1].legend(fontsize=8)
    save(fig, "requery", "Re-queried images fixed (not full faithful to full faithful) and broken (the reverse)")

    statuses = ("all_verified", "partially_verified", "withheld")
    fig, axes = plt.subplots(1, len(ds), figsize=(5 * len(ds), 3.6), sharey=True)
    for ax, d in zip(axes, ds):
        ks = [k for k in methods_of(m["datasets"][d]) if "explanation_status" in m["datasets"][d][k]]
        for i, s in enumerate(statuses):
            ax.bar([j + (i - 1) * 0.27 for j in range(len(ks))],
                   [m["datasets"][d][k]["explanation_status"][s]["full_rate"] for k in ks], width=0.27, label=s)
        ax.set_xticks(range(len(ks)), [METHODS[k] for k in ks])
        ax.set_title(DATASETS[d])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Full faithful rate")
    axes[-1].legend(fontsize=8)
    save(fig, "status", "Full faithful rate by explanation status")

    fig, axes = plt.subplots(2, len(ds), figsize=(6 * len(ds), 6.4), sharey="row")
    # one marker and colour per method in every panel
    markers = dict(zip(METHODS, "osD^vPX*"))
    method_colors = {k: f"C{i}" for i, k in enumerate(METHODS)}
    for col, d in enumerate(ds):
        for row, key in enumerate(("full_faithful", "auroc")):
            ax = axes[row][col]
            for k in methods_of(m["datasets"][d]):
                pc = m["datasets"][d][k].get("per_category") or {}
                cats = sorted(pc)
                vals = [pc[c].get(key) for c in cats]
                if not cats or all(v is None for v in vals):
                    continue
                ax.scatter(range(len(cats)), [float("nan") if v is None else v for v in vals],
                           marker=markers[k], color=method_colors[k], label=METHODS[k], s=18)
                ax.set_xticks(range(len(cats)), cats, rotation=60, ha="right", fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_title(f"{DATASETS[d]}: {key.replace('_', ' ')}")
        axes[0][col].legend(fontsize=7, ncol=4)
    save(fig, "per_category", "Per-category full faithful rate and AUROC")

    pm = ph["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    roles = list(pm["e1"])
    bottom = [0] * len(roles)
    for c in ("cleared_normal", "cleared_in_grid", "out_of_grid", "other_error"):
        v = [pm["e1"][r]["first_output"].get(c, 0) for r in roles]
        axes[0].bar(roles, v, bottom=bottom, label=c)
        bottom = [b + x for b, x in zip(bottom, v)]
    axes[0].set_title("E1 first output (grid drawn)")
    axes[0].legend(fontsize=7)
    cols = sorted({int(c) for r in roles for c in pm["e2"][r]["columns"]})
    for i, r in enumerate(roles):
        axes[1].bar([j + (i - 0.5) * 0.35 for j in range(len(cols))],
                    [pm["e2"][r]["columns"].get(str(c), 0) for c in cols], width=0.35, label=r)
    axes[1].set_xticks(range(len(cols)), [str(c) for c in cols])
    axes[1].set_xlabel("Columns answered")
    axes[1].set_title("E2 column count")
    axes[1].legend(fontsize=7)
    kinds = list(pm["e3"])
    for i, key in enumerate(("records", "fixable", "full_faithful_after_fix")):
        axes[2].bar([j + (i - 1) * 0.27 for j in range(len(kinds))], [pm["e3"][k][key] for k in kinds], width=0.27,
                    label=key)
    axes[2].set_xticks(range(len(kinds)), kinds)
    axes[2].set_title("E3 clipped to the grid")
    axes[2].legend(fontsize=7)
    save(fig, "posthoc", "Post-hoc column-4 experiment: E1 first outputs, E2 column counts, E3 repaired answers")

    # extension: GLM-4.6V-Flash full faithful rate and AUROC per method, and the H9 paired differences
    x5 = ex["metrics"]["x5"]
    fig, axes = plt.subplots(2, len(x5), figsize=(5 * len(x5), 6.4), sharey="row")
    for col, (d, b) in enumerate(x5.items()):
        ks = [k for k in EXT_METHODS if k in b]
        v = [b[k]["full_faithful"]["rate"] for k in ks]
        axes[0][col].bar([EXT_METHODS[k] for k in ks], v, yerr=bar_err(v, [b[k]["full_faithful"]["ci"] for k in ks]),
                         capsize=3, color="tab:blue")
        axes[0][col].set_title(f"{DATASETS[d]}: full faithful")
        a = [b[k]["auroc"] for k in ks]
        axes[1][col].bar([EXT_METHODS[k] for k in ks], [x["main"] for x in a],
                         yerr=bar_err([x["main"] for x in a], [x["ci"] for x in a]), capsize=3, color="tab:green")
        # chance level of AUROC
        axes[1][col].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        axes[1][col].set_title(f"{DATASETS[d]}: AUROC of each method's own signal")
        for row in range(2):
            axes[row][col].set_ylim(0, 1)
    axes[0][0].set_ylabel("Full faithful rate (95% CI)")
    axes[1][0].set_ylabel("AUROC (95% CI)")
    save(fig, "ext_x5", "Extension, GLM-4.6V-Flash: full faithful rate and AUROC by method with 95% CI")

    h9, bounds = ex["metrics"]["h9"], {"location_correct": "location", "existence_correct": "existence",
                                       "full_faithful": "full faithful"}
    fig, axes = plt.subplots(1, len(h9), figsize=(5 * len(h9), 3.2), sharex=True)
    for ax, (d, x) in zip(axes, h9.items()):
        items = [(label, x[key]) for key, label in bounds.items()] + [("P0 AUROC", x["p0_auroc"]["main"]["diff"])]
        for i, (label, v) in enumerate(items):
            ax.errorbar([v["value"]], [i], xerr=[[v["value"] - v["ci"][0]], [v["ci"][1] - v["value"]]], fmt="o",
                        color="tab:blue", capsize=4)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(items)), [label for label, _ in items])
        ax.invert_yaxis()
        ax.set_title(DATASETS[d])
    axes[0].set_xlabel("B0-grid minus B0 (paired 95% CI)")
    axes[-1].set_xlabel(axes[0].get_xlabel())
    save(fig, "ext_h9", "Extension, Qwen B0-grid minus B0 on the same images (exploratory H9)")

    return {name: f"![{cap}](report/figures/{name}.png)\n\n*{cap}*" for name, cap in captions.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis/report.yaml")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    cfg = yaml.safe_load(Path(args.config).read_text())
    source = Path(cfg["source"]).read_text()
    issues = number_issues(source, cfg["number_check"]["blocked"], cfg["number_check"]["allowed"])
    if issues:
        sys.exit("result numbers in the source:\n" + "\n".join(issues))

    paths = [cfg["eval"], cfg["posthoc"], *cfg["sessions"], cfg["extension"], *cfg["extension_sessions"].values()]
    loaded = {p: json.loads(Path(p).read_text()) for p in paths}
    ev, ph, ex = loaded[cfg["eval"]], loaded[cfg["posthoc"]], loaded[cfg["extension"]]
    ph_cfg = yaml.safe_load(Path(ph["config_path"]).read_text())
    # the hypotheses must be read from the plan the extension evaluation was run with
    if file_sha256(Path(cfg["extension_plan"])) != ex["provenance"]["inputs"][cfg["extension_plan"]]:
        sys.exit(f"{cfg['extension_plan']} differs from the plan recorded in {cfg['extension']}")
    plan = yaml.safe_load(Path(cfg["extension_plan"]).read_text())
    inputs = {p: (file_sha256(Path(p)), loaded[p]) for p in paths}
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    commit = git_commit()[:7] + (" (uncommitted changes)" if dirty.strip() else "")

    gen = Path(cfg["generated_dir"])
    (gen / "tables").mkdir(parents=True, exist_ok=True)
    (gen / "figures").mkdir(parents=True, exist_ok=True)
    tabs = tables(ev, ph, ph_cfg, [loaded[p] for p in cfg["sessions"]], inputs, commit)
    tabs.update(extension_tables(ex, plan, {k: loaded[p] for k, p in cfg["extension_sessions"].items()}))
    for name, text in tabs.items():
        (gen / "tables" / f"{name}.md").write_text(text + "\n")
    figs = figures(ev, ph, ex, gen / "figures", cfg["figure_dpi"])

    parts = {f"table:{k}": v for k, v in tabs.items()} | {f"figure:{k}": v for k, v in figs.items()}
    note = (f"<!-- Generated by scripts/analysis/report.py from {cfg['source']} at git commit {commit}; "
            f"do not edit by hand. -->\n\n")
    Path(cfg["output"]).write_text(note + assemble(source, parts))
    print(f"{len(tabs)} tables, {len(figs)} figures -> {gen}; report -> {cfg['output']}")


def _self_check() -> None:
    # the number patterns of the real config, checked on made-up lines
    check = yaml.safe_load(Path("configs/analysis/report.yaml").read_text())["number_check"]
    blocked, allowed = check["blocked"], check["allowed"]
    section = "13.1" + "\uc808"
    assert number_issues(f"SQ1.2 and {section}, v1.17, A2, r1c4, Qwen3.5, 3 seeds, 2026", blocked, allowed) == []
    assert len(number_issues("rate 0.493 here\nup 5% there", blocked, allowed)) == 2
    assert len(number_issues("BQ1 then 0.5", blocked, allowed)) == 1

    assert assemble("a <!-- table:x --> b <!-- figure:y -->", {"table:x": "T", "figure:y": "F"}) == "a T b F"
    for parts in ({"table:x": "T"}, {"table:x": "T", "figure:y": "F", "table:z": "Z"}):
        try:
            assemble("<!-- table:x --><!-- figure:y -->", parts)
        except SystemExit:
            pass
        else:
            raise AssertionError("missing or unused parts must stop the report")

    assert f3(None) == "-" and f3(0.12345) == "0.123" and ci([0.1, 0.25]) == "[0.100, 0.250]"
    assert frac({"correct": 1, "n": 4, "rate": 0.25}) == "0.250 (1/4)"
    assert md(["a", "b"], [[1, 2]]) == "| a | b |\n|---|---|\n| 1 | 2 |"
    low, high = bar_err([0.5], [[0.4, 0.7]])
    assert pair_label("p1-b0") == "P1 - B0" and abs(low[0] - 0.1) < 1e-9 and abs(high[0] - 0.2) < 1e-9

    # made-up post-hoc result: tables and the figure count without any real result file
    fake_cfg = {"reporting": {"e1_control_changes": ["is_defective"]},
                "hypotheses": {"h2_grid": {"measure": "m2", "above": 0.5},
                               "h1_layout": {"measure": "m1", "min_columns": 4, "above": 0.5}}}
    e1 = {"n": 2, "first_output": {"cleared_normal": 1, "out_of_grid": 1}, "cleared_share": 0.5,
          "final_output_reference": {"parsed": 2}}
    fake_ph = {"metrics": {
        "grid_matches_claims_cell_boxes": True,
        "e1": {"violation": e1, "control": {**e1, "changes_vs_main_b0": {"compared": 2, "b0_unparsed_skipped": 0,
                                                                        "is_defective": 1}}},
        "e2": {r: {"n": 2, "parsed": 2, "columns": {"2": 1, "4": 1}, "columns_at_least_min": 1, "share": 0.5}
               for r in ("violation", "control")},
        "hypotheses": {"h2_grid": {"value": 0.5, "holds": False},
                       "h1_layout": {"value": 0.5, "control": 0.5, "holds": False}},
        "e3": {"b0": {"records": 2, "fixable": 1, "full_faithful_after_fix": 0}}}}
    t = posthoc_tables(fake_ph, fake_cfg)
    assert "| violation | 2 | 1 | 0 | 1 | 0 | 0.500 | 2 |" in t["posthoc_e1"]
    assert "2 columns: 1, 4 columns: 1" in t["posthoc_e2"] and "does not hold" in t["posthoc_hypotheses"]
    assert "| B0 | 2 | 1 | 0 |" in t["posthoc_e3"]

    fake_sessions = [{"metrics": {"compliance": {"b0/visa": {"first_ok": 1}}}}, {"metrics": {"compliance": {}}}]
    assert session_compliance(fake_sessions, "b0/visa") == {"first_ok": 1}
    assert outcome(True) == "holds" and outcome(False) == "does not hold" and outcome(None) == "-"
    # the GLM model name is allowed, a decimal right after it is still blocked
    assert number_issues("GLM-4.6V-Flash and InternVL3.5", blocked, allowed) == []
    assert len(number_issues("GLM-4.6V-Flash 0.31", blocked, allowed)) == 1
    print("report self-check ok")


if __name__ == "__main__":
    main()
