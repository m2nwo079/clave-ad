"""Merge the VLM shards of one session into a result file, and bundle the whole request cache into a few files.

The bundle (cache.jsonl, one cached request per line, and oom.jsonl, every out-of-memory request so far) is the only
cache the kernel keeps as output and the next session restores. No ground-truth metric is computed here.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.runinfo import environment, header
from clave_ad.vlm.records import compliance


def p1_summary(records: list[dict]) -> dict:
    """Counts of the P0 check and the re-query decision, per dataset (operating-side counts only)."""
    out = {}
    for dataset in sorted({r["dataset"] for r in records}):
        rs = [r for r in records if r["dataset"] == dataset]
        checked = [r for r in rs if "p0" in r]
        out[dataset] = {
            "images": len(rs), "skipped": dict(Counter(r["skipped"] for r in rs if "skipped" in r)),
            "p0_checked": len(checked), "p0_all_pass": sum(r["p0"]["all_pass"] for r in checked),
            "p0_unverifiable_type": sum(r["p0"]["unverifiable"] for r in checked),
            "requeried": sum(r["requeried"] for r in checked),
            "requery_unparsed": sum(r.get("requery_unparsed", False) for r in checked),
            "requery_missing": sum(r["requeried"] and "requery_unparsed" not in r for r in checked),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main_run/run.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    shards = [json.loads(p.read_text()) for p in sorted((Path(cfg["output_dir"]) / "vlm_shards").glob("shard_*.json"))]
    records = [r for s in shards for r in s["records"]]
    requests = [r for r in records if r["kind"] != "p1"]

    by_kind = defaultdict(list)
    for r in requests:
        by_kind[r["kind"]].append(r)
        by_kind[f"{r['kind']}/{r['dataset']}"].append(r)
    attempts = [a for r in requests for a in r["attempts"]]
    # generations counts the attempts of every record, restored ones included; this splits requests by source
    sources = defaultdict(Counter)
    for s in shards:
        for group, counts in s["request_sources"].items():
            sources[group].update(counts)
    result = header(f"vlm_{cfg['session']}", config_path, "mvtec_ad+visa", cfg["seed"])
    result["session"] = {"name": cfg["session"], **cfg["sessions"][cfg["session"]]}
    result["detector"] = shards[0]["detector"] if shards else None
    result["metrics"] = {
        "complete": bool(shards) and all(s["complete"] for s in shards),
        "repeat_oom": any(s["repeat_oom"] for s in shards),
        "assigned_images": sum(s["assigned_images"] for s in shards),
        "oom": [o for s in shards for o in s["oom"]],
        "missing_b0": [m for s in shards for m in s["missing"]],
        "compliance": {k: compliance(v) for k, v in sorted(by_kind.items())},
        "p1": p1_summary([r for r in records if r["kind"] == "p1"]),
        "generations": len(attempts),
        "generation_seconds": sum(a["seconds"] for a in attempts),
        "input_tokens": sum(a["input_tokens"] for a in attempts),
        "new_tokens": sum(a["new_tokens"] for a in attempts),
        "request_sources": {g: {"new": c["new"], "restored": c["restored"]} for g, c in sorted(sources.items())},
        "shards": [{k: s[k] for k in ("shard", "device", "gpu", "kernels", "pytorch_cuda_alloc_conf", "prompts",
                                      "max_memory_reserved", "max_memory_allocated", "total_memory", "complete")}
                   for s in shards],
    }
    result["environment"] = environment()
    result["records"] = records
    out = Path(cfg["output_dir"]) / f"{result['run_id']}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")

    # bundle every cached request, restored or new, plus the out-of-memory history
    cache_dir, bundle = Path(cfg["cache_dir"]), Path(cfg["bundle_dir"])
    bundle.mkdir(parents=True, exist_ok=True)
    with open(bundle / "cache.jsonl", "w") as f:
        for p in sorted(cache_dir.glob("*.json")):
            f.write(json.dumps({"key": p.stem, "entry": json.loads(p.read_text())}) + "\n")
    earlier = (cache_dir / "_oom.jsonl").read_text().splitlines() if (cache_dir / "_oom.jsonl").exists() else []
    lines = sorted(set(earlier) | {json.dumps(o) for o in result["metrics"]["oom"]})
    (bundle / "oom.jsonl").write_text("".join(line + "\n" for line in lines))
    print(f"{len(records)} records (complete={result['metrics']['complete']}) -> {out}")
    sys.exit(0 if result["metrics"]["complete"] and not result["metrics"]["repeat_oom"] else 3)


if __name__ == "__main__":
    main()
