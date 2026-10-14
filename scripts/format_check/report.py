"""Merge format check shard records into one result file. No ground-truth metric is computed (design doc 13.3)."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clave_ad.runinfo import environment, header


def compliance(records: list[dict]) -> dict:
    """Counts for regular requests: first answer ok, ok after each retry, final violations and truncations."""
    out = {"requests": len(records), "first_ok": 0, "ok_after_retry_1": 0, "ok_after_retry_2": 0,
           "violations": 0, "violations_truncated": 0, "violations_think_block": 0, "ok_fenced": 0,
           "ok_control_char": 0, "nonfinite_requests": 0}
    for r in records:
        if r["ok"]:
            out[["first_ok", "ok_after_retry_1", "ok_after_retry_2"][r["retries"]]] += 1
            out["ok_fenced"] += r["fenced"]
            # records written before the control_char rule have no such field
            out["ok_control_char"] += r.get("control_char", False)
        else:
            out["violations"] += 1
            out["violations_truncated"] += r["attempts"][-1]["truncated"]
            out["violations_think_block"] += r["think_block"]
        out["nonfinite_requests"] += any(a["nonfinite"] for a in r["attempts"])
    return out


def metrics(records: list[dict], shards: list[dict]) -> dict:
    regular = [r for r in records if not r.get("forced_retry") and not r.get("consistency")]

    by_kind = defaultdict(list)
    for r in regular:
        by_kind[r["kind"]].append(r)
        by_kind[f"{r['kind']}/{r['dataset']}"].append(r)
    forced = defaultdict(list)
    hashes = defaultdict(dict)
    for r in records:
        if r.get("forced_retry"):
            forced[r["kind"]].append(r)
        else:
            hashes[(r["kind"], r["path"])][r["device"]] = r["output_sha256"]
    consistency = [{"kind": k, "path": p, "devices": sorted(h), "same_output": len(set(h.values())) == 1}
                   for (k, p), h in hashes.items() if len(h) > 1]
    attempts = [a for r in records for a in r["attempts"]]
    out = {
        "compliance": {k: compliance(v) for k, v in sorted(by_kind.items())},
        "forced_retry": {k: {"requests": len(v), "ok": sum(r["ok"] for r in v)} for k, v in sorted(forced.items())},
        "device_consistency": consistency,
        "generations": len(attempts),
        "generation_seconds": sum(a["seconds"] for a in attempts),
        "input_tokens": sum(a["input_tokens"] for a in attempts),
        "new_tokens": sum(a["new_tokens"] for a in attempts),
        "shards": [{k: s.get(k) for k in ("shard", "device", "gpu", "kernels", "pytorch_cuda_alloc_conf", "prompts", "kinds",
                                      "max_memory_reserved", "max_memory_allocated", "total_memory")}
                   for s in shards],
    }
    if by_kind["b0"]:
        # B3 runs only when the B0 answer parsed
        out["b3_skipped"] = len(by_kind["b0"]) - sum(r["ok"] for r in by_kind["b0"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/format_check/qwen3_5_9b.yaml")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text())
    shards = [json.loads(p.read_text()) for p in sorted((Path(cfg["output_dir"]) / "shards").glob("shard_*.json"))]
    records = [r for s in shards for r in s["records"]]
    result = header("format_check", config_path, "mvtec_ad+visa", cfg["seed"])
    # the same map source in every shard; null when no request kind used anomaly maps
    result["detector"] = shards[0].get("detector") if shards else None
    result["metrics"] = metrics(records, shards)
    result["environment"] = environment()
    result["records"] = records
    out = Path(cfg["output_dir"]) / f"{result['run_id']}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(records)} records -> {out}")


if __name__ == "__main__":
    main()
