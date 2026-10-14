"""Re-judge the raw outputs of an earlier format check run with the current parser, without new generations.

Each record of the source result is matched to its cached raw outputs by the final output hash and the token counts
of its attempts. The first attempt that passes the current parser decides the record; later attempts are dropped
because they would not have been requested.
"""

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clave_ad.runinfo import environment, header
from clave_ad.vlm.parse import has_control_char, has_think_block, parse_answer, parse_check, unwrap
from report import metrics

LIMITATION = ("B3 was generated from the B0 answer the source run accepted. Where the source run needed a retry for "
              "B0, B3 saw the answer after the retry, not the first answer that the current parser accepts.")


def match_key(texts_or_hash: str, attempts: list[dict]) -> tuple:
    return texts_or_hash, tuple((a["input_tokens"], a["new_tokens"]) for a in attempts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/format_check/qwen3_5_9b.yaml")
    parser.add_argument("--source", required=True, help="results/format_check/format_check_<run_id>.json to re-judge")
    args = parser.parse_args()
    config_path, source_path = Path(args.config), Path(args.source)
    cfg = yaml.safe_load(config_path.read_text())
    types = yaml.safe_load(Path(cfg["data"]["type_names"]).read_text())
    source = json.loads(source_path.read_text())

    cache = {}
    for p in Path(cfg["cache_dir"]).glob("*.json"):
        entry = json.loads(p.read_text())
        if entry["attempts"]:
            final = hashlib.sha256(entry["attempts"][-1]["text"].encode()).hexdigest()
            cache[match_key(final, entry["attempts"])] = entry

    def entry_of(record: dict) -> dict:
        key = match_key(record["output_sha256"], record["attempts"])
        if key not in cache:
            sys.exit(f"no cached raw output for {record['kind']} {record['path']}")
        return cache[key]

    # B3 is judged against the B0 answer it was actually given
    b0_given = {r["path"]: entry_of(r)["parsed"] for r in source["records"]
                if r["kind"] == "b0" and not r.get("forced_retry") and not r.get("consistency")}

    records = []
    for r in source["records"]:
        entry = entry_of(r)
        if r["kind"] == "b3":
            parse = partial(parse_check, answer=b0_given[r["path"]])
        else:
            parse = partial(parse_answer, candidates=list(types[r["dataset"]]["categories"][r["category"]]))
        attempts = [{**a, "error": parse(e["text"])[1]} for a, e in zip(r["attempts"], entry["attempts"])]
        passing = [i for i, a in enumerate(attempts) if a["error"] is None]
        # a forced retry has exactly one generated attempt; otherwise the first passing attempt ends the request
        last = len(attempts) - 1 if r.get("forced_retry") or not passing else passing[0]
        text = entry["attempts"][last]["text"]
        records.append({
            **r, "ok": attempts[last]["error"] is None, "final_error": attempts[last]["error"],
            "retries": last + (1 if r.get("forced_retry") else 0), "attempts": attempts[:last + 1],
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "fenced": unwrap(text)[1], "control_char": has_control_char(text), "think_block": has_think_block(text),
        })

    result = header("format_check_rejudge", config_path, source["dataset"], source["seed"])
    result["source"] = {
        "result_file": str(source_path), "run_id": source["run_id"], "run_git_commit": source["git_commit"],
        "raw_outputs": f"{cfg['cache_dir']} (local, not tracked)",
    }
    result["metrics"] = metrics(records, source["metrics"]["shards"])
    # generation counts and times describe the source run, not the trimmed attempts above
    for key in ("generations", "generation_seconds", "input_tokens", "new_tokens"):
        result["metrics"][key] = source["metrics"][key]
    result["notes"] = [LIMITATION, "Generation counts, times and memory are those of the source run."]
    result["environment"] = environment()
    result["records"] = records
    out = Path(cfg["output_dir"]) / f"{result['run_id']}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(records)} records re-judged -> {out}")


if __name__ == "__main__":
    main()
