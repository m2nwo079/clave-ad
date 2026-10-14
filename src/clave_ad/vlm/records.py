"""Request records shared by the format check and the main run: a summary without raw text, and format compliance."""

import hashlib

from clave_ad.vlm.parse import has_control_char, has_think_block, unwrap


def summarize(kind: str, image: dict, outcome: dict, device: str, **flags: bool) -> dict:
    """Shard record without raw text; the final output is kept as a hash for cross-device comparison."""
    attempts = [{k: a[k] for k in ("input_tokens", "new_tokens", "seconds", "truncated", "nonfinite", "error")}
                for a in outcome["attempts"]]
    final_text = outcome["attempts"][-1]["text"] if outcome["attempts"] else ""
    return {
        "kind": kind, "dataset": image["dataset"], "category": image["category"], "path": image["path"],
        "device": device, "ok": outcome["parsed"] is not None, "final_error": outcome["error"],
        "retries": len(outcome["attempts"]) - (0 if flags.get("forced_retry") else 1),
        "output_sha256": hashlib.sha256(final_text.encode()).hexdigest(), "attempts": attempts,
        # tolerated wrapper and the think-tag violation, counted separately
        "fenced": unwrap(final_text)[1], "control_char": has_control_char(final_text),
        "think_block": has_think_block(final_text), **flags,
    }


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
