"""Strict parsing of VLM answers.

The answer must be one JSON object. Tolerated and counted separately: a single code fence around the whole answer
(removed, fenced) and control characters such as raw newlines inside strings (control_char).
Other surrounding text, several objects or think tags are violations.
"""

import json
import re
ANSWER_KEYS = {"is_defective", "defect_type", "location_cells", "confidence", "rationale"}
CHECK_KEYS = {"existence", "location", "type"}
CELL = re.compile(r"r[1-3]c[1-3]")
FENCE = re.compile(r"```(?:json)?\n(.*)\n```", re.DOTALL)
THINK_TAGS = ("<think>", "</think>")


def unwrap(text: str) -> tuple[str, bool]:
    """Strip whitespace and one code fence around the whole answer; the flag says whether a fence was removed."""
    stripped = text.strip()
    match = FENCE.fullmatch(stripped)
    return (match.group(1), True) if match else (stripped, False)


def has_think_block(text: str) -> bool:
    return any(tag in text for tag in THINK_TAGS)


def has_control_char(text: str) -> bool:
    """True when the answer parses only because control characters inside strings are allowed."""
    body = unwrap(text)[0]
    try:
        json.loads(body)
        return False
    except json.JSONDecodeError:
        pass
    try:
        json.loads(body, strict=False)
        return True
    except json.JSONDecodeError:
        return False


def _load_object(text: str) -> tuple[dict | None, str | None]:
    if has_think_block(text):
        return None, "the answer must not contain <think> or </think> tags"
    try:
        # strict=False allows control characters such as raw newlines inside strings
        obj = json.loads(unwrap(text)[0], strict=False)
    except json.JSONDecodeError as e:
        return None, f"not a single JSON object ({e.msg} at char {e.pos})"
    if not isinstance(obj, dict):
        return None, "the answer is JSON but not an object"
    return obj, None


def _key_error(obj: dict, keys: set[str]) -> str | None:
    if set(obj) != keys:
        missing, extra = sorted(keys - set(obj)), sorted(set(obj) - keys)
        return f"keys must be exactly {sorted(keys)} (missing {missing}, unexpected {extra})"
    return None


def parse_answer(text: str, candidates: list[str]) -> tuple[dict | None, str | None]:
    """Validate a B0/B1 answer. Returns (answer, None) or (None, error message for the retry prompt)."""
    obj, error = _load_object(text)
    if error:
        return None, error
    if error := _key_error(obj, ANSWER_KEYS):
        return None, error
    defective, dtype, cells, conf = obj["is_defective"], obj["defect_type"], obj["location_cells"], obj["confidence"]
    if not isinstance(defective, bool):
        return None, '"is_defective" must be true or false'
    if not isinstance(cells, list) or not all(isinstance(c, str) and CELL.fullmatch(c) for c in cells):
        return None, '"location_cells" must be a list of cells from "r1c1" to "r3c3"'
    if len(set(cells)) != len(cells):
        return None, '"location_cells" must not repeat a cell'
    # bool is a subclass of int, so it is excluded explicitly
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        return None, '"confidence" must be a number between 0 and 1'
    if not isinstance(obj["rationale"], str):
        return None, '"rationale" must be a string'
    if defective:
        if dtype not in candidates:
            return None, '"defect_type" must be one of the candidate defect types'
        if not cells:
            return None, '"location_cells" must not be empty when "is_defective" is true'
    elif dtype is not None or cells:
        return None, 'when "is_defective" is false, "defect_type" must be null and "location_cells" empty'
    return obj, None


def parse_check(text: str, answer: dict) -> tuple[dict | None, str | None]:
    """Validate a B3 self-check of `answer`. Location and type are null exactly when the answer is not defective."""
    obj, error = _load_object(text)
    if error:
        return None, error
    if error := _key_error(obj, CHECK_KEYS):
        return None, error
    if not isinstance(obj["existence"], bool):
        return None, '"existence" must be true or false'
    for key in ("location", "type"):
        value = obj[key]
        if answer["is_defective"] and not isinstance(value, bool):
            return None, f'"{key}" must be true or false'
        if not answer["is_defective"] and value is not None:
            return None, f'"{key}" must be null because the answer says not defective'
    return obj, None


def _self_check() -> None:
    cands = ["scratch", "crack"]
    ok = '{"is_defective": true, "defect_type": "scratch", "location_cells": ["r1c3"], "confidence": 0.8, "rationale": "x"}'
    assert parse_answer(ok, cands)[0] is not None
    assert parse_answer("```json\n" + ok + "\n```", cands)[0] is not None
    assert unwrap("```json\n" + ok + "\n```")[1] and not unwrap(ok)[1]
    assert parse_answer("Answer:\n```json\n" + ok + "\n```", cands)[0] is None
    assert parse_answer("```json\n" + ok + "\n```\n```json\n" + ok + "\n```", cands)[0] is None
    assert parse_answer(ok + "\n" + ok, cands)[0] is None
    assert parse_answer("<think>\n\n</think>\n\n" + ok, cands)[0] is None
    raw_newline = ok.replace('"x"}', '"x.\n"}')
    assert parse_answer(raw_newline, cands)[0] is not None
    assert has_control_char(raw_newline) and not has_control_char(ok)
    assert parse_answer(ok.replace('"scratch"', '"dent"'), cands)[0] is None
    assert parse_answer(ok.replace("0.8", "true"), cands)[0] is None
    assert parse_answer(ok.replace('"r1c3"', '"r4c1"'), cands)[0] is None
    normal = '{"is_defective": false, "defect_type": null, "location_cells": [], "confidence": 0.9, "rationale": "x"}'
    assert parse_answer(normal, cands)[0] is not None
    assert parse_check('{"existence": true, "location": null, "type": null}', {"is_defective": False})[0] is not None
    assert parse_check('{"existence": true, "location": null, "type": null}', {"is_defective": True})[0] is None


if __name__ == "__main__":
    _self_check()
    print("parse self-check ok")
