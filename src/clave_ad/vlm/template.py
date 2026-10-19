"""Model-specific chat template arguments and output decoding, shared by every VLM request (no torch needed).

The main-run Qwen settings carry enable_thinking and add_vision_id; other models set chat_template_kwargs in their
generation settings instead, so the Qwen requests and their cache keys stay as they were.
"""

from typing import Any


def template_kwargs(gen: dict) -> dict:
    """Extra apply_chat_template arguments of the model."""
    if "chat_template_kwargs" in gen:
        return dict(gen["chat_template_kwargs"])
    return {"enable_thinking": gen["enable_thinking"], "add_vision_id": gen["add_vision_id"]}


def edit_chat_template(template: str, replace: dict[str, str]) -> str:
    """Chat template with each old string replaced; each must occur exactly once, so a changed template fails."""
    for old, new in replace.items():
        assert template.count(old) == 1, f"chat template edit {old!r} matches {template.count(old)} times"
        template = template.replace(old, new)
    return template


def decode_output(processor: Any, new_ids: Any, prompt_ids: Any, gen: dict) -> str:
    """Answer text of the generated ids.

    With thought_block strip_leading_empty (Gemma 4, thinking off), the processor's response parser splits the
    output; an empty thought block is dropped and the answer is its content, while a thought block with content
    keeps the raw text so the format check counts it as a think block.
    """
    if gen.get("thought_block") != "strip_leading_empty":
        return processor.decode(new_ids, skip_special_tokens=True)
    raw = processor.decode(new_ids, skip_special_tokens=False)
    message = processor.parse_response(raw, prefix=prompt_ids)
    if (message.get("thinking") or "").strip():
        return raw
    return message.get("content") or ""


def _self_check() -> None:
    qwen = {"enable_thinking": False, "add_vision_id": True, "max_new_tokens": 512}
    # the main-run arguments, unchanged
    assert template_kwargs(qwen) == {"enable_thinking": False, "add_vision_id": True}
    gemma = {"chat_template_kwargs": {"enable_thinking": False}, "thought_block": "strip_leading_empty"}
    assert template_kwargs(gemma) == {"enable_thinking": False}
    assert template_kwargs({"chat_template_kwargs": {}}) == {}

    class FakeProcessor:
        """Decodes a list of strings; parse_response splits a made-up thought block."""

        def decode(self, ids: list[str], skip_special_tokens: bool) -> str:
            text = "".join(ids)
            return text.replace("<|channel>thought\n", "").replace("<channel|>", "") if skip_special_tokens else text

        def parse_response(self, raw: str, prefix: Any) -> dict:
            assert prefix == "prompt"
            if raw.startswith("<|channel>thought\n"):
                thinking, content = raw[len("<|channel>thought\n"):].split("<channel|>", 1)
                return {"role": "assistant", "thinking": thinking, "content": content}
            return {"role": "assistant", "content": raw}

    p = FakeProcessor()
    answer = '{"is_defective": false}'
    assert decode_output(p, ["<|channel>thought\n", "<channel|>", answer], "prompt", gemma) == answer
    assert decode_output(p, [answer], "prompt", gemma) == answer
    thought = ["<|channel>thought\n", "look at r1c1", "<channel|>", answer]
    assert decode_output(p, thought, "prompt", gemma) == "".join(thought)
    # without the setting the plain decode is used, as in the main run
    assert decode_output(p, [answer], "prompt", qwen) == answer

    template = "{%- if messages[0]['role'] != 'system' and default != '' %}SYS{%- endif %}"
    edited = edit_chat_template(template, {"messages[0]['role'] != 'system' and default != ''": "false"})
    assert edited == "{%- if false %}SYS{%- endif %}"
    assert edit_chat_template(template, {}) == template
    # a missing or repeated string is an error, not a silent no-op
    for bad in ("absent", "'"):
        try:
            edit_chat_template(template, {bad: "x"})
        except AssertionError:
            continue
        raise AssertionError(f"edit {bad!r} was accepted")
    print("template self-check ok")


if __name__ == "__main__":
    _self_check()
