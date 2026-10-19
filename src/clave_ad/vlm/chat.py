"""One VLM request with greedy generation, a NaN/inf check on raw lm_head outputs, and the format retry."""

import importlib.util
import time
from string import Template
from typing import Any, Callable

import torch
from PIL import Image

from clave_ad.vlm.template import decode_output, template_kwargs


class NonFiniteHook:
    """Flags NaN or inf in lm_head outputs, before any logits processor masks tokens with -inf."""

    def __init__(self, model: Any) -> None:
        self.seen = False
        model.get_output_embeddings().register_forward_hook(self._hook)

    def _hook(self, module: Any, inputs: Any, output: torch.Tensor) -> None:
        if not self.seen and not bool(torch.isfinite(output).all()):
            self.seen = True


def kernel_implementations() -> dict[str, bool]:
    """Whether the optional fast Gated DeltaNet kernels are importable; transformers uses them when present."""
    return {name: importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")}


def render(template: str, **values: str) -> str:
    # $-placeholders, so JSON braces in prompts need no escaping
    return Template(template).substitute(values)


def generate(model: Any, processor: Any, images: list[Image.Image], messages: list[dict], gen: dict, hook: NonFiniteHook) -> dict:
    """Run one greedy generation for a chat whose first user turn holds all images."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs(gen))
    inputs = processor(text=[text], images=images or None, return_tensors="pt").to(model.device)
    hook.seen = False
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=gen["max_new_tokens"], do_sample=gen["do_sample"],
                            eos_token_id=gen["eos_token_ids"])
    torch.cuda.synchronize()
    new_ids = output[0, inputs["input_ids"].shape[1]:]
    ended = bool(new_ids.numel()) and int(new_ids[-1]) in gen["eos_token_ids"]
    return {
        "text": decode_output(processor, new_ids, inputs["input_ids"][0], gen),
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "new_tokens": int(new_ids.numel()),
        "seconds": time.perf_counter() - start,
        # stopped by the token limit instead of an end token
        "truncated": not ended,
        "nonfinite": hook.seen,
    }


def request(model: Any, processor: Any, images: list[Image.Image], prompt: str, parse: Callable[[str], tuple],
            gen: dict, retry_template: str, hook: NonFiniteHook, forced_output: str | None = None) -> dict:
    """Ask once and retry up to max_retries times on a format violation, showing only the latest output.

    With forced_output, that broken text stands in for the first output and exactly one retry is run.
    """
    first = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]}]
    attempts = []
    if forced_output is None:
        result = generate(model, processor, images, first, gen, hook)
        parsed, error = parse(result["text"])
        attempts.append({**result, "error": error})
        retries = gen["max_retries"]
    else:
        result = {"text": forced_output}
        parsed, error = parse(forced_output)
        retries = 1
    for _ in range(retries):
        if parsed is not None:
            break
        messages = first + [
            {"role": "assistant", "content": [{"type": "text", "text": result["text"]}]},
            {"role": "user", "content": [{"type": "text", "text": render(retry_template, error=error)}]},
        ]
        result = generate(model, processor, images, messages, gen, hook)
        parsed, error = parse(result["text"])
        attempts.append({**result, "error": error})
    return {"parsed": parsed, "error": error, "attempts": attempts}
