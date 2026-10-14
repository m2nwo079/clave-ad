from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


def load_model(cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Load the 4-bit VLM and its processor from the `model` config block."""
    compute_dtype = getattr(torch, cfg["compute_dtype"])
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["name"],
        revision=cfg["revision"],
        quantization_config=quantization,
        dtype=compute_dtype,
        device_map=cfg["device_map"],
    )
    processor = AutoProcessor.from_pretrained(
        cfg["name"],
        revision=cfg["revision"],
        use_fast=cfg["processor_use_fast"],
    )
    return model, processor


def build_inputs(
    processor: Any,
    images: list[Image.Image],
    prompt: str,
    enable_thinking: bool,
    add_vision_id: bool,
) -> Any:
    """Build model inputs for one user turn: all images first, then the prompt text."""
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        add_vision_id=add_vision_id,
    )
    return processor(text=[text], images=images, return_tensors="pt")
