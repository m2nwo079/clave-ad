from typing import Any

import torch
from PIL import Image
from transformers import AttentionInterface, AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.integrations.sdpa_attention import repeat_kv
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

from clave_ad.vlm.template import edit_chat_template, template_kwargs


def sdpa_repeat_kv(module: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   attention_mask: torch.Tensor | None, dropout: float = 0.0, scaling: float | None = None,
                   is_causal: bool | None = None, **kwargs: Any) -> tuple[torch.Tensor, None]:
    """sdpa with the key and value heads repeated to the query heads first.

    On T4 (no flash attention) a grouped-query sdpa call falls back to the math kernel, which builds float32
    attention matrices of sequence length squared per head; with equal head counts the memory-efficient kernel runs.
    Same causal rule and output layout as transformers sdpa_attention_forward; the repeat follows the tensor shapes,
    not module.num_key_value_groups, so it never repeats twice.
    """
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal
    # looked up at call time, so the memory test can wrap it to record the kernel choice
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=scaling, is_causal=is_causal)
    return output.transpose(1, 2).contiguous(), None


AttentionInterface.register("sdpa_repeat_kv", sdpa_repeat_kv)
# masks are built per attention implementation name; without this entry every mask is None (plain causal)
ALL_MASK_ATTENTION_FUNCTIONS.register("sdpa_repeat_kv", sdpa_mask)


def load_model(cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Load the 4-bit VLM and its processor from the `model` config block.

    Optional keys, absent in the main-run Qwen config (which keeps the defaults): skip_4bit_modules (patterns of
    modules kept in compute_dtype; it replaces the transformers default, so lm_head must be listed),
    processor_kwargs, chat_template_replace (old: new strings of the processor chat template, each found once) and
    attn_implementation (absent keeps the transformers default).
    """
    compute_dtype = getattr(torch, cfg["compute_dtype"])
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=cfg.get("skip_4bit_modules"),
    )
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["name"],
        revision=cfg["revision"],
        quantization_config=quantization,
        dtype=compute_dtype,
        device_map=cfg["device_map"],
        # passed only when set: transformers assigns the keyword to the config even when it is None, so a config
        # without it (Qwen) makes exactly the call it made before
        **({"attn_implementation": cfg["attn_implementation"]} if "attn_implementation" in cfg else {}),
    )
    processor = AutoProcessor.from_pretrained(
        cfg["name"],
        revision=cfg["revision"],
        use_fast=cfg["processor_use_fast"],
        **cfg.get("processor_kwargs", {}),
    )
    if "chat_template_replace" in cfg:
        processor.chat_template = edit_chat_template(processor.chat_template, cfg["chat_template_replace"])
    return model, processor


def build_inputs(
    processor: Any,
    images: list[Image.Image],
    prompt: str,
    gen: dict,
) -> Any:
    """Build model inputs for one user turn: all images first, then the prompt text."""
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs(gen),
    )
    return processor(text=[text], images=images, return_tensors="pt")


def _self_check() -> None:
    """sdpa_repeat_kv gives the transformers sdpa output for grouped, masked and vision-style calls (CPU)."""
    from types import SimpleNamespace

    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    torch.manual_seed(0)
    causal_mask = torch.tril(torch.ones(7, 7, dtype=torch.bool))[None, None]
    block_mask = torch.zeros(1, 1, 7, 7)
    block_mask[..., :3, 3:] = float("-inf")
    block_mask[..., 3:, :3] = float("-inf")
    # (query heads, key-value heads, query length, mask, causal): Ministral prefill and decode, Gemma full layer,
    # vision encoder
    for heads, kv_heads, length, mask, causal in ((32, 8, 7, None, True), (32, 8, 1, None, True),
                                                  (16, 1, 7, causal_mask, True), (16, 16, 7, block_mask, False)):
        module = SimpleNamespace(num_key_value_groups=heads // kv_heads, is_causal=causal, layer_idx=0)
        q = torch.randn(1, heads, length, 16)
        k, v = torch.randn(2, 1, kv_heads, 7 if length > 1 else 5, 16)
        reference, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=0.25)
        output, _ = sdpa_repeat_kv(module, q, k, v, mask, scaling=0.25)
        assert output.shape == reference.shape == (1, length, heads, 16)
        assert torch.allclose(output, reference, atol=1e-5), (heads, kv_heads, length)

    # whole tiny text models, so the masks built from the implementation name are compared as well: Gemma 4 with
    # an input longer than its sliding window, Gemma 4 with an image block (bidirectional on sliding layers) and
    # Ministral 3 (plain causal)
    from transformers.models.gemma4_unified import modeling_gemma4_unified as gemma4
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
    from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
    from transformers.models.ministral3.modeling_ministral3 import Ministral3Model

    ids = torch.randint(0, 300, (1, 20))
    block = torch.full((1, 20), -1)
    block[0, 3:13] = 0
    tiny = dict(hidden_size=64, intermediate_size=128, vocab_size=300, num_attention_heads=4, head_dim=16)
    cases = [(gemma4.Gemma4UnifiedTextModel(Gemma4UnifiedTextConfig(
        **tiny, num_hidden_layers=6, vocab_size_per_layer_input=300, sliding_window=window, num_key_value_heads=2,
        use_bidirectional_attention="vision", attention_k_eq_v=True,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        per_layer_config={"05": {"head_dim": 32, "num_key_value_heads": 1}})), image)
        for window, image in ((8, False), (32, True))]
    cases.append((Ministral3Model(Ministral3Config(**tiny, num_hidden_layers=2, num_key_value_heads=1)), False))
    for model, image in cases:
        outputs = []
        for implementation in ("sdpa", "sdpa_repeat_kv"):
            model.config._attn_implementation = implementation
            masks = gemma4.create_masks_for_generate(
                config=model.config, inputs_embeds=model.embed_tokens(ids), attention_mask=None,
                past_key_values=None, position_ids=torch.arange(20)[None], block_sequence_ids=block) if image else None
            with torch.no_grad():
                outputs.append(model.eval()(input_ids=ids, attention_mask=masks).last_hidden_state)
        assert torch.allclose(*outputs, atol=1e-5), (type(model).__name__, image)

    # GLM-4.6V-Flash: text model with grouped-query attention and multimodal rotary positions, and the vision encoder
    # on two images (attention per image, is_causal passed as False)
    from transformers.models.glm4v import modeling_glm4v as glm4v
    from transformers.models.glm4v.configuration_glm4v import Glm4vTextConfig, Glm4vVisionConfig

    rope = {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 1, 1], "partial_rotary_factor": 0.5}
    text = glm4v.Glm4vTextModel(Glm4vTextConfig(**{k: v for k, v in tiny.items() if k != "head_dim"},
                                                num_hidden_layers=2, num_key_value_heads=1, rope_parameters=rope))
    vision = glm4v.Glm4vVisionModel(Glm4vVisionConfig(depth=2, hidden_size=64, intermediate_size=128, num_heads=4,
                                                      out_hidden_size=64, patch_size=14, image_size=56))
    pixels = torch.randn(32, 3 * 2 * 14 * 14)
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    for model, call in ((text, lambda m: m(input_ids=ids).last_hidden_state),
                        (vision, lambda m: m(pixels, grid_thw=grid).pooler_output)):
        outputs = []
        for implementation in ("sdpa", "sdpa_repeat_kv"):
            model.config._attn_implementation = implementation
            with torch.no_grad():
                outputs.append(call(model.eval()))
        assert torch.allclose(*outputs, atol=1e-5), type(model).__name__
    print("model self-check ok")


if __name__ == "__main__":
    _self_check()
