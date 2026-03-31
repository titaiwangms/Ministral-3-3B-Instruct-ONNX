# Ministral-3-3B-Instruct-ONNX
ONNX Graph of [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)

### Requirements

| Package      | Version  |
|--------------|----------|
| onnxscript   | nightly  |
| torch        | nightly  |
| onnx         | 1.19.1   |
| transformers | nightly (5.0.0/dev0) |

### Goals

We need to provide 2 models for the Ministral-3-3B multimodal architecture:

1. A **vision model** (`model-vision.onnx`) that encodes images via the Pixtral vision encoder
   and projects the features into the text embedding space.
2. An **embedding model** (`model-embedding.onnx`) that fuses token embeddings with
   vision features using a masked-scatter operation.

### Architecture overview

`Mistral3ForConditionalGeneration` is composed of:

- `vision_tower` – a `PixtralVisionModel` that accepts raw pixel values and returns patch
  hidden states.
- `multi_modal_projector` – `Mistral3MultiModalProjector` (patch merger + MLP) that maps
  vision features to the text hidden-size space.
- `language_model` – a standard Mistral decoder.

Compared to Qwen2.5-VL, key differences are:

1. Vision inputs are `(pixel_values [B, 3, H, W], image_sizes [B, 2])` rather than
   `(pixel_values, image_grid_thw)`.
2. The vision encoder uses standard (non-windowed) self-attention, so no custom
   `PackedAttention` / `cu_seqlens` operator replacement is required.
3. Image–text fusion is performed via `masked_scatter` (token-id matching) rather than a
   dedicated `get_fused_input_embeddings` method.

### Key rewrites (see `modeling_code/`)

1. `generate_block_attention_mask` in `PixtralVisionModel` uses a Python for-loop that must
   be rewritten for ONNX export.
2. `Mistral3PatchMerger.forward` uses list comprehensions and `split(...).tolist()` which
   require static image sizes or symbolic rewriting.
3. `get_image_features` in `Mistral3Model` calls `.tolist()` on a tensor derived from
   `image_sizes`; a fixed image size avoids this issue.
