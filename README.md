# Ministral-3-3B-Instruct-ONNX

ONNX export pipeline for [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512).

Exports the vision and embedding components of the Mistral 3 multimodal architecture to ONNX, for use with ONNX Runtime.

### Requirements

| Package      | Version              |
|--------------|----------------------|
| onnxscript   | nightly              |
| torch        | nightly              |
| onnx         | 1.19.1               |
| transformers | nightly (5.0.0/dev0) |
| onnxruntime  | ≥ 1.18 (for tests)   |

### Goals

We export 2 models from the Ministral-3-3B multimodal architecture:

1. **Vision model** (`model-vision.onnx`) — Pixtral vision encoder + multi-modal projector. Takes `pixel_values` of shape `[1, 3, H, W]` (dynamic H/W, multiples of 28) and outputs image features in the text embedding space.
2. **Embedding model** (`model-embedding.onnx`) — Token embedding layer with image feature fusion via `masked_scatter`. Takes `input_ids` + `image_features` and outputs fused `inputs_embeds`.

### Build

Export both models with pretrained weights:

```bash
python builder.py \
    -i /path/to/Ministral-3-3B-Instruct-2512 \
    -o ./output \
    -p fp32 \
    -e cpu
```

Export with random weights (no download required, useful for testing):

```bash
python builder.py \
    --no_weights \
    -i . \
    -o ./output \
    -p fp32 \
    -e cpu
```

Export a single component:

```bash
# Vision only
python builder.py --no_weights --part vision -i . -o ./output -p fp32 -e cpu

# Embedding only
python builder.py --no_weights --part embedding -i . -o ./output -p fp32 -e cpu
```

`builder.py` runs a parity check after each export, comparing the ONNX output against the PyTorch reference with `atol=0.01, rtol=0.01`.

### Tests

Run the full test suite:

```bash
pytest test/ -v
```

The test suite covers:
- **`modeling_code/test_modeling_mistral3.py`** — Unit tests for the PatchMerger rewrite (export vs. eager parity, shape correctness, weight preservation, gradient flow)
- **`test/test_vision_export.py`** — End-to-end vision export: builds the ONNX model, loads it in ONNX Runtime, validates shapes, dynamic H/W support, and parity
- **`test/test_embedding_export.py`** — End-to-end embedding export: builds the ONNX model, validates dynamic sequence lengths, masked_scatter correctness

All tests use `--no_weights` (random initialization) and require no model download.

### Architecture overview

`Mistral3ForConditionalGeneration` is composed of:

- **`vision_tower`** — `PixtralVisionModel` that accepts `pixel_values [1, 3, H, W]` and returns patch hidden states. H and W are dynamic (multiples of `patch_size × spatial_merge_size = 28`, range `[28, image_size]`).
- **`multi_modal_projector`** — `Mistral3MultiModalProjector` (PatchMerger + MLP) that maps vision features to the text hidden-size space.
- **`language_model`** — Standard Mistral decoder (not exported here).

Compared to Qwen2.5-VL:

1. Vision inputs are `(pixel_values [B, 3, H, W], image_sizes [B, 2])` rather than `(pixel_values, image_grid_thw)`.
2. The vision encoder uses standard self-attention — no custom `PackedAttention` / `cu_seqlens` operator needed.
3. Image–text fusion uses `masked_scatter` (token-id matching) rather than a dedicated fusion method.

### Project structure

```
builder.py                 # ONNX export pipeline (vision + embedding)
demo.py                    # HuggingFace inference demo (PyTorch, not ONNX)
modeling_code/
  __init__.py              # Public API: patch_model_for_onnx_export()
  modeling_mistral3.py     # ONNX-friendly PatchMerger (export + eager paths)
  modeling_mistral3_original.py  # Original HF code (kept for diff reference)
  test_modeling_mistral3.py      # Unit tests for PatchMerger rewrite
test/
  test_vision_export.py    # End-to-end vision ONNX export tests
  test_embedding_export.py # End-to-end embedding ONNX export tests
```

### Key rewrites (see `modeling_code/`)

Two components were rewritten for ONNX export compatibility with dynamic shapes:

**1. `Mistral3PatchMerger`** — Replaced with a dual-path implementation:
- `_forward_export()`: Pure tensor operations (`unfold` + reshape) — used during ONNX export via `torch.compiler.is_exporting()` dispatch.
- `_forward_eager()`: Original for-loop behavior — used during normal inference.
- Both paths produce numerically identical results (validated by tests).

**2. `PixtralVisionModel.forward`** — Replaced with export-aware dispatch:
- Skips `generate_block_attention_mask` — with batch=1 (single image), the attention mask is trivially all-zeros (full attention), so `attention_mask=None` is passed instead.
- Computes position IDs inline with `torch.arange`/`meshgrid` — replaces `position_ids_in_meshgrid` which uses a Python for-loop over the batch.
- Supports dynamic H/W (any multiple of `patch_size`).

**What was NOT rewritten:**

- `get_image_features` `.tolist()` — `builder.py`'s `_get_image_features_onnx` wrapper calls vision components directly, bypassing this.

### ONNX export fixes

Three issues were resolved to make the export work:

1. **`model.model.xxx` attribute path** — `AutoModel.from_config()` returns `Mistral3Model` (not `Mistral3ForConditionalGeneration`), so `multi_modal_projector` is directly on `model`, not `model.model`. The `patch_model_for_onnx_export()` function handles both layouts.

2. **`generate_block_attention_mask` dynamic shapes** — The original uses `torch.tensor()` from symbolic values and Python for-loops with symbolic bounds, which `torch.export` cannot trace under dynamic shapes. Fix: with batch=1, the mask is trivially all-zeros (single image = full attention), so we skip it entirely and pass `attention_mask=None`.

3. **`torch._check` guards** — The `unfold` operation in `_forward_export` requires explicit guards (`torch._check`) to prove that spatial dimensions are non-zero and large enough for the kernel. Without these, `torch.export` raises shape constraint errors.

4. **`position_ids_in_meshgrid` for-loop** — The original iterates over the batch with a Python for-loop. Fix: compute position IDs inline with `torch.arange`/`meshgrid` for the single image (batch=1).
