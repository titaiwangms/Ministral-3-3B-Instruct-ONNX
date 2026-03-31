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

1. **Vision model** (`model-vision.onnx`) — Pixtral vision encoder + multi-modal projector. Takes raw pixel values and outputs image features in the text embedding space.
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
- **`test/test_vision_export.py`** — End-to-end vision export: builds the ONNX model, loads it in ONNX Runtime, validates shapes and parity
- **`test/test_embedding_export.py`** — End-to-end embedding export: builds the ONNX model, validates dynamic sequence lengths, masked_scatter correctness

All tests use `--no_weights` (random initialization) and require no model download.

### Architecture overview

`Mistral3ForConditionalGeneration` is composed of:

- **`vision_tower`** — `PixtralVisionModel` that accepts raw pixel values and returns patch hidden states.
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

The only code change needed is to `Mistral3PatchMerger.forward()`. The original uses Python for-loops and list comprehensions that `torch.export` cannot trace.

**What was rewritten:**

`Mistral3PatchMerger` — Replaced with a dual-path implementation:
- `_forward_export()`: Pure tensor operations (`unfold` + reshape) — used during ONNX export via `torch.compiler.is_exporting()` dispatch.
- `_forward_eager()`: Original for-loop behavior — used during normal inference.
- Both paths produce numerically identical results (validated by tests).

**What was NOT rewritten:**

- `generate_block_attention_mask` — With static shapes (single 448×448 image), dynamo unrolls the single-iteration loop at compile time. No changes needed.
- `get_image_features` `.tolist()` — With static `image_sizes`, dynamo evaluates `.tolist()` at compile time. `builder.py` also works around this with a custom `_get_image_features_onnx` wrapper.

### ONNX export fixes

Three issues were resolved to make the export work:

1. **`model.model.xxx` attribute path** — `AutoModel.from_config()` returns `Mistral3Model` (not `Mistral3ForConditionalGeneration`), so `multi_modal_projector` is directly on `model`, not `model.model`. The `patch_model_for_onnx_export()` function handles both layouts.

2. **`image_sizes=None` guard** — Passing `image_sizes` as a tensor to `PixtralVisionModel.forward` triggers `GuardOnDataDependentSymNode` because the model slices using tensor values. Fix: `builder.py` passes `image_sizes=None` to the vision tower and constructs `image_sizes` from `pixel_values.shape` (Python ints) for the PatchMerger.

3. **`torch._check` guards** — The `unfold` operation in `_forward_export` requires explicit guards (`torch._check`) to prove that spatial dimensions are non-zero and large enough for the kernel. Without these, `torch.export` raises shape constraint errors.
