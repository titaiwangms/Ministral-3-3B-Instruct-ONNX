"""Real-weight tests for ONNX export of Ministral-3-3B-Instruct-2512.

These tests require:
- GPU with CUDA support
- ~6GB disk space for model download
- mistralai/Ministral-3-3B-Instruct-2512 accessible from HuggingFace

Run with: pytest test/test_real_weights.py -v
"""

import os
import sys
import tempfile

import numpy as np
import onnxruntime as ort
import pytest
import torch
from transformers import AutoModel, AutoConfig, FineGrainedFP8Config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from modeling_code import patch_model_for_onnx_export

MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"
PRECISION = torch.float16


@pytest.fixture(scope="module")
def real_model():
    """Load the real model once for all tests (with FP8 dequantization)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    config = AutoConfig.from_pretrained(MODEL_ID)

    quantization_config = None
    if getattr(config, "quantization_config", None) is not None:
        quant_method = config.quantization_config.get("quant_method", "")
        if quant_method == "fp8":
            quantization_config = FineGrainedFP8Config(dequantize=True)

    model = AutoModel.from_pretrained(
        MODEL_ID,
        attn_implementation="sdpa",
        trust_remote_code=True,
        dtype=PRECISION,
        device_map="cuda",
        quantization_config=quantization_config,
    ).eval()

    patch_model_for_onnx_export(model)
    yield model, config


@pytest.fixture(scope="module")
def vision_export(real_model):
    """Export vision model once and provide path + program."""
    model, config = real_model
    from onnxscript.rewriter import ort_fusions

    pixel_values = torch.randn(1, 3, 448, 448, dtype=PRECISION, device="cuda")

    def _get_image_features_onnx(pixel_values):
        image_outputs = model.vision_tower(pixel_values, return_dict=True)
        selected = image_outputs.last_hidden_state
        image_sizes = torch.tensor(
            [[pixel_values.shape[-2], pixel_values.shape[-1]]],
            dtype=torch.int64, device=pixel_values.device,
        )
        return model.multi_modal_projector(selected.squeeze(0), image_sizes)

    _original_forward = model.forward
    model.forward = _get_image_features_onnx

    max_image_size = config.vision_config.image_size
    height_dim = torch.export.Dim("height", min=28, max=max_image_size)
    width_dim = torch.export.Dim("width", min=28, max=max_image_size)

    try:
        with torch.no_grad():
            onnx_program = torch.onnx.export(
                model,
                kwargs={"pixel_values": pixel_values},
                input_names=["pixel_values"],
                output_names=["image_features"],
                dynamic_shapes={"pixel_values": {2: height_dim, 3: width_dim}},
                dynamo=True, optimize=True, opset_version=22,
            )
    finally:
        model.forward = _original_forward

    onnx_program.model, _ = ort_fusions.optimize_for_ort(onnx_program.model)

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(tmpdir, exist_ok=True)
        vision_path = os.path.join(tmpdir, "model-vision.onnx")
        onnx_program.model.graph.outputs[0].shape[0] = "num_logical_patches"
        onnx_program.save(vision_path, external_data=True)
        yield vision_path, onnx_program, _get_image_features_onnx


@pytest.fixture(scope="module")
def embedding_export(real_model):
    """Export embedding model once and provide path + program."""
    model, config = real_model

    batch_size = 2
    patches_per_image = 256
    sequence_length = patches_per_image + 10
    hidden_size = config.text_config.hidden_size

    input_ids = torch.randint(
        0, config.text_config.vocab_size, (batch_size, sequence_length),
        device="cuda", dtype=torch.int64,
    )
    for b in range(batch_size):
        input_ids[b, 3 : 3 + patches_per_image] = config.image_token_id
    image_features = torch.randn(
        batch_size * patches_per_image, hidden_size, device="cuda", dtype=PRECISION,
    )

    def _get_fused_input_embeddings(input_ids, image_features):
        inputs_embeds = model.get_input_embeddings()(input_ids)
        image_features_cast = image_features.to(inputs_embeds.dtype)
        special_image_mask = input_ids == model.config.image_token_id
        expanded_image_mask = (
            special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        return inputs_embeds.masked_scatter(expanded_image_mask, image_features_cast)

    _original_forward = model.forward
    model.forward = _get_fused_input_embeddings

    try:
        with torch.no_grad():
            onnx_program = torch.onnx.export(
                model,
                (input_ids, image_features),
                input_names=["input_ids", "image_features"],
                output_names=["inputs_embeds"],
                dynamic_shapes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "image_features": {0: "num_logical_patches"},
                },
                dynamo=True, optimize=True, opset_version=22,
            )
    finally:
        model.forward = _original_forward

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(tmpdir, exist_ok=True)
        embed_path = os.path.join(tmpdir, "model-embedding.onnx")
        onnx_program.save(embed_path, external_data=True)
        yield embed_path, onnx_program, _get_fused_input_embeddings


class TestFP8Dequantization:
    """Verify FP8 weights are properly dequantized."""

    def test_no_fp8_linear_layers(self, real_model):
        """No FP8Linear layers remain after loading with dequantize=True."""
        model, _ = real_model
        for name, module in model.named_modules():
            class_name = type(module).__name__
            assert "FP8" not in class_name, (
                f"Found FP8 layer: {name} ({class_name}). "
                "FP8 dequantization did not convert all layers."
            )


class TestVisionExportRealWeights:
    """Vision export tests with real model weights."""

    def test_export_succeeds(self, vision_export):
        """Vision model exports to ONNX without errors."""
        vision_path, _, _ = vision_export
        assert os.path.isfile(vision_path)
        assert os.path.getsize(vision_path) > 0

    def test_ort_loads(self, vision_export):
        """Exported vision model loads in ORT."""
        vision_path, _, _ = vision_export
        session = ort.InferenceSession(vision_path)
        assert session is not None
        assert [i.name for i in session.get_inputs()] == ["pixel_values"]
        assert [o.name for o in session.get_outputs()] == ["image_features"]

    def test_parity_448x448(self, vision_export, real_model):
        """ONNX output matches PyTorch for 448x448."""
        _, onnx_program, pytorch_fn = vision_export
        model, _ = real_model

        pixel_values = torch.randn(1, 3, 448, 448, dtype=PRECISION, device="cuda")
        with torch.no_grad():
            onnx_out = onnx_program(pixel_values)
            pytorch_out = pytorch_fn(pixel_values)

        torch.testing.assert_close(
            tuple(onnx_out), (pytorch_out,),
            atol=0.01, rtol=0.01, equal_nan=True, check_device=False,
        )

    def test_dynamic_sizes(self, vision_export):
        """Vision model works with multiple image sizes via ORT."""
        vision_path, _, _ = vision_export
        session = ort.InferenceSession(vision_path)

        test_cases = [
            (448, 448, 256),
            (224, 448, 128),
            (196, 392, 98),
        ]
        for h, w, expected_tokens in test_cases:
            pixel_values = np.random.randn(1, 3, h, w).astype(np.float16)
            outputs = session.run(None, {"pixel_values": pixel_values})
            assert outputs[0].shape[0] == expected_tokens, (
                f"For {h}x{w}: expected {expected_tokens}, got {outputs[0].shape[0]}"
            )
            assert np.all(np.isfinite(outputs[0])), f"Non-finite values for {h}x{w}"


class TestEmbeddingExportRealWeights:
    """Embedding export tests with real model weights."""

    def test_export_succeeds(self, embedding_export):
        """Embedding model exports to ONNX without errors."""
        embed_path, _, _ = embedding_export
        assert os.path.isfile(embed_path)
        assert os.path.getsize(embed_path) > 0

    def test_ort_loads(self, embedding_export):
        """Exported embedding model loads in ORT."""
        embed_path, _, _ = embedding_export
        session = ort.InferenceSession(embed_path)
        assert session is not None
        input_names = [i.name for i in session.get_inputs()]
        assert "input_ids" in input_names
        assert "image_features" in input_names

    def test_parity(self, embedding_export, real_model):
        """ONNX output matches PyTorch for embedding fusion."""
        _, onnx_program, pytorch_fn = embedding_export
        model, config = real_model

        batch_size = 1
        patches = 128
        seq_len = patches + 20
        hidden_size = config.text_config.hidden_size

        input_ids = torch.randint(
            0, config.text_config.vocab_size, (batch_size, seq_len), device="cuda"
        )
        input_ids[0, 5 : 5 + patches] = config.image_token_id
        image_features = torch.randn(patches, hidden_size, dtype=PRECISION, device="cuda")

        with torch.no_grad():
            onnx_out = onnx_program(input_ids, image_features)
            pytorch_out = pytorch_fn(input_ids, image_features)

        torch.testing.assert_close(
            tuple(onnx_out), (pytorch_out,),
            atol=0.01, rtol=0.01, equal_nan=True, check_device=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
