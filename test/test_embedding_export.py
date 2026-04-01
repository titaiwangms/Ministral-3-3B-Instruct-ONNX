"""End-to-end tests for ONNX embedding model export.

Validates that builder.py exports a working embedding ONNX model that:
1. Exports successfully with --no_weights (no pretrained weights needed)
2. Loads and runs in ONNX Runtime
3. Produces correct output shapes (batch_size, sequence_length, hidden_size)
4. Correctly fuses image features into token embeddings via masked_scatter
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import onnxruntime as ort
import pytest
import torch
from transformers import AutoModel

from test.conftest import create_test_config, REPO_ROOT


@pytest.fixture(scope="module")
def embedding_export_dir():
    """Export embedding model once for all tests in this module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable, "builder.py",
                "--no_weights",
                "--part", "embedding",
                "-i", ".",
                "-o", tmpdir,
                "-p", "fp32",
                "-e", "cpu",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, (
            f"builder.py embedding export failed:\nSTDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
        yield tmpdir


@pytest.fixture(scope="module")
def embedding_model_path(embedding_export_dir):
    """Path to the exported embedding ONNX model."""
    path = os.path.join(embedding_export_dir, "model-embedding.onnx")
    assert os.path.exists(path), f"Embedding ONNX model not found at {path}"
    return path


class TestEmbeddingExportSuccess:
    """Tests that the embedding export pipeline succeeds."""

    def test_export_produces_onnx_file(self, embedding_model_path):
        """builder.py creates the expected ONNX model file."""
        assert os.path.isfile(embedding_model_path)
        assert os.path.getsize(embedding_model_path) > 0


class TestEmbeddingOrtInference:
    """Tests that the exported embedding model runs correctly in ONNX Runtime."""

    def test_model_loads_in_ort(self, embedding_model_path):
        """ONNX model loads without errors in ONNX Runtime."""
        session = ort.InferenceSession(embedding_model_path)
        assert session is not None

    def test_input_names(self, embedding_model_path):
        """Model has the expected inputs: input_ids and image_features."""
        session = ort.InferenceSession(embedding_model_path)
        input_names = [inp.name for inp in session.get_inputs()]
        assert "input_ids" in input_names
        assert "image_features" in input_names

    def test_output_names(self, embedding_model_path):
        """Model has the expected output: inputs_embeds."""
        session = ort.InferenceSession(embedding_model_path)
        output_names = [out.name for out in session.get_outputs()]
        assert output_names == ["inputs_embeds"]

    def test_output_shape(self, embedding_model_path):
        """Output has correct shape: [batch, seq_len, hidden_size]."""
        session = ort.InferenceSession(embedding_model_path)

        batch_size = 1
        sequence_length = 266  # 256 image tokens + 10 text tokens
        hidden_size = 64

        input_ids = np.random.randint(0, 32000, (batch_size, sequence_length)).astype(
            np.int64
        )
        image_features = np.random.randn(256, hidden_size).astype(np.float32)

        outputs = session.run(
            None,
            {"input_ids": input_ids, "image_features": image_features},
        )

        assert outputs[0].shape == (batch_size, sequence_length, hidden_size)

    def test_dynamic_sequence_length(self, embedding_model_path):
        """Embedding model supports dynamic sequence lengths with masked_scatter."""
        session = ort.InferenceSession(embedding_model_path)
        config = create_test_config()

        # Shorter sequence with image tokens placed to exercise masked_scatter
        num_image_tokens = 50
        input_ids = np.random.randint(0, 32000, (1, 100)).astype(np.int64)
        input_ids[0, 5 : 5 + num_image_tokens] = config.image_token_id
        image_features = np.random.randn(num_image_tokens, 64).astype(np.float32)
        outputs = session.run(
            None,
            {"input_ids": input_ids, "image_features": image_features},
        )
        assert outputs[0].shape == (1, 100, 64)

        # Longer sequence with image tokens
        num_image_tokens = 300
        input_ids = np.random.randint(0, 32000, (1, 500)).astype(np.int64)
        input_ids[0, 10 : 10 + num_image_tokens] = config.image_token_id
        image_features = np.random.randn(num_image_tokens, 64).astype(np.float32)
        outputs = session.run(
            None,
            {"input_ids": input_ids, "image_features": image_features},
        )
        assert outputs[0].shape == (1, 500, 64)

    def test_output_is_finite(self, embedding_model_path):
        """Output contains no NaN or Inf values."""
        session = ort.InferenceSession(embedding_model_path)

        input_ids = np.random.randint(0, 32000, (1, 266)).astype(np.int64)
        image_features = np.random.randn(256, 64).astype(np.float32)

        outputs = session.run(
            None,
            {"input_ids": input_ids, "image_features": image_features},
        )
        assert np.all(np.isfinite(outputs[0]))


class TestEmbeddingParity:
    """Tests that ONNX output matches PyTorch reference for masked_scatter fusion."""

    def test_builder_parity_check_passes(self, embedding_export_dir):
        """builder.py's internal parity check passed (exit code 0)."""
        model_path = os.path.join(embedding_export_dir, "model-embedding.onnx")
        assert os.path.exists(model_path)

    def test_in_process_parity(self):
        """In-process export and verify with multiple batch/sequence sizes."""
        config = create_test_config()
        model = AutoModel.from_config(
            config, attn_implementation="sdpa", trust_remote_code=True
        ).eval()

        hidden_size = config.text_config.hidden_size

        # Use batch_size=2 for export (matches builder.py)
        batch_size = 2
        patches_per_image = 256
        sequence_length = patches_per_image + 10

        input_ids = torch.randint(
            0, config.text_config.vocab_size, (batch_size, sequence_length)
        )
        for b in range(batch_size):
            input_ids[b, 3 : 3 + patches_per_image] = config.image_token_id
        image_features = torch.randn(batch_size * patches_per_image, hidden_size)

        def _get_fused_input_embeddings(input_ids, image_features):
            inputs_embeds = model.get_input_embeddings()(input_ids)
            image_features_cast = image_features.to(inputs_embeds.dtype)
            special_image_mask = input_ids == model.config.image_token_id
            expanded_image_mask = (
                special_image_mask.unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            return inputs_embeds.masked_scatter(expanded_image_mask, image_features_cast)

        _original_forward = model.forward
        model.forward = _get_fused_input_embeddings

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
                dynamo=True,
                optimize=True,
                opset_version=22,
            )
        model.forward = _original_forward

        # Test parity at multiple batch sizes and sequence lengths
        test_cases = [
            # (batch_size, num_image_tokens, num_text_tokens)
            (1, 256, 10),
            (2, 128, 20),
            (1, 64, 50),
        ]
        for bs, n_img, n_txt in test_cases:
            seq_len = n_img + n_txt
            ids = torch.randint(0, config.text_config.vocab_size, (bs, seq_len))
            for b in range(bs):
                ids[b, 3 : 3 + n_img] = config.image_token_id
            feats = torch.randn(bs * n_img, hidden_size)

            with torch.no_grad():
                onnx_output = onnx_program(ids, feats)
                pytorch_output = _get_fused_input_embeddings(ids, feats)

            torch.testing.assert_close(
                tuple(onnx_output),
                (pytorch_output,),
                atol=0.01,
                rtol=0.01,
                equal_nan=True,
                check_device=False,
                msg=f"Parity failed for batch={bs}, img_tokens={n_img}, txt_tokens={n_txt}",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
