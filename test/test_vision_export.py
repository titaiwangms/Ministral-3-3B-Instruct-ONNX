"""End-to-end tests for ONNX vision model export.

Validates that builder.py exports a working vision ONNX model that:
1. Exports successfully with --no_weights (no pretrained weights needed)
2. Loads and runs in ONNX Runtime
3. Produces correct output shapes (256 merged tokens for 448x448 image)
4. Matches PyTorch reference output within tolerance
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
from modeling_code import patch_model_for_onnx_export


@pytest.fixture(scope="module")
def vision_export_dir():
    """Export vision model once for all tests in this module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                sys.executable, "builder.py",
                "--no_weights",
                "--part", "vision",
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
            f"builder.py vision export failed:\nSTDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
        yield tmpdir


@pytest.fixture(scope="module")
def vision_model_path(vision_export_dir):
    """Path to the exported vision ONNX model."""
    path = os.path.join(vision_export_dir, "vision_loop_export", "model-vision.onnx")
    assert os.path.exists(path), f"Vision ONNX model not found at {path}"
    return path


class TestVisionExportSuccess:
    """Tests that the vision export pipeline succeeds."""

    def test_export_produces_onnx_file(self, vision_model_path):
        """builder.py creates the expected ONNX model file."""
        assert os.path.isfile(vision_model_path)
        assert os.path.getsize(vision_model_path) > 0

    def test_export_produces_external_data(self, vision_export_dir):
        """External data file is created alongside the ONNX model."""
        data_path = os.path.join(
            vision_export_dir, "vision_loop_export", "model-vision.onnx.data"
        )
        assert os.path.isfile(data_path)


class TestVisionOrtInference:
    """Tests that the exported vision model runs correctly in ONNX Runtime."""

    def test_model_loads_in_ort(self, vision_model_path):
        """ONNX model loads without errors in ONNX Runtime."""
        session = ort.InferenceSession(vision_model_path)
        assert session is not None

    def test_input_names(self, vision_model_path):
        """Model has the expected input: pixel_values only."""
        session = ort.InferenceSession(vision_model_path)
        input_names = [inp.name for inp in session.get_inputs()]
        assert input_names == ["pixel_values"]

    def test_output_names(self, vision_model_path):
        """Model has the expected output: image_features."""
        session = ort.InferenceSession(vision_model_path)
        output_names = [out.name for out in session.get_outputs()]
        assert output_names == ["image_features"]

    def test_output_shape(self, vision_model_path):
        """Output has correct shape: [256, 64] for 448x448 image with --no_weights config."""
        session = ort.InferenceSession(vision_model_path)
        pixel_values = np.random.randn(1, 3, 448, 448).astype(np.float32)
        outputs = session.run(None, {"pixel_values": pixel_values})

        # 448/14 = 32 patches per side, spatial_merge_size=2 → 16x16 = 256 merged tokens
        # hidden_size = 64 (from --no_weights config)
        assert outputs[0].shape == (256, 64)

    def test_output_is_finite(self, vision_model_path):
        """Output contains no NaN or Inf values."""
        session = ort.InferenceSession(vision_model_path)
        pixel_values = np.random.randn(1, 3, 448, 448).astype(np.float32)
        outputs = session.run(None, {"pixel_values": pixel_values})

        assert np.all(np.isfinite(outputs[0]))

    def test_dynamic_height_width(self, vision_model_path):
        """Vision model accepts dynamic H/W (multiples of 28)."""
        session = ort.InferenceSession(vision_model_path)

        test_cases = [
            (448, 448, 256),   # 16*16 merged tokens
            (224, 448, 128),   # 8*16 merged tokens
            (28, 28, 1),       # minimum: 1*1 merged token
            (196, 392, 98),    # 7*14 merged tokens
        ]
        for h, w, expected_tokens in test_cases:
            pixel_values = np.random.randn(1, 3, h, w).astype(np.float32)
            outputs = session.run(None, {"pixel_values": pixel_values})
            assert outputs[0].shape[0] == expected_tokens, (
                f"For {h}x{w}: expected {expected_tokens} tokens, got {outputs[0].shape[0]}"
            )


class TestVisionParity:
    """Tests that ONNX output matches PyTorch reference."""

    def test_builder_parity_check_passes(self, vision_export_dir):
        """builder.py's internal parity check (torch.testing.assert_close) passed.

        Since builder.py exits with code 0, this means its internal
        torch.testing.assert_close(onnx_output, pytorch_output, atol=0.01, rtol=0.01)
        succeeded. This is the strongest parity guarantee since both outputs
        come from the same model instance.
        """
        # The fixture already asserts exit code 0, which implies parity passed.
        # This test documents that guarantee explicitly.
        model_path = os.path.join(
            vision_export_dir, "vision_loop_export", "model-vision.onnx"
        )
        assert os.path.exists(model_path)

    def test_in_process_parity(self):
        """In-process export and verify: same model weights for both paths."""
        config = create_test_config()
        model = AutoModel.from_config(
            config, attn_implementation="sdpa", trust_remote_code=True
        ).eval()
        patch_model_for_onnx_export(model)

        pixel_values = torch.randn(1, 3, 448, 448, dtype=torch.float32)

        def _get_image_features_onnx(pixel_values):
            image_outputs = model.vision_tower(
                pixel_values, return_dict=True,
            )
            selected = image_outputs.last_hidden_state
            image_sizes = torch.tensor(
                [[pixel_values.shape[-2], pixel_values.shape[-1]]],
                dtype=torch.int64,
                device=pixel_values.device,
            )
            return model.multi_modal_projector(selected.squeeze(0), image_sizes)

        _original_forward = model.forward
        model.forward = _get_image_features_onnx

        max_image_size = config.vision_config.image_size
        height_dim = torch.export.Dim("height", min=28, max=max_image_size)
        width_dim = torch.export.Dim("width", min=28, max=max_image_size)

        with torch.no_grad():
            onnx_program = torch.onnx.export(
                model,
                kwargs={"pixel_values": pixel_values},
                input_names=["pixel_values"],
                output_names=["image_features"],
                dynamic_shapes={"pixel_values": {2: height_dim, 3: width_dim}},
                dynamo=True,
                optimize=True,
                opset_version=22,
            )
        model.forward = _original_forward

        with torch.no_grad():
            onnx_output = onnx_program(pixel_values)
            pytorch_output = _get_image_features_onnx(pixel_values)

        torch.testing.assert_close(
            tuple(onnx_output),
            (pytorch_output,),
            atol=0.01,
            rtol=0.01,
            equal_nan=True,
            check_device=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
