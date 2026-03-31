"""Tests for ONNX-export-friendly Mistral 3 modeling code.

Validates that the modified Mistral3PatchMerger produces numerically identical
results to the original HuggingFace implementation, and that the
patch_model_for_onnx_export utility works correctly.
"""

import sys
import os

import torch
import pytest

# Ensure repo root is on the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import Mistral3Config, AutoModel

from modeling_code.modeling_mistral3 import Mistral3PatchMerger, patch_model_for_onnx_export
from modeling_code.modeling_mistral3_original import (
    Mistral3PatchMerger as OriginalMistral3PatchMerger,
)


def _create_test_config():
    """Create a minimal Mistral3Config for testing (matches builder.py --no_weights)."""
    return Mistral3Config(
        vision_config={
            "model_type": "pixtral",
            "num_hidden_layers": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "head_dim": 16,
            "patch_size": 14,
            "image_size": 448,
        },
        text_config={
            "model_type": "mistral",
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "vocab_size": 32000,
        },
    )


class TestMistral3PatchMerger:
    """Tests for the ONNX-friendly Mistral3PatchMerger."""

    def test_export_matches_eager_single_image(self):
        """Export path produces identical output to eager path for single 448x448 image."""
        config = _create_test_config()
        merger = Mistral3PatchMerger(config)

        # 448x448 image with patch_size=14 -> 32x32 = 1024 patches
        num_patches = (448 // 14) * (448 // 14)
        hidden_size = config.vision_config.hidden_size
        image_features = torch.randn(num_patches, hidden_size)
        image_sizes = torch.tensor([[448, 448]], dtype=torch.int64)

        eager_output = merger._forward_eager(image_features, image_sizes)
        export_output = merger._forward_export(image_features, image_sizes)

        torch.testing.assert_close(eager_output, export_output)

    def test_export_matches_eager_rectangular_image(self):
        """Export path matches eager for a non-square image (448x224)."""
        config = _create_test_config()
        merger = Mistral3PatchMerger(config)

        # 448x224: patches = (32, 16) = 512 tokens
        patch_h, patch_w = 448 // 14, 224 // 14
        num_patches = patch_h * patch_w
        hidden_size = config.vision_config.hidden_size
        image_features = torch.randn(num_patches, hidden_size)
        image_sizes = torch.tensor([[448, 224]], dtype=torch.int64)

        eager_output = merger._forward_eager(image_features, image_sizes)
        export_output = merger._forward_export(image_features, image_sizes)

        torch.testing.assert_close(eager_output, export_output)

    def test_output_shape_single_image(self):
        """Output shape is correct after spatial merging."""
        config = _create_test_config()
        merger = Mistral3PatchMerger(config)

        num_patches = 1024  # 32 * 32
        hidden_size = config.vision_config.hidden_size
        image_features = torch.randn(num_patches, hidden_size)
        image_sizes = torch.tensor([[448, 448]], dtype=torch.int64)

        output = merger._forward_export(image_features, image_sizes)

        # After spatial merge with size 2: 16*16 = 256 merged tokens
        spatial_merge_size = config.spatial_merge_size
        expected_tokens = (32 // spatial_merge_size) * (32 // spatial_merge_size)
        assert output.shape == (expected_tokens, hidden_size)

    def test_output_shape_rectangular(self):
        """Output shape is correct for rectangular image."""
        config = _create_test_config()
        merger = Mistral3PatchMerger(config)

        patch_h, patch_w = 32, 16  # 448x224
        num_patches = patch_h * patch_w
        hidden_size = config.vision_config.hidden_size
        image_features = torch.randn(num_patches, hidden_size)
        image_sizes = torch.tensor([[448, 224]], dtype=torch.int64)

        output = merger._forward_export(image_features, image_sizes)

        spatial_merge_size = config.spatial_merge_size
        expected_tokens = (patch_h // spatial_merge_size) * (
            patch_w // spatial_merge_size
        )
        assert output.shape == (expected_tokens, hidden_size)

    def test_forward_uses_eager_outside_export(self):
        """forward() uses eager path when not in export context."""
        config = _create_test_config()
        merger = Mistral3PatchMerger(config)

        num_patches = 1024
        hidden_size = config.vision_config.hidden_size
        image_features = torch.randn(num_patches, hidden_size)
        image_sizes = torch.tensor([[448, 448]], dtype=torch.int64)

        # Outside export context, forward() should use eager path
        # and produce the same result as _forward_eager
        forward_output = merger(image_features, image_sizes)
        eager_output = merger._forward_eager(image_features, image_sizes)

        torch.testing.assert_close(forward_output, eager_output)

    def test_matches_original_implementation(self):
        """Modified PatchMerger matches the original HF implementation numerically."""
        config = _create_test_config()

        original = OriginalMistral3PatchMerger(config)
        modified = Mistral3PatchMerger(config)

        # Copy weights from original to modified
        modified.load_state_dict(original.state_dict())

        num_patches = 1024
        hidden_size = config.vision_config.hidden_size
        image_features = torch.randn(num_patches, hidden_size)
        image_sizes = torch.tensor([[448, 448]], dtype=torch.int64)

        original_output = original(image_features, image_sizes)
        modified_eager = modified._forward_eager(image_features, image_sizes)
        modified_export = modified._forward_export(image_features, image_sizes)

        torch.testing.assert_close(original_output, modified_eager)
        torch.testing.assert_close(original_output, modified_export)

    def test_gradient_flow(self):
        """Gradients flow through the export path."""
        config = _create_test_config()
        merger = Mistral3PatchMerger(config)

        num_patches = 1024
        hidden_size = config.vision_config.hidden_size
        image_features = torch.randn(num_patches, hidden_size, requires_grad=True)
        image_sizes = torch.tensor([[448, 448]], dtype=torch.int64)

        output = merger._forward_export(image_features, image_sizes)
        loss = output.sum()
        loss.backward()

        assert image_features.grad is not None
        assert image_features.grad.shape == image_features.shape


class TestPatchModelForOnnxExport:
    """Tests for the patch_model_for_onnx_export utility."""

    def test_patches_conditional_generation_model(self):
        """Patches a Mistral3ForConditionalGeneration model."""
        config = _create_test_config()
        model = AutoModel.from_config(
            config,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        # Before patching: original PatchMerger class
        original_class = model.multi_modal_projector.patch_merger.__class__.__name__
        assert original_class == "Mistral3PatchMerger"

        patch_model_for_onnx_export(model)

        # After patching: our ONNX-friendly PatchMerger class
        assert (
            model.multi_modal_projector.patch_merger.__class__
            is Mistral3PatchMerger
        )

    def test_patched_model_preserves_weights(self):
        """Patching preserves the model's learned weights."""
        config = _create_test_config()
        model = AutoModel.from_config(
            config,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        # Capture weights before patching
        weight_before = (
            model.multi_modal_projector.patch_merger.merging_layer.weight.clone()
        )

        patch_model_for_onnx_export(model)

        # Weights should be identical after patching
        weight_after = model.multi_modal_projector.patch_merger.merging_layer.weight
        torch.testing.assert_close(weight_before, weight_after)

    def test_patched_model_produces_same_output(self):
        """Patched model produces identical output to unpatched model."""
        config = _create_test_config()
        model = AutoModel.from_config(
            config,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        num_patches = 1024
        hidden_size = config.vision_config.hidden_size
        image_features = torch.randn(num_patches, hidden_size)
        image_sizes = torch.tensor([[448, 448]], dtype=torch.int64)

        # Get output before patching
        with torch.no_grad():
            output_before = model.multi_modal_projector.patch_merger(
                image_features, image_sizes
            )

        patch_model_for_onnx_export(model)

        # Get output after patching (uses eager path outside export context)
        with torch.no_grad():
            output_after = model.multi_modal_projector.patch_merger(
                image_features, image_sizes
            )

        torch.testing.assert_close(output_before, output_after)

    def test_returns_same_model_instance(self):
        """patch_model_for_onnx_export returns the same model object."""
        config = _create_test_config()
        model = AutoModel.from_config(
            config,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        result = patch_model_for_onnx_export(model)
        assert result is model

    def test_raises_on_invalid_model(self):
        """Raises ValueError for models without multi_modal_projector."""
        with pytest.raises(ValueError, match="Cannot find multi_modal_projector"):
            patch_model_for_onnx_export(torch.nn.Linear(10, 10))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
