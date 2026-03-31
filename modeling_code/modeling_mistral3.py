"""ONNX-export-friendly modifications to HuggingFace Mistral 3 modeling code.

Original source:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral3/modeling_mistral3.py
License: Apache 2.0 (Copyright 2025 HuggingFace Inc.)

Changes from original (see modeling_mistral3_original.py for diff reference):

1. Mistral3PatchMerger.forward():
   - Added torch.compiler.is_exporting() dispatch
   - Export path (_forward_export): pure tensor operations, no Python for-loops
   - Eager path (_forward_eager): preserves original behavior with for-loops
   - Both paths produce numerically identical results

2. generate_block_attention_mask (modeling_pixtral.py):
   - NOT modified. With static shapes (single 448x448 image as used by builder.py),
     dynamo unrolls the single-iteration loop at compile time. The function uses
     Python lists and loops, but these evaluate to constants with fixed input shapes.

3. get_image_features .tolist():
   - NOT modified. With static image_sizes, dynamo evaluates .tolist() at compile
     time. builder.py's _get_image_features_onnx wrapper also handles this by
     concatenating the split results directly.

Usage:
    from modeling_code import patch_model_for_onnx_export

    model = AutoModel.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512", ...)
    patch_model_for_onnx_export(model)
    # Model is now ready for torch.onnx.export(dynamo=True)
"""

import torch
from torch import nn


class Mistral3PatchMerger(nn.Module):
    """ONNX-export-friendly Mistral3PatchMerger.

    Learned merging of spatial_merge_size ** 2 patches. This is a drop-in
    replacement for the HuggingFace Mistral3PatchMerger that uses pure tensor
    operations during ONNX export instead of Python for-loops.

    Changes from original:
    - forward() dispatches to _forward_export() when torch.compiler.is_exporting()
    - _forward_export() replaces Python for-loop with direct tensor reshaping
    - _forward_eager() preserves original behavior for regular inference
    - Both paths produce numerically identical results (see tests)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.vision_config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = self.config.vision_config.patch_size
        self.merging_layer = nn.Linear(
            hidden_size * self.spatial_merge_size**2, hidden_size, bias=False
        )

    def forward(
        self, image_features: torch.Tensor, image_sizes: torch.Tensor
    ) -> torch.Tensor:
        if torch.compiler.is_exporting():
            return self._forward_export(image_features, image_sizes)
        return self._forward_eager(image_features, image_sizes)

    def _forward_export(
        self, image_features: torch.Tensor, image_sizes: torch.Tensor
    ) -> torch.Tensor:
        """ONNX-export-friendly path using pure tensor operations.

        Replaces the Python for-loop and list comprehensions from the original
        with direct tensor indexing and reshaping. Handles single-image input
        as used by builder.py (one image at a time with fixed dimensions).

        The algorithm is identical to the original:
        1. Compute patch grid dimensions from pixel sizes
        2. Reshape flat patch tokens into a spatial grid [h, w, d]
        3. Use unfold to extract spatial_merge_size x spatial_merge_size windows
        4. Apply the learned merging linear layer
        """
        # Compute patch grid dimensions: image_sizes is [num_images, 2] (h, w in pixels)
        patch_h = image_sizes[0, 0] // self.patch_size
        patch_w = image_sizes[0, 1] // self.patch_size
        d = image_features.shape[-1]

        # Reshape flat tokens into spatial grid: [num_tokens, d] -> [1, d, h, w]
        image_grid = (
            image_features.view(patch_h, patch_w, d).permute(2, 0, 1).unsqueeze(0)
        )

        # Guards required by torch.export: unfold/im2col needs to verify that
        # spatial dimensions are non-zero and large enough for the kernel.
        torch._check(image_grid.shape[2] != 0)
        torch._check(image_grid.shape[3] != 0)
        torch._check(image_grid.shape[2] // self.spatial_merge_size > 0)
        torch._check(image_grid.shape[3] // self.spatial_merge_size > 0)

        # Extract spatial_merge_size x spatial_merge_size windows
        # unfold output: [1, d * merge_size^2, num_merged_patches]
        grid = torch.nn.functional.unfold(
            image_grid,
            kernel_size=self.spatial_merge_size,
            stride=self.spatial_merge_size,
        )

        # Transpose to [num_merged_patches, d * merge_size^2]
        image_features = grid.view(d * self.spatial_merge_size**2, -1).t()

        return self.merging_layer(image_features)

    def _forward_eager(
        self, image_features: torch.Tensor, image_sizes: torch.Tensor
    ) -> torch.Tensor:
        """Original eager-mode path with Python for-loops (not ONNX-exportable).

        Preserves the original HuggingFace behavior for regular inference. Supports
        multiple images of different sizes via the for-loop over split features.
        """
        image_sizes_list = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size)
            for image_size in image_sizes
        ]
        tokens_per_image = [h * w for h, w in image_sizes_list]
        d = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(
            image_features.split(tokens_per_image)
        ):
            h, w = image_sizes_list[image_index]
            image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
            grid = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            )
            grid = grid.view(d * self.spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        return self.merging_layer(image_features)


def patch_model_for_onnx_export(model):
    """Apply ONNX-export-friendly patches to a Mistral 3 model.

    Replaces the PatchMerger's class to use the ONNX-friendly forward method
    that dispatches to pure tensor operations when torch.compiler.is_exporting()
    returns True during ONNX export.

    The patch is transparent: outside of export context, the model behaves
    identically to the unpatched version (uses the original eager path).

    Args:
        model: A Mistral3ForConditionalGeneration or Mistral3Model instance
               (from HuggingFace transformers).

    Returns:
        The patched model (same object, modified in-place).

    Raises:
        ValueError: If the model structure doesn't match expected Mistral 3 layout.

    Example:
        from transformers import AutoModel
        from modeling_code import patch_model_for_onnx_export

        model = AutoModel.from_pretrained("mistralai/Ministral-3-3B-Instruct-2512")
        patch_model_for_onnx_export(model)
        # Now safe to use with torch.onnx.export(dynamo=True)
    """
    # Navigate to the Mistral3Model (handles both ForConditionalGeneration and Model)
    if hasattr(model, "model") and hasattr(model.model, "multi_modal_projector"):
        patch_merger = model.model.multi_modal_projector.patch_merger
    elif hasattr(model, "multi_modal_projector"):
        patch_merger = model.multi_modal_projector.patch_merger
    else:
        raise ValueError(
            "Cannot find multi_modal_projector.patch_merger on the model. "
            "Expected a Mistral3ForConditionalGeneration or Mistral3Model instance."
        )

    # Replace the class to get the ONNX-friendly forward method.
    # Instance attributes (weights, config, etc.) are preserved.
    patch_merger.__class__ = Mistral3PatchMerger

    return model
