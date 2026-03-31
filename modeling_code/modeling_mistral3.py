"""ONNX-export-friendly modifications to HuggingFace Mistral 3 modeling code.

Original source:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral3/modeling_mistral3.py
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/modeling_pixtral.py
License: Apache 2.0 (Copyright 2025 HuggingFace Inc.)

Changes from original (see modeling_mistral3.diff for a detailed comparison):

1. Mistral3PatchMerger.forward():
   - Added torch.compiler.is_exporting() dispatch
   - Export path (_forward_export): pure tensor operations replacing Python
     for-loops and list comprehensions that block torch.onnx.export(dynamo=True)
   - Eager path (_forward_eager): preserves original behavior for regular inference
   - Added torch._check guards for unfold/im2col symbolic shape validation
   - Both paths produce numerically identical results

2. PixtralVisionModel patching via pixtral_vision_forward_export():
   - Replaces the original forward that uses generate_block_attention_mask
     (Python for-loops + torch.tensor from symbolic values) and
     position_ids_in_meshgrid (Python for-loop over batch)
   - With batch=1, block attention mask is trivially all-zeros (single image =
     full attention), so we skip it entirely and pass attention_mask=None
   - Position IDs computed inline with torch.arange/meshgrid (no for-loop)
   - Supports dynamic H/W (multiples of 28, range [28, config.image_size])
   - Enforced with torch._check(pixel_values.shape[0] == 1) guard

3. get_image_features .tolist():
   - NOT modified. builder.py's _get_image_features_onnx wrapper handles this
     by calling vision components directly.

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


def pixtral_vision_forward_export(self, pixel_values, **kwargs):
    """ONNX-export-friendly forward for PixtralVisionModel (batch=1).

    Replaces the original forward which uses:
    - generate_block_attention_mask: Python for-loops + torch.tensor from symbolic
      values (incompatible with torch.export under dynamic shapes)
    - position_ids_in_meshgrid: Python for-loop over batch

    With batch=1 (single image), the block attention mask is trivially all-zeros
    (one image = full attention), so we skip it entirely and pass attention_mask=None
    to SDPA. Position IDs are computed inline with torch.arange/meshgrid.

    Supports dynamic H/W where H and W are multiples of patch_size.
    """
    # Structurally enforce batch=1 — the mask-skip is only valid for single image
    torch._check(pixel_values.shape[0] == 1)

    # Conv2d: [1, 3, H, W] -> [1, hidden, H/patch_size, W/patch_size]
    target_dtype = self.patch_conv.weight.dtype
    patch_embeds = self.patch_conv(pixel_values.to(dtype=target_dtype))

    # For batch=1, extract the single image's patch embeddings directly
    # patch_embeds shape: [1, hidden, grid_h, grid_w]
    grid_h = patch_embeds.shape[2]
    grid_w = patch_embeds.shape[3]

    # Flatten spatial dims and transpose: [1, hidden, grid_h, grid_w] -> [1, grid_h*grid_w, hidden]
    patch_embeds = patch_embeds[0].flatten(1).T.unsqueeze(0)
    patch_embeds = self.ln_pre(patch_embeds)

    # Compute position IDs inline (replaces position_ids_in_meshgrid for batch=1)
    max_width = self.config.image_size // self.config.patch_size
    h_indices = torch.arange(grid_h, device=pixel_values.device)
    w_indices = torch.arange(grid_w, device=pixel_values.device)
    mesh_h, mesh_w = torch.meshgrid(h_indices, w_indices, indexing="ij")
    position_ids = (mesh_h * max_width + mesh_w).reshape(-1)
    kwargs["position_ids"] = position_ids.unsqueeze(0)

    position_embeddings = self.patch_positional_embedding(patch_embeds, position_ids)

    # Skip generate_block_attention_mask: with batch=1 (single image), the mask
    # would be all zeros (full attention). Passing None is semantically equivalent.
    return self.transformer(
        patch_embeds,
        attention_mask=None,
        position_embeddings=position_embeddings,
        **kwargs,
    )


def _pixtral_vision_forward_dispatch(self, pixel_values, **kwargs):
    """Dispatch between export and eager forward paths for PixtralVisionModel."""
    if torch.compiler.is_exporting():
        return pixtral_vision_forward_export(self, pixel_values, **kwargs)
    return self._original_forward(pixel_values, **kwargs)


def patch_model_for_onnx_export(model):
    """Apply ONNX-export-friendly patches to a Mistral 3 model.

    Patches two components:
    1. PatchMerger: Replaces forward with version that uses pure tensor ops
       during export (no Python for-loops).
    2. PixtralVisionModel: Replaces forward with version that skips
       generate_block_attention_mask (not needed for batch=1) and computes
       position_ids inline (no Python for-loop).

    Both patches use torch.compiler.is_exporting() dispatch, so the model
    behaves identically to the unpatched version outside export context.

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
        vision_tower = model.model.vision_tower
    elif hasattr(model, "multi_modal_projector"):
        patch_merger = model.multi_modal_projector.patch_merger
        vision_tower = model.vision_tower
    else:
        raise ValueError(
            "Cannot find multi_modal_projector.patch_merger on the model. "
            "Expected a Mistral3ForConditionalGeneration or Mistral3Model instance."
        )

    # Patch 1: Replace PatchMerger class for ONNX-friendly forward
    patch_merger.__class__ = Mistral3PatchMerger

    # Patch 2: Replace PixtralVisionModel forward with export-aware dispatch.
    # Store original forward as bound method so eager path still works.
    import types

    vision_tower._original_forward = vision_tower.forward
    vision_tower.forward = types.MethodType(
        _pixtral_vision_forward_dispatch, vision_tower
    )

    return model
