"""Original HuggingFace Mistral 3 modeling code (partial snapshot for diff reference).

Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral3/modeling_mistral3.py
License: Apache 2.0 (Copyright 2025 HuggingFace Inc.)

This file contains the ORIGINAL implementations of the classes/functions that are
modified in modeling_mistral3.py for ONNX export compatibility. It exists solely
as a diff reference — do not import from this file.

To see what changed for ONNX export:
    diff modeling_code/modeling_mistral3_original.py modeling_code/modeling_mistral3.py

For the complete original source, see the URL above.

ONNX Export Blockers in Original Code
--------------------------------------

1. Mistral3PatchMerger.forward():
   - Python list comprehension to compute patch sizes from image_sizes
   - Python list comprehension to compute tokens_per_image
   - Python for-loop over image_features.split(tokens_per_image)
   These prevent torch.onnx.export(dynamo=True) from tracing a single graph.

2. Mistral3Model.get_image_features():
   - .tolist() call to convert split_sizes tensor to Python list
   - builder.py's _get_image_features_onnx wrapper handles this, and with
     static image_sizes dynamo can evaluate .tolist() at compile time.
   - No changes needed in the modeling code for this.

3. generate_block_attention_mask() (in modeling_pixtral.py):
   - Uses Python for-loop and torch.tensor() from Python lists
   - With static shapes (single 448x448 image), dynamo unrolls the single-
     iteration loop and evaluates constants at compile time.
   - No changes needed for static-shape export.
"""

import torch
from torch import nn


class Mistral3PatchMerger(nn.Module):
    """Original HuggingFace Mistral3PatchMerger (NOT ONNX-exportable).

    This is the unmodified implementation that uses Python for-loops and list
    comprehensions. See modeling_mistral3.py for the ONNX-friendly version.

    Learned merging of spatial_merge_size ** 2 patches.
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
        # ONNX BLOCKER: list comprehension with Python iteration over tensor
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size)
            for image_size in image_sizes
        ]

        # ONNX BLOCKER: list comprehension with tuple unpacking
        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]

        # ONNX BLOCKER: Python for-loop with data-dependent split
        permuted_tensor = []
        for image_index, image_tokens in enumerate(
            image_features.split(tokens_per_image)
        ):
            h, w = image_sizes[image_index]
            image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
            grid = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            )
            grid = grid.view(d * self.spatial_merge_size**2, -1).t()
            permuted_tensor.append(grid)

        image_features = torch.cat(permuted_tensor, dim=0)
        image_features = self.merging_layer(image_features)
        return image_features


# ---- Reference: get_image_features .tolist() usage ----
#
# In Mistral3Model.get_image_features(), the original code computes:
#
#   downsample_ratio = self.vision_tower.patch_size * self.config.spatial_merge_size
#   split_sizes = (
#       (torch.as_tensor(image_sizes, device=image_features.device) // downsample_ratio)
#       .prod(dim=-1)
#       .tolist()
#   )
#   image_features = torch.split(image_features.squeeze(0), split_sizes)
#
# The .tolist() converts a tensor to a Python list for use as torch.split sizes.
# With static image_sizes (as used in builder.py), dynamo evaluates this at compile
# time since the values are known constants. No modification needed.
