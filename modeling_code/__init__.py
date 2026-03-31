"""ONNX-export-friendly model rewrites for Mistral 3.

This package contains modified versions of HuggingFace Mistral 3 modeling code
that are compatible with torch.onnx.export(dynamo=True).

Usage:
    from modeling_code import patch_model_for_onnx_export

    model = AutoModel.from_pretrained(...)
    patch_model_for_onnx_export(model)
    # Now model can be exported with torch.onnx.export(dynamo=True)
"""

from modeling_code.modeling_mistral3 import (
    Mistral3PatchMerger,
    patch_model_for_onnx_export,
)

__all__ = [
    "Mistral3PatchMerger",
    "patch_model_for_onnx_export",
]
