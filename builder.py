import argparse
import os
import torch
import shutil

from onnxscript.rewriter import ort_fusions
from transformers import Mistral3Config, AutoModel, AutoConfig

from modeling_code import patch_model_for_onnx_export


def build_vision(args):
    # NOTE: pixel_values shape: [1, 3, H, W] for a single image.
    # Using a 448x448 image (32x32 patches with patch_size=14, yielding 256 logical
    # tokens after spatial merge by 2).
    pixel_values = torch.randn((1, 3, 448, 448), dtype=torch.float32)
    pixel_values = pixel_values.to(args.precision).to(
        args.execution_provider.replace("dml", "cuda")
    )

    dummy_inputs = {"pixel_values": pixel_values}
    # Shape is fixed for the vision model because the Pixtral patch-merger relies on
    # static image dimensions when constructing the block-attention mask.
    dynamic_shapes = {"pixel_values": {}}

    # Export-friendly wrapper that calls the vision pipeline components directly.
    # We bypass get_image_features() to avoid two ONNX export blockers:
    # 1. PixtralVisionModel.forward uses tensor-valued image_sizes for slicing
    #    (GuardOnDataDependentSymNode). Passing image_sizes=None lets Pixtral
    #    derive dimensions from pixel_values.shape (static Python ints).
    # 2. get_image_features .tolist() for torch.split – unnecessary for single image.
    def _get_image_features_onnx(pixel_values):
        image_outputs = model.vision_tower(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        vision_feature_layer = model.config.vision_feature_layer
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        else:
            hs_pool = [
                image_outputs.hidden_states[layer_idx]
                for layer_idx in vision_feature_layer
            ]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        # Construct image_sizes from pixel_values.shape (Python ints, not tensor values)
        # so PatchMerger._forward_export can use them without data-dependent guards.
        image_sizes = torch.tensor(
            [[pixel_values.shape[-2], pixel_values.shape[-1]]],
            dtype=torch.int64,
            device=pixel_values.device,
        )
        image_features = model.multi_modal_projector(
            selected_image_feature.squeeze(0), image_sizes
        )
        return image_features

    # NOTE: hack to vision model export – swap forward with the ONNX-friendly wrapper.
    _original_forward = model.forward
    model.forward = _get_image_features_onnx

    try:
        with torch.no_grad():
            vision_onnx_program = torch.onnx.export(
                model,
                kwargs=dummy_inputs,
                input_names=["pixel_values"],
                output_names=["image_features"],
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
                optimize=True,
                opset_version=22,
            )
    finally:
        # Restore original forward method even if export fails
        model.forward = _original_forward

    # Apply ORT fusions
    vision_onnx_program.model, optimized_count = ort_fusions.optimize_for_ort(
        vision_onnx_program.model
    )
    print("ORT optimized fusion counts:", optimized_count)

    # Save ONNX model
    filename = "model-vision.onnx"
    vision_init_export = os.path.join(args.output, "vision_init_export")
    os.makedirs(vision_init_export, exist_ok=True)
    vision_path = os.path.join(vision_init_export, filename)
    vision_onnx_program.save(vision_path, external_data=True)

    # NOTE: We need to rename output shape name to match the expected name
    vision_onnx_program.model.graph.outputs[0].shape[0] = "num_logical_patches"

    vision_loop_export = os.path.join(args.output, "vision_loop_export")
    os.makedirs(vision_loop_export, exist_ok=True)
    vision_path = os.path.join(vision_loop_export, filename)
    vision_onnx_program.save(vision_path, external_data=True)
    shutil.rmtree(vision_init_export)

    # Verify parity
    onnx_outputs = vision_onnx_program(pixel_values)
    pytorch_outputs = _get_image_features_onnx(pixel_values)

    torch.testing.assert_close(
        tuple(onnx_outputs),
        (pytorch_outputs,),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
        check_device=False,
    )


def build_embedding(args):
    text_hidden_size = config.text_config.hidden_size

    # For a 448x448 image with patch_size=14 and spatial_merge_size=2:
    #   patches per side = 448 / 14 = 32
    #   after spatial merge (2x2): 16 per side → 16 * 16 = 256 logical tokens
    patches_per_image = 256
    batch_size = 1
    sequence_length = patches_per_image + 10  # 10 extra text tokens

    img_start_index = 3
    img_end_index = img_start_index + patches_per_image

    input_ids = torch.randint(
        low=0,
        high=config.text_config.vocab_size,
        size=(batch_size, sequence_length),
        device=args.execution_provider.replace("dml", "cuda"),
        dtype=torch.int64,
    )
    # Mark image-token positions with the model's image_token_id
    input_ids[0][img_start_index:img_end_index] = config.image_token_id

    image_features = torch.randn(
        patches_per_image,
        text_hidden_size,
        device=args.execution_provider.replace("dml", "cuda"),
        dtype=args.precision,
    )

    dummy_inputs = (input_ids, image_features)
    dynamic_shapes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "image_features": {0: "num_logical_patches"},
    }

    # Export-friendly wrapper that fuses token and image embeddings.
    # Equivalent to the masked_scatter step in Mistral3Model.forward.
    def _get_fused_input_embeddings(input_ids, image_features):
        inputs_embeds = model.get_input_embeddings()(input_ids)
        image_features = image_features.to(inputs_embeds.dtype)
        special_image_mask = input_ids == model.config.image_token_id
        expanded_image_mask = (
            special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        )
        inputs_embeds = inputs_embeds.masked_scatter(expanded_image_mask, image_features)
        return inputs_embeds

    # NOTE: hack to embedding model export – swap forward with the ONNX-friendly wrapper.
    _original_forward = model.forward
    model.forward = _get_fused_input_embeddings

    try:
        with torch.no_grad():
            embedding_onnx_program = torch.onnx.export(
                model,
                dummy_inputs,
                input_names=["input_ids", "image_features"],
                output_names=["inputs_embeds"],
                dynamic_shapes=dynamic_shapes,
                dynamo=True,
                optimize=True,
                opset_version=22,
            )
    finally:
        # Restore original forward method even if export fails
        model.forward = _original_forward

    # Save ONNX model
    os.makedirs(args.output, exist_ok=True)
    filename = "model-embedding.onnx"
    fpath = os.path.join(args.output, filename)
    embedding_onnx_program.save(fpath, external_data=True)

    # Verify parity
    onnx_outputs = embedding_onnx_program(input_ids, image_features)
    pytorch_outputs = _get_fused_input_embeddings(input_ids, image_features)

    torch.testing.assert_close(
        tuple(onnx_outputs),
        (pytorch_outputs,),
        atol=0.01,
        rtol=0.01,
        equal_nan=True,
        check_device=False,
    )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to folder on disk containing the Hugging Face config, model, tokenizer, etc.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to folder to store ONNX model and additional files (e.g. GenAI config, external data files, etc.)",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=True,
        choices=["bf16", "fp16", "fp32"],
        help="Precision to export PyTorch components with",
    )

    parser.add_argument(
        "-e",
        "--execution_provider",
        required=True,
        choices=["cpu", "cuda", "dml"],
        help="Execution provider",
    )

    parser.add_argument(
        "-c",
        "--cache_dir",
        required=False,
        default=os.path.join(".", "cache_dir"),
        help="Cache directory for Hugging Face files and temporary ONNX external data files",
    )

    parser.add_argument(
        "--part",
        required=False,
        default="all",
        help="Which component to export: embedding, vision, or all",
    )

    parser.add_argument(
        "--no_weights",
        action="store_true",
        help="If set, initialise a small random model without loading pretrained weights",
    )

    args = parser.parse_args()
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    args.precision = mapping[args.precision]
    return args


if __name__ == "__main__":
    args = get_args()

    if args.no_weights:
        # NOTE: Build a small model config without loading weights.
        #       Adjust the sub-config fields as needed for testing.
        #       Feel free to adjust model config as needed.
        config = Mistral3Config(
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
        model = AutoModel.from_config(
            config,
            attn_implementation="sdpa",
            trust_remote_code=True,
            dtype=args.precision,
        ).to(args.execution_provider.replace("dml", "cuda")).eval()
    else:
        config = AutoConfig.from_pretrained(args.input)
        device_map = "cuda" if args.execution_provider == "cuda" else None
        model = AutoModel.from_pretrained(
            args.input,
            attn_implementation="sdpa",
            trust_remote_code=True,
            dtype=args.precision,
            device_map=device_map,
        ).eval()
        if device_map is None:
            model = model.to(args.execution_provider.replace("dml", "cuda"))

    # Apply ONNX-export-friendly patches (replaces PatchMerger with version
    # that uses pure tensor ops instead of Python for-loops during export).
    patch_model_for_onnx_export(model)

    # Build model components
    if args.part == "embedding":
        build_embedding(args)
    elif args.part == "vision":
        build_vision(args)
    else:
        build_embedding(args)
        build_vision(args)
