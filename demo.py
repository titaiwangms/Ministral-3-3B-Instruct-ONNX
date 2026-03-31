"""HuggingFace PyTorch inference demo for Ministral-3-3B-Instruct.

This demo runs the full model in PyTorch. For ONNX inference, use the exported
models from builder.py (model-vision.onnx + model-embedding.onnx) with ONNX Runtime.
See README.md for build instructions.
"""

import torch
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

model_id = "mistralai/Ministral-3-3B-Instruct-2512"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
processor = AutoProcessor.from_pretrained(model_id)
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            },
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

inputs["input_ids"] = inputs["input_ids"].to(device=device)
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(
        dtype=torch.bfloat16, device=device
    )
    image_sizes = [inputs["pixel_values"].shape[-2:]]
else:
    image_sizes = None

# Generate
generated_ids = model.generate(
    **inputs,
    image_sizes=image_sizes,
    max_new_tokens=512,
)[0]

output_text = processor.decode(
    generated_ids[len(inputs["input_ids"][0]):],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output_text)
