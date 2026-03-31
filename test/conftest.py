"""Shared test configuration for Ministral-3-3B-Instruct ONNX export tests."""

import os
import sys

from transformers import Mistral3Config

# Ensure repo root is on the path for all test files
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def create_test_config():
    """Create a minimal Mistral3Config for testing (matches builder.py --no_weights).

    This is the single source of truth for test model configuration.
    builder.py has its own copy in __main__ for CLI use.
    """
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
