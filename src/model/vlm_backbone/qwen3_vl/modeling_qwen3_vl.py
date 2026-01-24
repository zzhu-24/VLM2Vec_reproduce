# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Qwen3-VL model wrapper for VLM2Vec.
This module provides a wrapper around transformers' native Qwen3-VL implementation.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_utils import PreTrainedModel

# Try to import Qwen3-VL from transformers, fallback to AutoModelForCausalLM
try:
    from transformers import Qwen3VLForConditionalGeneration as TransformersQwen3VLForConditionalGeneration
    HAS_NATIVE_QWEN3 = True
except ImportError:
    HAS_NATIVE_QWEN3 = False
    TransformersQwen3VLForConditionalGeneration = None


def _apply_delete_layers_if_needed(model: PreTrainedModel, config) -> None:
    """
    Apply `delete_L` / `delete_n` behavior (implemented in our Qwen2-VL fork) to Qwen3-VL.

    In `src/model/vlm_backbone/qwen2_vl/modeling_qwen2_vl.py`, the decoder layers are constructed as:
      [0, ..., delete_L - delete_n - 1] + [delete_L, ..., num_hidden_layers - 1]
    i.e., removing `delete_n` layers right before index `delete_L`.

    Now supports list form: delete_L and delete_n are lists, and elements are paired in order
    to form multiple deletion segments. The shorter list determines the number of pairs.

    HF's Qwen3-VL doesn't support these knobs natively, so we prune the loaded language decoder layers in-place.
    """
    delete_n = getattr(config, "delete_n", None)
    if delete_n is None:
        return

    # Qwen3-VL uses nested configs; use text_config.num_hidden_layers as the reference total.
    text_cfg = getattr(config, "text_config", None)
    num_layers = getattr(text_cfg, "num_hidden_layers", None)
    if num_layers is None:
        return

    delete_L = getattr(config, "delete_L", None)
    if delete_L is None:
        delete_L = num_layers

    # Convert to lists if not already (for backward compatibility)
    if not isinstance(delete_L, (list, tuple)):
        delete_L = [delete_L]
    if not isinstance(delete_n, (list, tuple)):
        delete_n = [delete_n]

    # Use the shorter list length (truncate the longer one)
    min_len = min(len(delete_L), len(delete_n))
    delete_L = list(delete_L[:min_len])
    delete_n = list(delete_n[:min_len])

    # If all delete_n are <= 0, no deletion needed
    if all(int(n) <= 0 for n in delete_n):
        return

    # Convert to integers and clamp to valid ranges
    delete_L = [max(0, min(int(L), num_layers)) for L in delete_L]
    delete_n = [max(0, min(int(n), L)) for L, n in zip(delete_L, delete_n)]

    # Collect all indices to delete
    delete_indices_set = set()
    for L, n in zip(delete_L, delete_n):
        if n > 0:
            start = L - n
            # Delete layers from start to L-1 (inclusive)
            delete_indices_set.update(range(start, L))

    # If nothing to delete, return early
    if not delete_indices_set:
        return

    # Locate Qwen3 language model container.
    # - Transformers Qwen3VLForConditionalGeneration: model.model.language_model.layers
    # - Transformers Qwen3VLModel: model.language_model.layers
    # - Fallbacks: model.model.layers / model.layers
    language_model = None
    if hasattr(model, "model") and hasattr(getattr(model, "model"), "language_model"):
        language_model = getattr(model, "model").language_model
    elif hasattr(model, "language_model"):
        language_model = getattr(model, "language_model")
    elif hasattr(model, "model") and hasattr(getattr(model, "model"), "layers"):
        language_model = getattr(model, "model")
    elif hasattr(model, "layers"):
        language_model = model

    if language_model is None or not hasattr(language_model, "layers"):
        return

    layers = getattr(language_model, "layers")
    if not isinstance(layers, (nn.ModuleList, list, tuple)):
        return

    if len(layers) < num_layers:
        # Already modified elsewhere; avoid double-pruning.
        return

    # Keep all indices that are not in delete_indices_set
    keep_indices = [i for i in range(num_layers) if i not in delete_indices_set]
    # Re-wrap as ModuleList so parameters remain registered.
    language_model.layers = nn.ModuleList([layers[i] for i in keep_indices])


class Qwen3VLForConditionalGeneration(PreTrainedModel):
    """
    Qwen3-VL model wrapper for VLM2Vec.
    This class wraps transformers' native Qwen3-VL implementation to ensure compatibility.
    """
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        # Use transformers native implementation if available
        if HAS_NATIVE_QWEN3 and TransformersQwen3VLForConditionalGeneration is not None:
            self.model = TransformersQwen3VLForConditionalGeneration(config, *args, **kwargs)
        else:
            # Fallback to AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, *args, **kwargs)
        
        # Copy config reference
        self.config = self.model.config
    
    @classmethod
    def _from_config(cls, config, **kwargs):
        """Create model from config (used internally)."""
        instance = cls.__new__(cls)
        # Initialize the base class first
        PreTrainedModel.__init__(instance, config)
        # Then set the model
        if HAS_NATIVE_QWEN3 and TransformersQwen3VLForConditionalGeneration is not None:
            instance.model = TransformersQwen3VLForConditionalGeneration(config, **kwargs)
        else:
            instance.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, **kwargs)
        instance.config = instance.model.config
        return instance
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, config=None, *args, **kwargs):
        """
        Load a pretrained Qwen3-VL model.
        """
        if config is None:
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        
        # Handle vision config compatibility
        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
            # Ensure fullatt_block_indexes exists for compatibility
            if not hasattr(vision_config, 'fullatt_block_indexes'):
                # Set default value based on Qwen2.5-VL pattern
                # This is a workaround for Qwen3-VL config differences
                vision_config.fullatt_block_indexes = [7, 15, 23, 31]  # Default from Qwen2.5-VL
        
        # Use transformers native implementation if available
        if HAS_NATIVE_QWEN3 and TransformersQwen3VLForConditionalGeneration is not None:
            model = TransformersQwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path, 
                config=config,
                trust_remote_code=True,
                *args, 
                **kwargs
            )
        else:
            # Fallback to AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                trust_remote_code=True,
                *args,
                **kwargs
            )

        # Apply VLM2Vec layer-deletion knobs (delete_L/delete_n) for Qwen3-VL.
        _apply_delete_layers_if_needed(model=model, config=config)
        
        # For now, directly return the model since it already implements the required interface
        # This avoids PyTorch's attribute setting restrictions
        return model
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)
    
    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """Set input embeddings."""
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        """Get output embeddings."""
        return self.model.get_output_embeddings()
    
    def set_output_embeddings(self, value):
        """Set output embeddings."""
        self.model.set_output_embeddings(value)
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying model."""
        if name in ['model', 'config']:
            return super().__getattribute__(name)
        try:
            return getattr(self.model, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

