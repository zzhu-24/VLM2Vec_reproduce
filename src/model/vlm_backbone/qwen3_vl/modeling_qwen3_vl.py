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

import json
import logging

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger(__name__)

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


def _apply_prune_heads_if_needed(model: PreTrainedModel, config) -> None:
    """
    Structured pruning of attention heads at KV-group granularity for Qwen3-VL.

    Each KV group consists of `q_per_group` Q-heads sharing 1 KV head.
    Pruning a KV group removes the corresponding rows from q_proj, k_proj, v_proj
    and corresponding columns from o_proj, then replaces them with new compact
    nn.Linear modules to ensure physically contiguous weight tensors.

    Config attributes read:
      - head_prune_config: dict  {str(layer_idx): [group_indices_to_prune]}
      - head_prune_n: int        uniform number of groups to prune per layer
                                  (requires head_importance JSON with scores to
                                   auto-select the least important groups)
    """
    head_prune_config = getattr(config, "head_prune_config", None)
    head_prune_n = getattr(config, "head_prune_n", 0)

    if not head_prune_config and (head_prune_n is None or head_prune_n <= 0):
        return

    # ---- Locate language model layers (same logic as _apply_delete_layers_if_needed) ----
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
        logger.warning("Cannot locate decoder layers for head pruning; skipping.")
        return

    layers = language_model.layers
    num_layers = len(layers)

    # ---- Read head layout from the first layer ----
    first_attn = layers[0].self_attn
    head_dim = getattr(first_attn, 'head_dim', None)
    if head_dim is None:
        text_cfg = getattr(config, 'text_config', config)
        head_dim = getattr(text_cfg, 'head_dim', None) or (text_cfg.hidden_size // text_cfg.num_attention_heads)
    # Derive num_heads and num_kv_heads from weight shapes (works across transformers versions)
    num_heads = first_attn.q_proj.weight.shape[0] // head_dim
    num_kv_heads = first_attn.k_proj.weight.shape[0] // head_dim
    q_per_group = num_heads // num_kv_heads        # Q heads per KV group
    num_kv_groups = num_kv_heads                   # 1 group per KV head

    # ---- Build per-layer pruning spec ----
    # prune_spec[layer_idx] = sorted list of KV group indices to prune
    prune_spec = {}

    if head_prune_config:
        # head_prune_config can be a dict or a path to a JSON file
        if isinstance(head_prune_config, str):
            with open(head_prune_config, 'r') as f:
                head_prune_config = json.load(f)
        for layer_key, groups in head_prune_config.items():
            l = int(layer_key)
            if 0 <= l < num_layers:
                valid = sorted(set(int(g) for g in groups if 0 <= int(g) < num_kv_groups))
                if valid:
                    prune_spec[l] = valid

    elif head_prune_n and head_prune_n > 0:
        # Uniform pruning: prune head_prune_n groups from every layer.
        # If an importance JSON is attached, use it to select the least important.
        head_importance = getattr(config, "head_importance", None)
        if head_importance:
            if isinstance(head_importance, str):
                with open(head_importance, 'r') as f:
                    head_importance = json.load(f)
            for l in range(num_layers):
                layer_scores = head_importance.get(str(l), {})
                # Sort groups by importance (ascending) and take head_prune_n least important
                ranked = sorted(range(num_kv_groups),
                                key=lambda g: float(layer_scores.get(str(g), 0.0)))
                prune_spec[l] = sorted(ranked[:head_prune_n])
        else:
            # No importance scores -> prune the last head_prune_n groups from each layer
            logger.warning(
                "head_prune_n=%d specified but no importance scores provided; "
                "pruning the last %d KV groups from every layer.", head_prune_n, head_prune_n
            )
            groups_to_prune = list(range(num_kv_groups - head_prune_n, num_kv_groups))
            for l in range(num_layers):
                prune_spec[l] = groups_to_prune

    if not prune_spec:
        return

    logger.info("Applying attention head pruning (KV-group granularity):")
    for l in sorted(prune_spec.keys()):
        logger.info(f"  Layer {l}: pruning KV groups {prune_spec[l]}")

    # ---- Prune each specified layer ----
    for layer_idx, groups_to_prune in prune_spec.items():
        layer = layers[layer_idx]
        attn = layer.self_attn

        cur_num_heads = attn.q_proj.weight.shape[0] // head_dim
        cur_num_kv_heads = attn.k_proj.weight.shape[0] // head_dim
        cur_q_per_group = cur_num_heads // cur_num_kv_heads
        cur_num_kv_groups = cur_num_kv_heads

        groups_to_keep = sorted(set(range(cur_num_kv_groups)) - set(groups_to_prune))
        if len(groups_to_keep) == 0:
            logger.warning(f"  Layer {layer_idx}: cannot prune all groups; skipping.")
            continue
        if len(groups_to_keep) == cur_num_kv_groups:
            continue  # nothing to prune

        new_num_kv_heads = len(groups_to_keep)
        new_num_heads = new_num_kv_heads * cur_q_per_group

        # --- q_proj: (cur_num_heads * head_dim, hidden_size) ---
        old_q_weight = attn.q_proj.weight.data  # (out_features, in_features)
        hidden_size = old_q_weight.shape[1]
        keep_q_slices = []
        for g in groups_to_keep:
            start = g * cur_q_per_group * head_dim
            end = (g + 1) * cur_q_per_group * head_dim
            keep_q_slices.append(old_q_weight[start:end, :])
        new_q_weight = torch.cat(keep_q_slices, dim=0).clone().contiguous()

        # --- k_proj: (cur_num_kv_heads * head_dim, hidden_size) ---
        old_k_weight = attn.k_proj.weight.data
        keep_k_slices = []
        for g in groups_to_keep:
            start = g * head_dim
            end = (g + 1) * head_dim
            keep_k_slices.append(old_k_weight[start:end, :])
        new_k_weight = torch.cat(keep_k_slices, dim=0).clone().contiguous()

        # --- v_proj: (cur_num_kv_heads * head_dim, hidden_size) ---
        old_v_weight = attn.v_proj.weight.data
        keep_v_slices = []
        for g in groups_to_keep:
            start = g * head_dim
            end = (g + 1) * head_dim
            keep_v_slices.append(old_v_weight[start:end, :])
        new_v_weight = torch.cat(keep_v_slices, dim=0).clone().contiguous()

        # --- o_proj: (hidden_size, cur_num_heads * head_dim) ---
        old_o_weight = attn.o_proj.weight.data
        keep_o_col_slices = []
        for g in groups_to_keep:
            start = g * cur_q_per_group * head_dim
            end = (g + 1) * cur_q_per_group * head_dim
            keep_o_col_slices.append(old_o_weight[:, start:end])
        new_o_weight = torch.cat(keep_o_col_slices, dim=1).clone().contiguous()

        # --- Create new Linear modules ---
        device = old_q_weight.device
        dtype = old_q_weight.dtype

        new_q_proj = nn.Linear(hidden_size, new_num_heads * head_dim, bias=False,
                               device=device, dtype=dtype)
        new_q_proj.weight.data.copy_(new_q_weight)

        new_k_proj = nn.Linear(hidden_size, new_num_kv_heads * head_dim, bias=False,
                               device=device, dtype=dtype)
        new_k_proj.weight.data.copy_(new_k_weight)

        new_v_proj = nn.Linear(hidden_size, new_num_kv_heads * head_dim, bias=False,
                               device=device, dtype=dtype)
        new_v_proj.weight.data.copy_(new_v_weight)

        new_o_proj = nn.Linear(new_num_heads * head_dim, hidden_size, bias=False,
                               device=device, dtype=dtype)
        new_o_proj.weight.data.copy_(new_o_weight)

        # --- Replace modules on the attention layer ---
        attn.q_proj = new_q_proj
        attn.k_proj = new_k_proj
        attn.v_proj = new_v_proj
        attn.o_proj = new_o_proj

        # --- Update attention attributes ---
        attn.num_heads = new_num_heads
        attn.num_key_value_heads = new_num_kv_heads
        attn.num_key_value_groups = new_num_heads // new_num_kv_heads
        # head_dim stays the same

        # --- Contiguity and dimension assertions ---
        assert new_q_proj.weight.data.is_contiguous(), \
            f"Layer {layer_idx}: q_proj weight is not contiguous after pruning"
        assert new_k_proj.weight.data.is_contiguous(), \
            f"Layer {layer_idx}: k_proj weight is not contiguous after pruning"
        assert new_v_proj.weight.data.is_contiguous(), \
            f"Layer {layer_idx}: v_proj weight is not contiguous after pruning"
        assert new_o_proj.weight.data.is_contiguous(), \
            f"Layer {layer_idx}: o_proj weight is not contiguous after pruning"
        assert new_q_proj.weight.shape[0] == new_num_heads * head_dim, \
            f"Layer {layer_idx}: q_proj shape mismatch after pruning"
        assert new_k_proj.weight.shape[0] == new_num_kv_heads * head_dim, \
            f"Layer {layer_idx}: k_proj shape mismatch after pruning"
        assert new_o_proj.weight.shape[1] == new_num_heads * head_dim, \
            f"Layer {layer_idx}: o_proj column dim mismatch after pruning"
        assert new_num_heads % new_num_kv_heads == 0, \
            f"Layer {layer_idx}: GQA grouping broken after pruning"

        logger.info(
            f"  Layer {layer_idx} pruned: {cur_num_heads} -> {new_num_heads} Q-heads, "
            f"{cur_num_kv_heads} -> {new_num_kv_heads} KV-heads"
        )


def _apply_prune_mlp_if_needed(model: PreTrainedModel, config) -> None:
    """
    Structured pruning of MLP intermediate neurons for Qwen3-VL (SwiGLU).

    Each MLP layer has:
      gate_proj: Linear(hidden_size, intermediate_size)  # weight (intermediate_size, hidden_size)
      up_proj:   Linear(hidden_size, intermediate_size)  # weight (intermediate_size, hidden_size)
      down_proj: Linear(intermediate_size, hidden_size)  # weight (hidden_size, intermediate_size)

    Pruning removes intermediate neurons (rows of gate/up_proj, columns of down_proj)
    based on importance scores. New nn.Linear modules are created to ensure contiguous
    weight tensors (no vector fragmentation).

    Config attributes read:
      - mlp_prune_ratio: float (0~1), fraction of intermediate neurons to prune
      - mlp_importance: dict {str(layer_idx): [neuron_scores]} or None
    """
    mlp_prune_ratio = getattr(config, "mlp_prune_ratio", 0.0)

    if mlp_prune_ratio is None or mlp_prune_ratio <= 0.0:
        return

    if mlp_prune_ratio >= 1.0:
        logger.warning("mlp_prune_ratio >= 1.0 would remove all MLP neurons; skipping.")
        return

    # ---- Locate language model layers ----
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
        logger.warning("Cannot locate decoder layers for MLP pruning; skipping.")
        return

    layers = language_model.layers
    num_layers = len(layers)

    # ---- Load importance scores ----
    mlp_importance = getattr(config, "mlp_importance", None)
    if mlp_importance is not None and isinstance(mlp_importance, str):
        with open(mlp_importance, 'r') as f:
            mlp_importance = json.load(f)

    if mlp_importance is None:
        logger.warning(
            "mlp_prune_ratio=%.3f specified but no mlp_importance scores provided; "
            "pruning the last N intermediate neurons (arbitrary) from each layer.",
            mlp_prune_ratio
        )

    # ---- Prune each layer's MLP ----
    first_mlp = layers[0].mlp
    original_intermediate_size = first_mlp.gate_proj.weight.shape[0]  # (intermediate_size, hidden_size)
    new_intermediate_size = round(original_intermediate_size * (1.0 - mlp_prune_ratio))
    if new_intermediate_size <= 0:
        logger.warning("MLP pruning would remove all neurons; skipping.")
        return

    logger.info(
        "Applying MLP pruning: ratio=%.3f, intermediate_size %d -> %d (removing %d neurons/layer)",
        mlp_prune_ratio, original_intermediate_size, new_intermediate_size,
        original_intermediate_size - new_intermediate_size
    )

    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        mlp = layer.mlp

        cur_intermediate_size = mlp.gate_proj.weight.shape[0]

        # Determine which neurons to keep
        if mlp_importance is not None and str(layer_idx) in mlp_importance:
            scores = mlp_importance[str(layer_idx)]
            if len(scores) != cur_intermediate_size:
                logger.warning(
                    f"  Layer {layer_idx}: importance scores length ({len(scores)}) != "
                    f"intermediate_size ({cur_intermediate_size}); using arbitrary pruning."
                )
                keep_indices = list(range(new_intermediate_size))
            else:
                # Sort neuron indices by importance descending, keep top-K
                indexed_scores = list(enumerate(scores))
                indexed_scores.sort(key=lambda x: x[1], reverse=True)
                keep_indices = sorted([idx for idx, _ in indexed_scores[:new_intermediate_size]])
        else:
            # No importance for this layer: keep first new_intermediate_size neurons
            keep_indices = list(range(new_intermediate_size))

        keep_indices_tensor = torch.tensor(keep_indices, dtype=torch.long)

        # ---- gate_proj: weight shape (intermediate_size, hidden_size) -> keep rows ----
        old_gate_weight = mlp.gate_proj.weight.data
        hidden_size = old_gate_weight.shape[1]
        new_gate_weight = old_gate_weight[keep_indices_tensor, :].clone().contiguous()

        # ---- up_proj: weight shape (intermediate_size, hidden_size) -> keep rows ----
        old_up_weight = mlp.up_proj.weight.data
        new_up_weight = old_up_weight[keep_indices_tensor, :].clone().contiguous()

        # ---- down_proj: weight shape (hidden_size, intermediate_size) -> keep columns ----
        old_down_weight = mlp.down_proj.weight.data
        new_down_weight = old_down_weight[:, keep_indices_tensor].clone().contiguous()

        # ---- Create new Linear modules ----
        device = old_gate_weight.device
        dtype = old_gate_weight.dtype

        new_gate_proj = nn.Linear(hidden_size, new_intermediate_size, bias=False,
                                  device=device, dtype=dtype)
        new_gate_proj.weight.data.copy_(new_gate_weight)

        new_up_proj = nn.Linear(hidden_size, new_intermediate_size, bias=False,
                                device=device, dtype=dtype)
        new_up_proj.weight.data.copy_(new_up_weight)

        new_down_proj = nn.Linear(new_intermediate_size, hidden_size, bias=False,
                                  device=device, dtype=dtype)
        new_down_proj.weight.data.copy_(new_down_weight)

        # ---- Replace modules on the MLP layer ----
        mlp.gate_proj = new_gate_proj
        mlp.up_proj = new_up_proj
        mlp.down_proj = new_down_proj

        # ---- Update intermediate_size attribute if present ----
        if hasattr(mlp, 'intermediate_size'):
            mlp.intermediate_size = new_intermediate_size

        # ---- Contiguity and dimension assertions ----
        assert new_gate_proj.weight.data.is_contiguous(), \
            f"Layer {layer_idx}: gate_proj weight is not contiguous after MLP pruning"
        assert new_up_proj.weight.data.is_contiguous(), \
            f"Layer {layer_idx}: up_proj weight is not contiguous after MLP pruning"
        assert new_down_proj.weight.data.is_contiguous(), \
            f"Layer {layer_idx}: down_proj weight is not contiguous after MLP pruning"
        assert new_gate_proj.weight.shape == (new_intermediate_size, hidden_size), \
            f"Layer {layer_idx}: gate_proj shape mismatch after MLP pruning"
        assert new_up_proj.weight.shape == (new_intermediate_size, hidden_size), \
            f"Layer {layer_idx}: up_proj shape mismatch after MLP pruning"
        assert new_down_proj.weight.shape == (hidden_size, new_intermediate_size), \
            f"Layer {layer_idx}: down_proj shape mismatch after MLP pruning"

        logger.info(
            f"  Layer {layer_idx} MLP pruned: intermediate_size {cur_intermediate_size} -> {new_intermediate_size}"
        )


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

        # Apply attention head pruning (KV-group granularity) after layer deletion.
        _apply_prune_heads_if_needed(model=model, config=config)

        # Apply MLP intermediate neuron pruning after head pruning.
        _apply_prune_mlp_if_needed(model=model, config=config)
        
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

