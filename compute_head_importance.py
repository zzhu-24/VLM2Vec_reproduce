"""
Compute attention head AND MLP neuron importance scores for Qwen3-VL LLM decoder layers.

Uses Taylor importance (Michel et al., 2019): importance = mean(|activation * gradient|)

- Attention heads: scored per KV-group via hooks on o_proj input
- MLP neurons: scored per intermediate neuron via hooks on down_proj input

Outputs:
  - head_importance.json   -- per-layer per-KV-group importance scores
  - mlp_importance.json    -- per-layer per-neuron importance scores
  - head_importance_heatmap.png, head_importance_ranking.png
  - mlp_importance_heatmap.png, mlp_importance_boxplot.png
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: locate the LLM decoder layers inside a (possibly wrapped) model
# ---------------------------------------------------------------------------

def _get_language_model_layers(model):
    """
    Traverse common wrapper patterns used by Qwen3-VL / HF to reach
    the nn.ModuleList of decoder layers.
    Returns (layers: nn.ModuleList, language_model_ref).
    """
    # Unwrap MMEBModel -> encoder -> (optional PeftModel base_model) -> actual VLM
    encoder = model
    if hasattr(encoder, 'encoder'):
        encoder = encoder.encoder
    # PeftModel wrapping
    if hasattr(encoder, 'base_model'):
        encoder = encoder.base_model
    if hasattr(encoder, 'model'):
        encoder = encoder.model  # PeftModel.model -> actual model

    # Now encoder should be Qwen3VLForConditionalGeneration (HF native)
    # Structure: model.model.language_model.layers  OR  model.language_model.layers
    lm = None
    if hasattr(encoder, 'model') and hasattr(encoder.model, 'language_model'):
        lm = encoder.model.language_model
    elif hasattr(encoder, 'language_model'):
        lm = encoder.language_model
    elif hasattr(encoder, 'model') and hasattr(encoder.model, 'layers'):
        lm = encoder.model
    elif hasattr(encoder, 'layers'):
        lm = encoder

    if lm is None or not hasattr(lm, 'layers'):
        raise RuntimeError(
            "Cannot locate decoder layers. Tried model.model.language_model.layers "
            "and several fallbacks."
        )
    return lm.layers, lm


def _get_attn_module(decoder_layer):
    """Return the self-attention sub-module from a decoder layer."""
    if hasattr(decoder_layer, 'self_attn'):
        return decoder_layer.self_attn
    raise RuntimeError(f"Cannot find self_attn in {type(decoder_layer)}")


def _get_attn_head_info(attn):
    """
    Get (num_heads, num_kv_heads, head_dim) from an attention module.
    Handles different attribute naming across transformers versions:
      - Some store attn.num_heads / attn.num_key_value_heads directly
      - Newer Qwen3VLTextAttention stores them only in attn.config
    Falls back to deriving from weight shapes.
    """
    # head_dim
    head_dim = getattr(attn, 'head_dim', None)
    if head_dim is None:
        cfg = getattr(attn, 'config', None)
        if cfg:
            head_dim = getattr(cfg, 'head_dim', None)
            if head_dim is None:
                head_dim = cfg.hidden_size // cfg.num_attention_heads

    # num_heads (total Q heads)
    num_heads = getattr(attn, 'num_heads', None)
    if num_heads is None:
        cfg = getattr(attn, 'config', None)
        if cfg:
            num_heads = getattr(cfg, 'num_attention_heads', None)
    if num_heads is None and head_dim:
        num_heads = attn.q_proj.weight.shape[0] // head_dim

    # num_kv_heads
    num_kv_heads = getattr(attn, 'num_key_value_heads', None)
    if num_kv_heads is None:
        cfg = getattr(attn, 'config', None)
        if cfg:
            num_kv_heads = getattr(cfg, 'num_key_value_heads', None)
    if num_kv_heads is None and head_dim:
        num_kv_heads = attn.k_proj.weight.shape[0] // head_dim

    return num_heads, num_kv_heads, head_dim


def _get_mlp_module(decoder_layer):
    """Return the MLP sub-module from a decoder layer."""
    if hasattr(decoder_layer, 'mlp'):
        return decoder_layer.mlp
    raise RuntimeError(f"Cannot find mlp in {type(decoder_layer)}")


# ---------------------------------------------------------------------------
# Hook-based importance accumulation
# ---------------------------------------------------------------------------

class HeadImportanceAccumulator:
    """
    Registers forward + backward hooks on the *output projection* (o_proj) of
    each decoder layer's attention.  From the activation flowing *into* o_proj
    we can recover per-head information.

    For a GQA model with `num_heads` Q-heads grouped into `num_kv_groups` KV
    groups (each group has `q_per_group` Q-heads), the importance of KV-group g
    is computed as the sum of Taylor importance over all Q-heads in that group.
    """

    def __init__(self, layers: nn.ModuleList):
        self.num_layers = len(layers)
        self.hooks = []
        self.activations = {}   # layer_idx -> tensor
        self.importance = None  # (num_layers, num_kv_groups) accumulated

        # Inspect first layer to discover head layout
        attn0 = _get_attn_module(layers[0])
        self.num_heads, self.num_kv_heads, self.head_dim = _get_attn_head_info(attn0)
        self.q_per_group = self.num_heads // self.num_kv_heads
        self.num_kv_groups = self.num_kv_heads  # one group per KV head

        self.importance = np.zeros((self.num_layers, self.num_kv_groups), dtype=np.float64)
        self.num_samples = 0

        # Register hooks on o_proj *input* (= attention head outputs before projection)
        for layer_idx, layer in enumerate(layers):
            attn = _get_attn_module(layer)
            # We hook the o_proj Linear.  Its input has shape
            # (batch, seq_len, num_heads * head_dim).
            h_fwd = attn.o_proj.register_forward_hook(
                self._make_fwd_hook(layer_idx)
            )
            h_bwd = attn.o_proj.register_full_backward_hook(
                self._make_bwd_hook(layer_idx)
            )
            self.hooks.extend([h_fwd, h_bwd])

    # -- hook factories --------------------------------------------------

    def _make_fwd_hook(self, layer_idx):
        def hook_fn(module, inp, out):
            # inp[0] shape: (batch, seq, num_heads * head_dim)
            self.activations[layer_idx] = inp[0].detach()
        return hook_fn

    def _make_bwd_hook(self, layer_idx):
        def hook_fn(module, grad_input, grad_output):
            # grad_input[0] corresponds to the input of o_proj (same shape as activation)
            act = self.activations.get(layer_idx)
            grad = grad_input[0]
            if act is None or grad is None:
                return
            # Taylor importance: |activation * gradient|, summed over batch & seq
            taylor = (act.float() * grad.float()).abs()  # (B, S, num_heads*head_dim)
            # Reshape to (B, S, num_kv_groups, q_per_group, head_dim)
            B, S, _ = taylor.shape
            taylor = taylor.view(B, S, self.num_kv_groups, self.q_per_group, self.head_dim)
            # Sum over batch, seq, q_per_group, head_dim -> (num_kv_groups,)
            group_importance = taylor.sum(dim=(0, 1, 3, 4)).cpu().numpy()
            self.importance[layer_idx] += group_importance
            self.num_samples += B  # only count once per backward; see note below

            # Clean up to save memory
            del self.activations[layer_idx]
        return hook_fn

    def finalize(self):
        """Average accumulated importance over samples."""
        # num_samples was incremented once per layer per backward, so divide
        # by (num_samples / num_layers) to get per-sample average.
        total_backward_calls = self.importance.sum()
        if self.num_samples > 0:
            # Each backward triggers num_layers hooks, each adding B to num_samples
            effective_samples = self.num_samples / self.num_layers
            self.importance /= max(effective_samples, 1.0)
        return self.importance

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class MlpImportanceAccumulator:
    """
    Registers forward + backward hooks on the *down_proj* of each decoder
    layer's MLP.  The input to down_proj has shape (B, S, intermediate_size)
    and represents the SwiGLU-gated intermediate activations.

    Taylor importance per neuron n: importance(l, n) = sum(|act_n * grad_n|)
    """

    def __init__(self, layers: nn.ModuleList):
        self.num_layers = len(layers)
        self.hooks = []
        self.activations = {}  # layer_idx -> tensor

        # Discover intermediate_size from first layer
        mlp0 = _get_mlp_module(layers[0])
        self.intermediate_size = mlp0.down_proj.in_features
        self.importance = np.zeros((self.num_layers, self.intermediate_size), dtype=np.float64)
        self.num_samples = 0

        for layer_idx, layer in enumerate(layers):
            mlp = _get_mlp_module(layer)
            h_fwd = mlp.down_proj.register_forward_hook(
                self._make_fwd_hook(layer_idx)
            )
            h_bwd = mlp.down_proj.register_full_backward_hook(
                self._make_bwd_hook(layer_idx)
            )
            self.hooks.extend([h_fwd, h_bwd])

    def _make_fwd_hook(self, layer_idx):
        def hook_fn(module, inp, out):
            # inp[0] shape: (B, S, intermediate_size)
            self.activations[layer_idx] = inp[0].detach()
        return hook_fn

    def _make_bwd_hook(self, layer_idx):
        def hook_fn(module, grad_input, grad_output):
            act = self.activations.get(layer_idx)
            grad = grad_input[0]
            if act is None or grad is None:
                return
            # Taylor importance per neuron: |act * grad|, summed over batch & seq
            taylor = (act.float() * grad.float()).abs()  # (B, S, intermediate_size)
            neuron_importance = taylor.sum(dim=(0, 1)).cpu().numpy()  # (intermediate_size,)
            self.importance[layer_idx] += neuron_importance
            self.num_samples += taylor.shape[0]  # B

            del self.activations[layer_idx]
        return hook_fn

    def finalize(self):
        """Average accumulated importance over samples."""
        if self.num_samples > 0:
            effective_samples = self.num_samples / self.num_layers
            self.importance /= max(effective_samples, 1.0)
        return self.importance

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_heatmap(importance: np.ndarray, output_path: str):
    """Generate and save a heatmap of head importance scores."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib / seaborn not installed; skipping heatmap generation.")
        return

    num_layers, num_groups = importance.shape

    # Normalize to [0, 1] for color mapping
    vmin, vmax = importance.min(), importance.max()
    if vmax - vmin < 1e-12:
        norm_imp = np.zeros_like(importance)
    else:
        norm_imp = (importance - vmin) / (vmax - vmin)

    fig, ax = plt.subplots(figsize=(max(8, num_groups * 1.2), max(6, num_layers * 0.5)))
    sns.heatmap(
        norm_imp,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        xticklabels=[f'G{i}' for i in range(num_groups)],
        yticklabels=[f'L{i}' for i in range(num_layers)],
        ax=ax,
        vmin=0, vmax=1,
        linewidths=0.5,
    )
    ax.set_xlabel('KV Group Index')
    ax.set_ylabel('Layer Index (post layer-deletion)')
    ax.set_title('Attention Head Importance (Taylor, normalized)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Heatmap saved to {output_path}")


def plot_global_ranking(importance: np.ndarray, output_path: str):
    """Bar chart of all (layer, group) pairs sorted by importance."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping ranking bar chart.")
        return

    num_layers, num_groups = importance.shape
    entries = []
    for l in range(num_layers):
        for g in range(num_groups):
            entries.append((f'L{l}-G{g}', importance[l, g]))
    entries.sort(key=lambda x: x[1])

    labels = [e[0] for e in entries]
    values = [e[1] for e in entries]

    fig, ax = plt.subplots(figsize=(max(10, len(entries) * 0.3), 6))
    colors = plt.cm.RdYlGn(np.linspace(0, 1, len(entries)))
    ax.barh(range(len(entries)), values, color=colors)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels(labels, fontsize=max(4, 8 - len(entries) // 20))
    ax.set_xlabel('Importance Score (Taylor)')
    ax.set_title('Global Head Importance Ranking (ascending)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Ranking chart saved to {output_path}")


def print_importance_table(importance: np.ndarray):
    """Pretty-print the importance matrix to stdout."""
    num_layers, num_groups = importance.shape
    header = '       ' + ''.join([f'  G{g:<6d}' for g in range(num_groups)])
    print('\n' + '=' * len(header))
    print('  Head Importance Scores (Taylor)')
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for l in range(num_layers):
        row = f'  L{l:<3d} '
        for g in range(num_groups):
            row += f'  {importance[l, g]:<8.4f}'
        print(row)
    print('=' * len(header) + '\n')


# -- MLP visualization functions ------------------------------------------

def plot_mlp_heatmap(importance: np.ndarray, output_path: str, bin_size: int = 128):
    """
    Generate a heatmap of MLP neuron importance, binned for readability.
    importance shape: (num_layers, intermediate_size)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib / seaborn not installed; skipping MLP heatmap.")
        return

    num_layers, intermediate_size = importance.shape
    num_bins = (intermediate_size + bin_size - 1) // bin_size

    # Aggregate neuron importance into bins (mean within each bin)
    binned = np.zeros((num_layers, num_bins), dtype=np.float64)
    for b in range(num_bins):
        start = b * bin_size
        end = min(start + bin_size, intermediate_size)
        binned[:, b] = importance[:, start:end].mean(axis=1)

    # Normalize to [0, 1]
    vmin, vmax = binned.min(), binned.max()
    if vmax - vmin < 1e-12:
        norm_binned = np.zeros_like(binned)
    else:
        norm_binned = (binned - vmin) / (vmax - vmin)

    fig, ax = plt.subplots(figsize=(max(12, num_bins * 0.25), max(6, num_layers * 0.5)))
    sns.heatmap(
        norm_binned,
        cmap='YlOrRd',
        xticklabels=[f'{b * bin_size}' for b in range(num_bins)],
        yticklabels=[f'L{i}' for i in range(num_layers)],
        ax=ax,
        vmin=0, vmax=1,
        linewidths=0.3,
    )
    ax.set_xlabel(f'Neuron Bin (each bin = {bin_size} neurons)')
    ax.set_ylabel('Layer Index (post layer-deletion)')
    ax.set_title('MLP Neuron Importance (Taylor, binned & normalized)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"MLP heatmap saved to {output_path}")


def plot_mlp_boxplot(importance: np.ndarray, output_path: str):
    """
    Box-plot showing distribution of MLP neuron importance per layer.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping MLP boxplot.")
        return

    num_layers = importance.shape[0]
    fig, ax = plt.subplots(figsize=(max(10, num_layers * 0.8), 6))

    data_for_box = [importance[l, :] for l in range(num_layers)]
    bp = ax.boxplot(data_for_box, vert=True, patch_artist=True, showfliers=False)

    # Color boxes by median importance
    medians = [np.median(d) for d in data_for_box]
    med_min, med_max = min(medians), max(medians)
    cmap = plt.cm.YlOrRd
    for i, patch in enumerate(bp['boxes']):
        if med_max - med_min > 1e-12:
            norm_val = (medians[i] - med_min) / (med_max - med_min)
        else:
            norm_val = 0.5
        patch.set_facecolor(cmap(norm_val))

    ax.set_xticklabels([f'L{l}' for l in range(num_layers)])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Neuron Importance (Taylor)')
    ax.set_title('MLP Neuron Importance Distribution per Layer')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"MLP boxplot saved to {output_path}")


def print_mlp_importance_summary(importance: np.ndarray, prune_ratios=(0.10, 0.25, 0.50)):
    """Print summary statistics for MLP neuron importance."""
    num_layers, intermediate_size = importance.shape
    print('\n' + '=' * 80)
    print('  MLP Neuron Importance Summary (Taylor)')
    print('=' * 80)
    print(f'  Layers: {num_layers}, Intermediate size: {intermediate_size}')
    print('-' * 80)
    header = f'  {"Layer":<8s} {"Mean":>10s} {"Std":>10s} {"Min":>10s} {"Max":>10s}'
    for r in prune_ratios:
        header += f' {"Thr@" + str(int(r*100)) + "%":>10s}'
    print(header)
    print('-' * 80)
    for l in range(num_layers):
        row_scores = importance[l]
        sorted_scores = np.sort(row_scores)
        line = f'  L{l:<6d} {row_scores.mean():>10.4f} {row_scores.std():>10.4f} {row_scores.min():>10.4f} {row_scores.max():>10.4f}'
        for r in prune_ratios:
            n_prune = int(intermediate_size * r)
            threshold = sorted_scores[n_prune] if n_prune < intermediate_size else sorted_scores[-1]
            line += f' {threshold:>10.4f}'
        print(line)
    print('=' * 80 + '\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Compute attention head importance for Qwen3-VL')
    parser.add_argument('--model_name', type=str, required=True,
                        help='HuggingFace model name or path (e.g. Qwen/Qwen3-VL-4B-Instruct)')
    parser.add_argument('--dataset_config', type=str, required=True,
                        help='YAML config for calibration dataset (same format as training)')
    parser.add_argument('--data_basedir', type=str, default=None,
                        help='Base directory for datasets')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save importance scores and visualizations')
    parser.add_argument('--num_calibration_batches', type=int, default=32,
                        help='Number of calibration batches to process')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for calibration')
    parser.add_argument('--delete_L', type=int, nargs='+', default=[36],
                        help='Layer deletion parameter (list)')
    parser.add_argument('--delete_n', type=int, nargs='+', default=[0],
                        help='Number of layers to delete (list)')
    parser.add_argument('--pooling', type=str, default='eos')
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--temperature', type=float, default=0.02)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--resize_min_pixels', type=int, default=28*28*4)
    parser.add_argument('--resize_max_pixels', type=int, default=28*28*1280)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # ---- Build model (reuse the same config path as training) ----
    from transformers import HfArgumentParser, AutoConfig
    from src.arguments import ModelArguments, DataArguments
    from src.model.model import MMEBModel
    from src.model.processor import load_processor, get_backbone_name, QWEN3_VL

    # Construct a minimal ModelArguments
    model_args = ModelArguments(
        model_name=args.model_name,
        pooling=args.pooling,
        normalize=args.normalize,
        temperature=args.temperature,
        lora=False,  # No LoRA for importance evaluation
        delete_L=args.delete_L,
        delete_n=args.delete_n,
    )

    logger.info(f"Building model: {args.model_name}")
    logger.info(f"Layer deletion: delete_L={args.delete_L}, delete_n={args.delete_n}")
    model = MMEBModel.build(model_args)

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    model_backbone = get_backbone_name(hf_config=config)
    setattr(model_args, 'model_backbone', model_backbone)

    # Move to device
    model = model.to(device)
    if args.bf16:
        model = model.to(torch.bfloat16)
    model.eval()

    # ---- Locate decoder layers ----
    layers, lm = _get_language_model_layers(model)
    num_layers = len(layers)
    attn0 = _get_attn_module(layers[0])
    _num_heads, _num_kv_heads, _head_dim = _get_attn_head_info(attn0)
    logger.info(f"Found {num_layers} decoder layers after layer deletion")
    logger.info(f"  num_heads={_num_heads}, num_kv_heads={_num_kv_heads}, "
                f"head_dim={_head_dim}, q_per_group={_num_heads // _num_kv_heads}")

    # ---- Setup importance accumulators (head + MLP) ----
    head_accumulator = HeadImportanceAccumulator(layers)
    mlp_accumulator = MlpImportanceAccumulator(layers)
    logger.info(f"  MLP intermediate_size={mlp_accumulator.intermediate_size}")

    # ---- Load processor and data ----
    data_args = DataArguments(
        dataset_config=args.dataset_config,
        data_basedir=args.data_basedir,
        resize_min_pixels=args.resize_min_pixels,
        resize_max_pixels=args.resize_max_pixels,
    )
    processor = load_processor(model_args, data_args)
    setattr(model, 'processor', processor)

    from src.model.processor import process_vlm_inputs_fns
    process_fn = process_vlm_inputs_fns.get(model_backbone)
    if process_fn is None:
        raise RuntimeError(f"No process function found for backbone {model_backbone}")

    # Load dataset
    with open(args.dataset_config, 'r') as f:
        dataset_config = yaml.safe_load(f)

    from src.data.dataset.base_pair_dataset import AutoPairDataset
    # Use just the first dataset for calibration
    first_key = list(dataset_config.keys())[0]
    first_cfg = dataset_config[first_key]
    logger.info(f"Using calibration dataset: {first_key}")

    # Build a minimal training_args-like object for dataset instantiation
    class _FakeTrainingArgs:
        per_device_train_batch_size = args.batch_size
        seed = 42
        interleave_batch_size = 0
        interleave_stopping_strategy = "all_exhausted"
    
    fake_training_args = _FakeTrainingArgs()

    cal_dataset = AutoPairDataset.instantiate(
        model_args=model_args, data_args=data_args,
        training_args=fake_training_args, **first_cfg
    )
    logger.info(f"Calibration dataset has {cal_dataset.num_rows} rows")

    # ---- Prepare collator ----
    from src.data.collator.train_collator import MultimodalDataCollator

    class _FakeTrainingArgsForCollator:
        model_backbone = model_backbone
    
    collator = MultimodalDataCollator(processor, model_args, data_args, _FakeTrainingArgsForCollator())

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        cal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Single process for simplicity
        drop_last=True,
    )

    # ---- Run calibration forward + backward passes ----
    logger.info(f"Running {args.num_calibration_batches} calibration batches (batch_size={args.batch_size})...")
    model.eval_layers = [-1]
    model.joint_training_layers = [-1]

    num_processed = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.num_calibration_batches:
            break

        # The collator returns (qry_dict, pos_dict) tuple
        qry, tgt = batch

        if qry is None and tgt is None:
            continue

        # Move tensors to device
        def move_to_device(d):
            if d is None:
                return None
            result = {}
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v.to(device)
                elif isinstance(v, list):
                    result[k] = [
                        x.to(device) if isinstance(x, torch.Tensor) else x
                        for x in v
                    ]
                else:
                    result[k] = v
            return result

        # Process query side (or target side if query is None)
        inputs = move_to_device(qry) if qry is not None else move_to_device(tgt)
        if inputs is None:
            continue

        # Remove non-model keys
        for k in ['texts', 'images', 'global_dataset_name']:
            inputs.pop(k, None)

        # Convert list-form pixel_values / image_grid_thw to tensors
        # (Qwen3-VL collator keeps them as lists because samples have different image counts)
        def _consolidate_visual_inputs(inp):
            for pv_key, grid_key in [('pixel_values', 'image_grid_thw'),
                                      ('pixel_values_videos', 'video_grid_thw')]:
                if pv_key in inp and isinstance(inp[pv_key], list):
                    bsz = inp['input_ids'].shape[0]
                    idx_with_visual = [
                        i for i in range(bsz)
                        if inp[pv_key][i] is not None and (
                            grid_key not in inp or not isinstance(inp.get(grid_key), list)
                            or inp[grid_key][i] is not None
                        )
                    ]
                    if len(idx_with_visual) > 0:
                        valid_pv = [
                            inp[pv_key][i] if isinstance(inp[pv_key][i], torch.Tensor)
                            else torch.from_numpy(inp[pv_key][i])
                            for i in idx_with_visual
                        ]
                        inp[pv_key] = torch.cat(valid_pv, dim=0).to(inp['input_ids'].device)
                        if grid_key in inp and isinstance(inp[grid_key], list):
                            valid_grid = [
                                inp[grid_key][i] if isinstance(inp[grid_key][i], torch.Tensor)
                                else torch.from_numpy(inp[grid_key][i])
                                for i in idx_with_visual
                            ]
                            inp[grid_key] = torch.cat(valid_grid, dim=0).to(inp['input_ids'].device)
                    else:
                        inp[pv_key] = None
                        if grid_key in inp:
                            inp[grid_key] = None
            # Remove None-valued visual keys so model forward doesn't choke
            for key in ['pixel_values', 'image_grid_thw', 'pixel_values_videos', 'video_grid_thw']:
                if key in inp and inp[key] is None:
                    del inp[key]

        _consolidate_visual_inputs(inputs)

        try:
            # Forward pass with gradient computation
            model.train()  # Enable gradient computation
            outputs = model.encoder(**inputs, return_dict=True, output_hidden_states=True)
            # Use the last hidden state's last token as a proxy for contrastive loss
            hidden = outputs.hidden_states[-1]
            # Simple proxy loss: sum of last-token hidden states (to generate gradients)
            if 'attention_mask' in inputs:
                mask = inputs['attention_mask']
                seq_lens = mask.sum(dim=1) - 1
                batch_size = hidden.shape[0]
                last_token_hidden = hidden[torch.arange(batch_size, device=device), seq_lens.long()]
            else:
                last_token_hidden = hidden[:, -1, :]
            
            # Proxy loss: L2 norm to generate meaningful gradients
            proxy_loss = last_token_hidden.norm(dim=-1).mean()
            proxy_loss.backward()

            num_processed += 1
            if (batch_idx + 1) % 8 == 0:
                logger.info(f"  Processed batch {batch_idx + 1}/{args.num_calibration_batches}")

        except Exception as e:
            logger.warning(f"  Skipping batch {batch_idx} due to error: {e}")
            continue
        finally:
            model.zero_grad()

    model.eval()
    head_accumulator.remove_hooks()
    mlp_accumulator.remove_hooks()

    if num_processed == 0:
        logger.error("No batches were successfully processed. Cannot compute importance.")
        return

    logger.info(f"Successfully processed {num_processed} batches")

    # ---- Finalize head importance ----
    head_importance = head_accumulator.finalize()
    print_importance_table(head_importance)

    # Save head importance JSON
    json_path = os.path.join(args.output_dir, f'{args.delete_L}_{args.delete_n}_head_importance.json')
    head_importance_dict = {}
    for l in range(head_importance.shape[0]):
        head_importance_dict[str(l)] = {}
        for g in range(head_importance.shape[1]):
            head_importance_dict[str(l)][str(g)] = float(head_importance[l, g])
    with open(json_path, 'w') as f:
        json.dump(head_importance_dict, f, indent=2)
    logger.info(f"Head importance scores saved to {json_path}")

    # Save head heatmap & ranking
    plot_heatmap(head_importance, os.path.join(args.output_dir, 'head_importance_heatmap.png'))
    plot_global_ranking(head_importance, os.path.join(args.output_dir, 'head_importance_ranking.png'))

    # Print least important KV groups
    flat = [(l, g, head_importance[l, g])
            for l in range(head_importance.shape[0])
            for g in range(head_importance.shape[1])]
    flat.sort(key=lambda x: x[2])
    print("\n--- Least important KV groups (candidates for pruning) ---")
    for rank, (l, g, score) in enumerate(flat[:20]):
        print(f"  #{rank+1:>3d}  Layer {l}, Group {g}  (score={score:.6f})")
    print()

    # ---- Finalize MLP importance ----
    mlp_importance = mlp_accumulator.finalize()
    print_mlp_importance_summary(mlp_importance)

    # Save MLP importance JSON: {layer_idx: [neuron_0_score, ..., neuron_N_score]}
    mlp_json_path = os.path.join(args.output_dir, 'mlp_importance.json')
    mlp_importance_dict = {}
    for l in range(mlp_importance.shape[0]):
        mlp_importance_dict[str(l)] = [float(v) for v in mlp_importance[l]]
    with open(mlp_json_path, 'w') as f:
        json.dump(mlp_importance_dict, f, indent=2)
    logger.info(f"MLP importance scores saved to {mlp_json_path}")

    # Save MLP heatmap (binned) & boxplot
    plot_mlp_heatmap(mlp_importance, os.path.join(args.output_dir, 'mlp_importance_heatmap.png'))
    plot_mlp_boxplot(mlp_importance, os.path.join(args.output_dir, 'mlp_importance_boxplot.png'))


if __name__ == '__main__':
    main()
