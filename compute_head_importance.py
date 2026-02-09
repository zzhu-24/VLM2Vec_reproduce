"""
Compute attention head importance scores for Qwen3-VL LLM decoder layers.

Uses Taylor importance (Michel et al., 2019): importance(l, g) = mean(||h_g * grad(h_g)||_1)
where h_g is the concatenated output of all Q heads in a KV group.

Outputs:
  - A JSON file with per-layer per-KV-group importance scores
  - A heatmap PNG visualization (layers x KV groups)
  - An optional global ranking bar chart
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
        self.num_heads = attn0.num_heads
        self.num_kv_heads = attn0.num_key_value_heads
        self.head_dim = attn0.head_dim
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
    logger.info(f"Found {num_layers} decoder layers after layer deletion")
    logger.info(f"  num_heads={attn0.num_heads}, num_kv_heads={attn0.num_key_value_heads}, "
                f"head_dim={attn0.head_dim}, q_per_group={attn0.num_heads // attn0.num_key_value_heads}")

    # ---- Setup importance accumulator ----
    accumulator = HeadImportanceAccumulator(layers)

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

        # The collator returns a dict with 'qry' and 'tgt' keys
        qry = batch.get('qry')
        tgt = batch.get('tgt')

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
        for k in ['texts', 'images']:
            inputs.pop(k, None)

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
    accumulator.remove_hooks()

    if num_processed == 0:
        logger.error("No batches were successfully processed. Cannot compute importance.")
        return

    logger.info(f"Successfully processed {num_processed} batches")

    # ---- Finalize and output ----
    importance = accumulator.finalize()

    # Print text table
    print_importance_table(importance)

    # Save JSON
    json_path = os.path.join(args.output_dir, 'head_importance.json')
    importance_dict = {}
    for l in range(importance.shape[0]):
        importance_dict[str(l)] = {}
        for g in range(importance.shape[1]):
            importance_dict[str(l)][str(g)] = float(importance[l, g])
    with open(json_path, 'w') as f:
        json.dump(importance_dict, f, indent=2)
    logger.info(f"Importance scores saved to {json_path}")

    # Save heatmap
    heatmap_path = os.path.join(args.output_dir, 'head_importance_heatmap.png')
    plot_heatmap(importance, heatmap_path)

    # Save global ranking chart
    ranking_path = os.path.join(args.output_dir, 'head_importance_ranking.png')
    plot_global_ranking(importance, ranking_path)

    # Print summary: least important groups
    flat = [(l, g, importance[l, g]) for l in range(importance.shape[0]) for g in range(importance.shape[1])]
    flat.sort(key=lambda x: x[2])
    print("\n--- Least important KV groups (candidates for pruning) ---")
    for rank, (l, g, score) in enumerate(flat[:20]):
        print(f"  #{rank+1:>3d}  Layer {l}, Group {g}  (score={score:.6f})")
    print()


if __name__ == '__main__':
    main()
