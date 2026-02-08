#!/usr/bin/env python3
"""
临时脚本：计算给定模型的参数数量。
用法（在项目根目录下）:
  python adhoc/count_model_params.py --path <模型或 checkpoint 路径>
  python adhoc/count_model_params.py --path <LoRA adapter 目录> --model_name <基座模型名或路径>

若 path 下存在 adapter_config.json，会自动按 LoRA 加载并需提供 --model_name（或 adapter_config 中含 base_model）。
"""
import argparse
import json
import os
import sys

# 保证从项目根运行时能导入 src
if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

import torch
from peft import PeftModel

from src.arguments import ModelArguments
from src.model.model import MMEBModel


def get_param_count(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if not trainable_only or p.requires_grad)


def get_base_param_count(model):
    """纯基座参数（不含 LoRA）。"""
    unwrapped = model.module if hasattr(model, "module") else model
    if not hasattr(unwrapped, "encoder"):
        return None
    encoder = unwrapped.encoder
    if isinstance(encoder, PeftModel):
        base = encoder.get_base_model()
        return sum(p.numel() for p in base.parameters())
    return sum(p.numel() for p in encoder.parameters())


def get_detailed_param_breakdown(model):
    """
    按图像编码、merger、LLM 细分参数。
    返回 dict: image_encoder, merger, llm, other, (以及 base 总与可选 lora_total)。
    若结构不支持则部分键为 None。
    """
    unwrapped = model.module if hasattr(model, "module") else model
    if not hasattr(unwrapped, "encoder"):
        return {}
    encoder = unwrapped.encoder
    is_peft = isinstance(encoder, PeftModel)
    base = encoder.get_base_model() if is_peft else encoder

    out = {"image_encoder": None, "merger": None, "llm": None, "other": None}
    if not hasattr(base, "visual") or not hasattr(base, "model"):
        return out

    visual = base.visual
    llm_module = base.model

    # 图像编码：visual 中除 merger 外的所有参数（patch_embed + blocks + rotary 等）
    image_encoder_count = 0
    merger_count = 0
    for name, param in visual.named_parameters():
        n = param.numel()
        if "merger" in name:
            merger_count += n
        else:
            image_encoder_count += n
    out["image_encoder"] = image_encoder_count
    out["merger"] = merger_count

    # LLM 部分
    out["llm"] = sum(p.numel() for p in llm_module.parameters())

    # 其他（如 lm_head 等不在 model 里的）
    base_total = sum(p.numel() for p in base.parameters())
    out["other"] = base_total - (image_encoder_count + merger_count + out["llm"])
    if out["other"] < 0:
        out["other"] = 0

    if is_peft:
        total_with_lora = sum(p.numel() for p in encoder.parameters())
        out["lora_total"] = total_with_lora - base_total
    else:
        out["lora_total"] = 0

    return out


def main():
    parser = argparse.ArgumentParser(description="计算模型参数数量")
    parser.add_argument("--path", type=str, required=True, help="模型/checkpoint 路径或 HF 模型名")
    parser.add_argument("--model_name", type=str, default=None, help="基座模型名或路径（path 为 LoRA 目录时必填或由 adapter_config 提供）")
    parser.add_argument("--lora", action="store_true", help="强制将 path 视为 LoRA adapter 目录")
    parser.add_argument("--device", type=str, default="cpu", help="加载设备，默认 cpu 以省显存")
    args = parser.parse_args()

    path = args.path.rstrip("/")
    model_name = args.model_name
    use_lora = args.lora

    # 自动检测 LoRA：path 目录下存在 adapter_config.json
    adapter_config_path = os.path.join(path, "adapter_config.json")
    if os.path.isfile(adapter_config_path):
        use_lora = True
        if model_name is None:
            with open(adapter_config_path) as f:
                cfg = json.load(f)
            model_name = cfg.get("base_model_name_or_path") or cfg.get("base_model_name")
            if not model_name:
                print("Path 为 LoRA 目录但 adapter_config.json 中无 base_model，请指定 --model_name")
                sys.exit(1)

    if use_lora and not model_name:
        print("LoRA 模式需提供基座模型，请指定 --model_name")
        sys.exit(1)

    # 构建 ModelArguments
    model_args = ModelArguments(
        model_name=model_name if use_lora else path,
        checkpoint_path=path if use_lora else None,
        lora=use_lora,
    )

    print(f"加载模型: path={path}, lora={use_lora}, model_name={model_args.model_name}")
    with torch.device(args.device):
        model = MMEBModel.load(model_args, is_trainable=False)

    total = get_param_count(model, trainable_only=False)
    trainable = get_param_count(model, trainable_only=True)
    base_only = get_base_param_count(model)
    breakdown = get_detailed_param_breakdown(model)

    print("\n--- 参数统计 ---")
    print(f"  总参数:           {total:,}")
    print(f"  可训练参数:       {trainable:,}")
    if base_only is not None:
        print(f"  基座参数(不含LoRA): {base_only:,}")
        if use_lora:
            lora_params = total - base_only
            print(f"  LoRA 参数:        {lora_params:,}")

    if breakdown:
        print("\n--- 细分参数（图像编码 / merger / LLM）---")
        if breakdown.get("image_encoder") is not None:
            print(f"  图像编码 (ViT):   {breakdown['image_encoder']:,}")
        if breakdown.get("merger") is not None:
            print(f"  Merger:           {breakdown['merger']:,}")
        if breakdown.get("llm") is not None:
            print(f"  LLM:              {breakdown['llm']:,}")
        if breakdown.get("other") is not None and breakdown["other"] > 0:
            print(f"  其他:             {breakdown['other']:,}")
        if breakdown.get("lora_total") is not None and breakdown["lora_total"] > 0:
            print(f"  LoRA (额外):      {breakdown['lora_total']:,}")
        # 校验：基座细分之和
        if all(
            breakdown.get(k) is not None
            for k in ("image_encoder", "merger", "llm")
        ):
            base_sum = (
                breakdown["image_encoder"]
                + breakdown["merger"]
                + breakdown["llm"]
                + (breakdown.get("other") or 0)
            )
            print(f"  [基座细分合计]    {base_sum:,}")


if __name__ == "__main__":
    main()
