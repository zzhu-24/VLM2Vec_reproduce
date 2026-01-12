# 模型并行训练脚本 - 专门针对 Qwen3-VL-4B-Instruct
# 实现模型分块（模型并行），将模型层分配到不同的GPU上
import logging
import os.path
import sys
import json
import time

import torch
torch.autograd.set_detect_anomaly(True)
from numpy import ndarray
torch.serialization.add_safe_globals([ndarray])

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Debug logging helper
DEBUG_LOG_PATH = "/home/infres/zzhu-24/PRIM/VLM2Vec/.cursor/debug.log"
def debug_log(location, message, data, hypothesis_id=None):
    try:
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000)
        }
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

import sys
import torch
import torch.nn as nn
import wandb
import yaml
from transformers import HfArgumentParser
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.data.collator.train_collator import MultimodalDataCollator
from src.data.loader.mixed_dataset import init_mixed_dataset
from src.model.model import MMEBModel
from src.trainer import GradCacheLateProcessTrainer
from src.utils import print_rank, print_master, find_latest_checkpoint
from src.model.processor import load_processor, get_backbone_name, QWEN3_VL


def find_transformer_layers(model):
    """
    找到 Qwen3-VL 模型的 transformer 层位置
    Qwen3-VL 结构: model.language_model.layers 或 encoder.model.language_model.layers
    """
    encoder = model.encoder
    
    # 处理LoRA包装的模型
    if hasattr(encoder, 'base_model'):
        encoder = encoder.base_model
    
    layers = None
    layer_path = None
    
    # Qwen3-VL: model.language_model.layers
    if hasattr(encoder, 'model') and hasattr(encoder.model, 'language_model'):
        if hasattr(encoder.model.language_model, 'layers'):
            layers = encoder.model.language_model.layers
            layer_path = 'model.language_model.layers'
    elif hasattr(encoder, 'language_model'):
        if hasattr(encoder.language_model, 'layers'):
            layers = encoder.language_model.layers
            layer_path = 'language_model.layers'
    
    if layers is None:
        raise ValueError("无法找到 Qwen3-VL 的 transformer 层。请检查模型结构。")
    
    return layers, layer_path


def distribute_model_layers(model, world_size):
    """
    将模型的 transformer 层分配到不同的 GPU 上
    每个 GPU 负责一部分层（模型并行）
    """
    layers, layer_path = find_transformer_layers(model)
    num_layers = len(layers)
    
    print_master(f"找到 {num_layers} 个 transformer 层，路径: {layer_path}")
    print_master(f"使用 {world_size} 个 GPU 进行模型并行（模型分卡）")
    
    # 计算每个 GPU 负责的层范围
    layers_per_gpu = num_layers // world_size
    remainder = num_layers % world_size
    
    # 分配层到不同的 GPU
    start_idx = 0
    for rank in range(world_size):
        # 前 remainder 个 GPU 多分配一层
        end_idx = start_idx + layers_per_gpu + (1 if rank < remainder else 0)
        
        print_master(f"GPU {rank}: 层 {start_idx} 到 {end_idx-1} (共 {end_idx-start_idx} 层)")
        
        # 将层移动到对应的 GPU
        for i in range(start_idx, end_idx):
            target_device = f'cuda:{rank}'
            # #region agent log
            debug_log(
                f"train_qwen3vl_mp.py:{87}",
                f"开始移动层 {i} 到 {target_device}",
                {
                    "layer_idx": i,
                    "target_device": target_device,
                    "initial_device": str(next(layers[i].parameters()).device) if list(layers[i].parameters()) else "unknown"
                },
                hypothesis_id="H1"
            )
            # #endregion
            # 使用 .to() 移动整个模块
            layers[i] = layers[i].to(target_device)
            # 递归确保所有子模块的参数和缓冲区都在目标设备上
            # 这很重要，因为某些子模块（如 input_layernorm）可能在 .to() 后仍然在错误的设备上
            params_moved = 0
            buffers_moved = 0
            for name, module in layers[i].named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    if param.device != torch.device(target_device):
                        param.data = param.data.to(target_device)
                        params_moved += 1
                for buffer_name, buffer in module.named_buffers(recurse=False):
                    if buffer.device != torch.device(target_device):
                        buffer.data = buffer.data.to(target_device)
                        buffers_moved += 1
            # #region agent log
            debug_log(
                f"train_qwen3vl_mp.py:{100}",
                f"层 {i} 移动完成",
                {
                    "layer_idx": i,
                    "target_device": target_device,
                    "final_device": str(next(layers[i].parameters()).device) if list(layers[i].parameters()) else "unknown",
                    "params_moved": params_moved,
                    "buffers_moved": buffers_moved
                },
                hypothesis_id="H1"
            )
            # #endregion
            torch.cuda.empty_cache()
        
        start_idx = end_idx
    
    # 移动 embedding 层到 GPU 0
    encoder = model.encoder
    if hasattr(encoder, 'base_model'):
        encoder = encoder.base_model
    
    # 移动 visual encoder 到 GPU 0（如果存在，Qwen3-VL 的 visual encoder 只在第一层需要）
    if hasattr(encoder, 'model') and hasattr(encoder.model, 'visual'):
        # #region agent log
        debug_log(
            f"train_qwen3vl_mp.py:{159}",
            "找到 visual encoder，移动到 GPU 0",
            {
                "visual_device_before": str(next(encoder.model.visual.parameters()).device) if list(encoder.model.visual.parameters()) else "unknown"
            },
            hypothesis_id="H10"
        )
        # #endregion
        encoder.model.visual = encoder.model.visual.to('cuda:0')
        # 确保所有参数都在 GPU 0
        for param in encoder.model.visual.parameters():
            if param.device != torch.device('cuda:0'):
                param.data = param.data.to('cuda:0')
        for buffer in encoder.model.visual.buffers():
            if buffer.device != torch.device('cuda:0'):
                buffer.data = buffer.data.to('cuda:0')
        # #region agent log
        debug_log(
            f"train_qwen3vl_mp.py:{159}",
            "visual encoder 移动完成",
            {
                "visual_device_after": str(next(encoder.model.visual.parameters()).device) if list(encoder.model.visual.parameters()) else "unknown"
            },
            hypothesis_id="H10"
        )
        # #endregion
    elif hasattr(encoder, 'visual'):
        # #region agent log
        debug_log(
            f"train_qwen3vl_mp.py:{162}",
            "找到 visual encoder（直接路径），移动到 GPU 0",
            {
                "visual_device_before": str(next(encoder.visual.parameters()).device) if list(encoder.visual.parameters()) else "unknown"
            },
            hypothesis_id="H10"
        )
        # #endregion
        encoder.visual = encoder.visual.to('cuda:0')
        for param in encoder.visual.parameters():
            if param.device != torch.device('cuda:0'):
                param.data = param.data.to('cuda:0')
        for buffer in encoder.visual.buffers():
            if buffer.device != torch.device('cuda:0'):
                buffer.data = buffer.data.to('cuda:0')
        # #region agent log
        debug_log(
            f"train_qwen3vl_mp.py:{162}",
            "visual encoder 移动完成（直接路径）",
            {
                "visual_device_after": str(next(encoder.visual.parameters()).device) if list(encoder.visual.parameters()) else "unknown"
            },
            hypothesis_id="H10"
        )
        # #endregion
    
    if hasattr(encoder, 'model') and hasattr(encoder.model, 'language_model'):
        if hasattr(encoder.model.language_model, 'embed_tokens'):
            encoder.model.language_model.embed_tokens = encoder.model.language_model.embed_tokens.to('cuda:0')
    elif hasattr(encoder, 'language_model') and hasattr(encoder.language_model, 'embed_tokens'):
        encoder.language_model.embed_tokens = encoder.language_model.embed_tokens.to('cuda:0')
    
    # 移动 norm 层到最后一个 GPU
    # #region agent log
    debug_log(
        f"train_qwen3vl_mp.py:{115}",
        "开始移动 norm 层",
        {
            "world_size": world_size,
            "target_device": f'cuda:{world_size-1}'
        },
        hypothesis_id="H3"
    )
    # #endregion
    if hasattr(encoder, 'model') and hasattr(encoder.model, 'language_model'):
        if hasattr(encoder.model.language_model, 'norm'):
            norm_layer = encoder.model.language_model.norm
            # #region agent log
            debug_log(
                f"train_qwen3vl_mp.py:{119}",
                "找到 norm 层（路径1）",
                {
                    "norm_weight_device_before": str(next(norm_layer.parameters()).device) if list(norm_layer.parameters()) else "unknown"
                },
                hypothesis_id="H3"
            )
            # #endregion
            norm_layer = norm_layer.to(f'cuda:{world_size-1}')
            params_moved = 0
            buffers_moved = 0
            for param in norm_layer.parameters():
                if param.device != torch.device(f'cuda:{world_size-1}'):
                    param.data = param.data.to(f'cuda:{world_size-1}')
                    params_moved += 1
            for buffer in norm_layer.buffers():
                if buffer.device != torch.device(f'cuda:{world_size-1}'):
                    buffer.data = buffer.data.to(f'cuda:{world_size-1}')
                    buffers_moved += 1
            # #region agent log
            debug_log(
                f"train_qwen3vl_mp.py:{126}",
                "norm 层移动完成（路径1）",
                {
                    "norm_weight_device_after": str(next(norm_layer.parameters()).device) if list(norm_layer.parameters()) else "unknown",
                    "params_moved": params_moved,
                    "buffers_moved": buffers_moved
                },
                hypothesis_id="H3"
            )
            # #endregion
            encoder.model.language_model.norm = norm_layer
    elif hasattr(encoder, 'language_model') and hasattr(encoder.language_model, 'norm'):
        norm_layer = encoder.language_model.norm
        # #region agent log
        debug_log(
            f"train_qwen3vl_mp.py:{128}",
            "找到 norm 层（路径2）",
            {
                "norm_weight_device_before": str(next(norm_layer.parameters()).device) if list(norm_layer.parameters()) else "unknown"
            },
            hypothesis_id="H3"
        )
        # #endregion
        norm_layer = norm_layer.to(f'cuda:{world_size-1}')
        params_moved = 0
        buffers_moved = 0
        for param in norm_layer.parameters():
            if param.device != torch.device(f'cuda:{world_size-1}'):
                param.data = param.data.to(f'cuda:{world_size-1}')
                params_moved += 1
        for buffer in norm_layer.buffers():
            if buffer.device != torch.device(f'cuda:{world_size-1}'):
                buffer.data = buffer.data.to(f'cuda:{world_size-1}')
                buffers_moved += 1
        # #region agent log
        debug_log(
            f"train_qwen3vl_mp.py:{136}",
            "norm 层移动完成（路径2）",
            {
                "norm_weight_device_after": str(next(norm_layer.parameters()).device) if list(norm_layer.parameters()) else "unknown",
                "params_moved": params_moved,
                "buffers_moved": buffers_moved
            },
            hypothesis_id="H3"
        )
        # #endregion
        encoder.language_model.norm = norm_layer
    
    # 存储层分配信息，用于前向传播
    model._mp_layer_ranges = []
    start_idx = 0
    for rank in range(world_size):
        end_idx = start_idx + layers_per_gpu + (1 if rank < remainder else 0)
        model._mp_layer_ranges.append((start_idx, end_idx, rank))
        start_idx = end_idx
    
    model._mp_world_size = world_size
    model._mp_layer_path = layer_path
    model._mp_layers = layers
    
    print_master("模型层分配完成")
    return model


class ModelParallelMMEBModel(MMEBModel):
    """
    支持模型并行的 MMEBModel（专门针对 Qwen3-VL）
    重写 encode_input 方法以支持跨设备的前向传播
    """
    
    @property
    def device(self):
        """对于模型并行，返回第一个 GPU（embedding 层所在的设备）"""
        return torch.device('cuda:0')
    
    def encode_input(self, input, selected_layers):
        """
        重写 encode_input 以支持模型并行
        数据在 GPU 间传递，每层在对应的 GPU 上计算
        """
        from transformers.modeling_outputs import BaseModelOutputWithPast
        
        model_backbone = get_backbone_name(hf_config=self.config)
        
        # 处理 Qwen3-VL 的特殊输入格式
        if model_backbone == QWEN3_VL:
            if 'pixel_values' in input and isinstance(input['pixel_values'], list):
                bsz = input['input_ids'].shape[0]
                if 'image_grid_thw' in input and isinstance(input['image_grid_thw'], list):
                    idx_w_image = [i for i in range(bsz) if input['pixel_values'][i] is not None and 
                                  input['image_grid_thw'][i] is not None]
                else:
                    idx_w_image = [i for i in range(bsz) if input['pixel_values'][i] is not None]

                if len(idx_w_image) > 0:
                    valid_pixel_values = [
                        input['pixel_values'][i] if isinstance(input['pixel_values'][i], torch.Tensor) 
                        else torch.from_numpy(input['pixel_values'][i]) 
                        for i in idx_w_image
                    ]
                    input['pixel_values'] = torch.cat(valid_pixel_values, dim=0).to(input['input_ids'].device)
                    
                    if 'image_grid_thw' in input and isinstance(input['image_grid_thw'], list):
                        valid_grid_thw = [
                            input['image_grid_thw'][i] if isinstance(input['image_grid_thw'][i], torch.Tensor) 
                            else torch.from_numpy(input['image_grid_thw'][i]) 
                            for i in idx_w_image
                        ]
                        input['image_grid_thw'] = torch.cat(valid_grid_thw, dim=0).to(input['input_ids'].device)
                else:
                    input['pixel_values'] = None
                    if 'image_grid_thw' in input:
                        input['image_grid_thw'] = None

            if 'pixel_values_videos' in input and isinstance(input['pixel_values_videos'], list):
                bsz = input['input_ids'].shape[0]
                if 'video_grid_thw' in input and isinstance(input['video_grid_thw'], list):
                    idx_w_video = [i for i in range(bsz) if input['pixel_values_videos'][i] is not None and 
                                  input['video_grid_thw'][i] is not None]
                else:
                    idx_w_video = [i for i in range(bsz) if input['pixel_values_videos'][i] is not None]
                
                if len(idx_w_video) > 0:
                    valid_pixel_values_videos = [
                        input['pixel_values_videos'][i] if isinstance(input['pixel_values_videos'][i], torch.Tensor) 
                        else torch.from_numpy(input['pixel_values_videos'][i]) 
                        for i in idx_w_video
                    ]
                    input['pixel_values_videos'] = torch.cat(valid_pixel_values_videos, dim=0).to(input['input_ids'].device)
                    
                    if 'video_grid_thw' in input:
                        valid_video_grid_thw = [
                            input['video_grid_thw'][i] if isinstance(input['video_grid_thw'][i], torch.Tensor) 
                            else torch.from_numpy(input['video_grid_thw'][i]) 
                            for i in idx_w_video
                        ]
                        input['video_grid_thw'] = torch.cat(valid_video_grid_thw, dim=0).to(input['input_ids'].device)
                else:
                    input['pixel_values_videos'] = None
                    if 'video_grid_thw' in input:
                        input['video_grid_thw'] = None
        
        # 模型并行前向传播
        device = 'cuda:0'
        # #region agent log
        debug_log(
            f"train_qwen3vl_mp.py:{345}",
            "encode_input 开始，记录初始内存状态",
            {
                "gpu_memory_initial": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {},
                "gpu_memory_reserved": {f"cuda:{gpu}": torch.cuda.memory_reserved(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {},
                "input_keys": list(input.keys()),
                "pixel_values_shape": list(input.get('pixel_values', 'N/A').shape) if isinstance(input.get('pixel_values'), torch.Tensor) else str(type(input.get('pixel_values'))),
                "pixel_values_size_mb": input.get('pixel_values', 'N/A').numel() * input.get('pixel_values', 'N/A').element_size() / 1024**2 if isinstance(input.get('pixel_values'), torch.Tensor) else 0
            },
            hypothesis_id="H6"
        )
        # #endregion
        input_on_device = {}
        for k, v in input.items():
            if isinstance(v, torch.Tensor):
                input_on_device[k] = v.to(device)
            elif isinstance(v, list):
                input_on_device[k] = v
            else:
                input_on_device[k] = v
        
        # 获取 encoder 和 layers
        encoder = self.encoder
        if hasattr(encoder, 'base_model'):
            encoder = encoder.base_model
        
        # 获取 embedding 层
        if hasattr(encoder, 'model') and hasattr(encoder.model, 'language_model'):
            embed_tokens = encoder.model.language_model.embed_tokens
            model_part = encoder.model.language_model
        elif hasattr(encoder, 'language_model'):
            embed_tokens = encoder.language_model.embed_tokens
            model_part = encoder.language_model
        else:
            raise ValueError("无法找到 embedding 层")
        
        # 获取 layers
        if hasattr(self, '_mp_layers'):
            layers = self._mp_layers
        else:
            layers, _ = find_transformer_layers(self)
        
        # 获取输入
        input_ids = input_on_device.get('input_ids')
        if input_ids is None:
            raise ValueError("需要 input_ids 进行前向传播")
        
        attention_mask = input_on_device.get('attention_mask')
        position_ids = input_on_device.get('position_ids')
        
        # 处理 position_ids
        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=device).unsqueeze(0).expand_as(input_ids)
        
        # Qwen3-VL 需要 3D position_ids
        if position_ids.dim() == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        
        # Embedding
        hidden_states = embed_tokens(input_ids)
        
        # 计算 position_embeddings（Qwen3-VL 必需）
        rotary_emb = None
        if hasattr(encoder, 'model') and hasattr(encoder.model, 'language_model'):
            lang_model = encoder.model.language_model
            if hasattr(lang_model, 'rotary_emb'):
                rotary_emb = lang_model.rotary_emb
        elif hasattr(encoder, 'language_model'):
            lang_model = encoder.language_model
            if hasattr(lang_model, 'rotary_emb'):
                rotary_emb = lang_model.rotary_emb
        
        if rotary_emb is None:
            if hasattr(model_part, 'rotary_emb'):
                rotary_emb = model_part.rotary_emb
        
        position_embeddings = None
        if rotary_emb is not None:
            try:
                position_embeddings = rotary_emb(hidden_states, position_ids)
            except Exception as e:
                logger.error(f"计算 position_embeddings 失败: {e}")
                raise RuntimeError(f"无法计算 position_embeddings: {e}")
        else:
            raise RuntimeError("未找到 rotary_emb，无法计算 position_embeddings")
        
        # 存储所有层的 hidden states
        # 注意：需要保存所有层，因为 selected_layers 可能包含负数索引（如 -1）
        # 但可以在不需要时及时清理
        all_hidden_states = [hidden_states]
        
        # #region agent log
        debug_log(
            f"train_qwen3vl_mp.py:{425}",
            "开始逐层前向传播",
            {
                "num_layers": len(layers),
                "selected_layers": selected_layers,
                "save_all_hidden_states": False,
                "gpu_memory_before_layers": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
            },
            hypothesis_id="H7"
        )
        # #endregion
        
        # 逐层前向传播，在对应的 GPU 上计算
        for layer_idx, layer in enumerate(layers):
            # 确定这一层应该在哪个 GPU 上
            target_device = None
            for start_idx, end_idx, gpu_rank in self._mp_layer_ranges:
                if start_idx <= layer_idx < end_idx:
                    target_device = gpu_rank
                    break
            
            if target_device is None:
                raise ValueError(f"无法确定层 {layer_idx} 应该在哪个 GPU 上")
            
            # 移动 hidden_states 到目标 GPU
            target_device_str = f'cuda:{target_device}'
            hidden_states = hidden_states.to(target_device_str)
            
            # 确保 layer 及其所有子模块都在目标设备上（重要！）
            # 这可以防止 input_layernorm 等子模块的参数在错误的设备上
            layer_device = next(layer.parameters()).device if list(layer.parameters()) else torch.device(target_device_str)
            # #region agent log
            debug_log(
                f"train_qwen3vl_mp.py:{333}",
                f"检查层 {layer_idx} 的设备",
                {
                    "layer_idx": layer_idx,
                    "layer_device": str(layer_device),
                    "target_device": target_device_str,
                    "match": layer_device == torch.device(target_device_str)
                },
                hypothesis_id="H2"
            )
            # #endregion
            if layer_device != torch.device(target_device_str):
                logger.warning(f"层 {layer_idx} 的设备 ({layer_device}) 与目标设备 ({target_device_str}) 不匹配，正在移动...")
                # #region agent log
                debug_log(
                    f"train_qwen3vl_mp.py:{336}",
                    f"层 {layer_idx} 设备不匹配，开始移动",
                    {
                        "layer_idx": layer_idx,
                        "from_device": str(layer_device),
                        "to_device": target_device_str
                    },
                    hypothesis_id="H2"
                )
                # #endregion
                layer = layer.to(target_device_str)
                # 确保所有参数和缓冲区都在目标设备上
                params_moved = 0
                buffers_moved = 0
                for param in layer.parameters():
                    if param.device != torch.device(target_device_str):
                        param.data = param.data.to(target_device_str)
                        params_moved += 1
                for buffer in layer.buffers():
                    if buffer.device != torch.device(target_device_str):
                        buffer.data = buffer.data.to(target_device_str)
                        buffers_moved += 1
                # 更新 layers 列表中的引用
                layers[layer_idx] = layer
                # #region agent log
                debug_log(
                    f"train_qwen3vl_mp.py:{345}",
                    f"层 {layer_idx} 移动完成",
                    {
                        "layer_idx": layer_idx,
                        "final_device": str(next(layer.parameters()).device) if list(layer.parameters()) else "unknown",
                        "params_moved": params_moved,
                        "buffers_moved": buffers_moved
                    },
                    hypothesis_id="H2"
                )
                # #endregion
            
            # 准备 layer 的输入
            layer_kwargs = {
                'hidden_states': hidden_states,
                'attention_mask': attention_mask.to(target_device_str) if attention_mask is not None else None,
                'position_ids': position_ids.to(target_device_str),
                'output_hidden_states': True,
            }
            
            # 添加 position_embeddings（优化：只在需要时移动，避免重复复制）
            if position_embeddings is not None:
                if isinstance(position_embeddings, tuple):
                    # 检查是否已经在目标设备上
                    if position_embeddings[0].device == torch.device(target_device_str):
                        layer_kwargs['position_embeddings'] = position_embeddings
                    else:
                        layer_kwargs['position_embeddings'] = (
                            position_embeddings[0].to(target_device_str),
                            position_embeddings[1].to(target_device_str)
                        )
                else:
                    if position_embeddings.device == torch.device(target_device_str):
                        layer_kwargs['position_embeddings'] = position_embeddings
                    else:
                        layer_kwargs['position_embeddings'] = position_embeddings.to(target_device_str)
            
            # 添加视觉相关的输入（优化：只在第一层传递，后续层不需要）
            # Qwen3-VL 的视觉编码只在第一层处理，后续层不需要 pixel_values
            if layer_idx == 0:
                for k in ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw']:
                    if k in input_on_device and input_on_device[k] is not None:
                        if isinstance(input_on_device[k], torch.Tensor):
                            if input_on_device[k].device != torch.device(target_device_str):
                                layer_kwargs[k] = input_on_device[k].to(target_device_str)
                            else:
                                layer_kwargs[k] = input_on_device[k]
                        else:
                            layer_kwargs[k] = input_on_device[k]
                # #region agent log
                debug_log(
                    f"train_qwen3vl_mp.py:{616}",
                    "第一层添加视觉输入",
                    {
                        "layer_idx": layer_idx,
                        "has_pixel_values": 'pixel_values' in layer_kwargs and layer_kwargs['pixel_values'] is not None,
                        "pixel_values_size_mb": layer_kwargs.get('pixel_values', 'N/A').numel() * layer_kwargs.get('pixel_values', 'N/A').element_size() / 1024**2 if isinstance(layer_kwargs.get('pixel_values'), torch.Tensor) else 0,
                        "gpu_memory": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                    },
                    hypothesis_id="H6"
                )
                # #endregion
            
            # 调用 layer
            try:
                layer_outputs = layer(**layer_kwargs)
                
                # 处理输出
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                elif hasattr(layer_outputs, 'hidden_states'):
                    hidden_states = layer_outputs.hidden_states
                elif hasattr(layer_outputs, 'last_hidden_state'):
                    hidden_states = layer_outputs.last_hidden_state
                else:
                    hidden_states = layer_outputs
                
                # 清理中间结果
                del layer_outputs
                for k in ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw']:
                    if k in layer_kwargs:
                        del layer_kwargs[k]
                del layer_kwargs
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"层 {layer_idx} 前向传播失败: {e}")
                raise
            
            # 保存 hidden states（需要所有层，因为 selected_layers 可能包含负数索引）
            all_hidden_states.append(hidden_states)
            
            # #region agent log
            if layer_idx % 5 == 0 or layer_idx == len(layers) - 1:
                debug_log(
                    f"train_qwen3vl_mp.py:{557}",
                    f"层 {layer_idx} 前向传播完成",
                    {
                        "layer_idx": layer_idx,
                        "all_hidden_states_count": len(all_hidden_states),
                        "gpu_memory": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {},
                        "gpu_memory_reserved": {f"cuda:{gpu}": torch.cuda.memory_reserved(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                    },
                    hypothesis_id="H7"
                )
            # #endregion
        
        # 应用 norm 层（在最后一个 GPU 上）
        encoder = self.encoder
        if hasattr(encoder, 'base_model'):
            encoder = encoder.base_model
        
        norm = None
        if hasattr(encoder, 'model') and hasattr(encoder.model, 'language_model'):
            if hasattr(encoder.model.language_model, 'norm'):
                norm = encoder.model.language_model.norm
        elif hasattr(encoder, 'language_model') and hasattr(encoder.language_model, 'norm'):
            norm = encoder.language_model.norm
        
        if norm is not None:
            target_norm_device = f'cuda:{self._mp_world_size-1}'
            # #region agent log
            debug_log(
                f"train_qwen3vl_mp.py:{417}",
                "准备应用 norm 层",
                {
                    "target_norm_device": target_norm_device,
                    "hidden_states_device": str(hidden_states.device) if hasattr(hidden_states, 'device') else "unknown",
                    "norm_weight_device": str(next(norm.parameters()).device) if list(norm.parameters()) else "unknown"
                },
                hypothesis_id="H4"
            )
            # #endregion
            hidden_states = hidden_states.to(target_norm_device)
            # #region agent log
            debug_log(
                f"train_qwen3vl_mp.py:{419}",
                "hidden_states 已移动到 norm 设备",
                {
                    "hidden_states_device_after": str(hidden_states.device) if hasattr(hidden_states, 'device') else "unknown",
                    "norm_weight_device": str(next(norm.parameters()).device) if list(norm.parameters()) else "unknown"
                },
                hypothesis_id="H4"
            )
            # #endregion
            # 再次确保 norm 的所有参数都在目标设备上
            for param in norm.parameters():
                if param.device != torch.device(target_norm_device):
                    # #region agent log
                    debug_log(
                        f"train_qwen3vl_mp.py:{425}",
                        "发现 norm 参数在错误设备，正在移动",
                        {
                            "param_device": str(param.device),
                            "target_device": target_norm_device
                        },
                        hypothesis_id="H4"
                    )
                    # #endregion
                    param.data = param.data.to(target_norm_device)
            hidden_states = norm(hidden_states)
            all_hidden_states[-1] = hidden_states
        
        # 提取 selected_layers 的 hidden states 并 pooling
        all_outputs = []
        attention_mask_device = attention_mask.device if attention_mask is not None else 'cuda:0'
        
        for layer in selected_layers:
            if layer < 0:
                layer = len(all_hidden_states) + layer
            
            if layer >= len(all_hidden_states):
                logger.warning(f"层 {layer} 超出范围，使用最后一层")
                layer = len(all_hidden_states) - 1
            
            layer_hidden_states = all_hidden_states[layer].to(attention_mask_device)
            pooled_output = self._pooling(layer_hidden_states, attention_mask)
            all_outputs.append(pooled_output)
        
        # 清理大张量
        for k in ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw']:
            if k in input_on_device:
                del input_on_device[k]
        
        # 清理 position_embeddings（如果不再需要）
        del position_embeddings
        if 'position_embeddings' in locals():
            del position_embeddings
        
        # 强制同步并清空所有 GPU 的缓存
        if torch.cuda.is_available():
            for gpu in range(self._mp_world_size):
                torch.cuda.synchronize(f'cuda:{gpu}')
        torch.cuda.empty_cache()
        
        # #region agent log
        debug_log(
            f"train_qwen3vl_mp.py:{631}",
            "encode_input 结束，记录最终内存状态",
            {
                "gpu_memory_final": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {},
                "gpu_memory_reserved": {f"cuda:{gpu}": torch.cuda.memory_reserved(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {},
                "all_hidden_states_count": len(all_hidden_states)
            },
            hypothesis_id="H7"
        )
        # #endregion
        
        return all_outputs


def main():
    # 处理 torch.distributed.launch 的参数
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    print(f"Model args: {model_args}")
    print(f"Data args: {data_args}")
    print(f"Training args: {training_args}")

    # 获取 world_size 用于模型并行
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        world_size = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        world_size = torch.cuda.device_count()
    print_master(f"使用 {world_size} 个 GPU 进行模型并行（模型分卡）")

    # 检查 checkpoint
    if training_args.resume_from == 'auto':
        resume_checkpoint_dir = find_latest_checkpoint(training_args.output_dir)
        if resume_checkpoint_dir:
            logger.info(f"从 checkpoint 恢复: {resume_checkpoint_dir}")
    elif training_args.resume_from.isdigit():
        resume_checkpoint_dir = os.path.join(training_args.output_dir, f'checkpoint-{training_args.resume_from}')
        if os.path.exists(resume_checkpoint_dir):
            logger.info(f"从 checkpoint 恢复: {resume_checkpoint_dir}")
    else:
        resume_checkpoint_dir = None
        logger.info("未找到 checkpoint，从头开始训练")

    # 初始化 WandB
    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('初始化 wandb')
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online", settings=wandb.Settings(init_timeout=120))
            wandb.config.update(model_args)
            wandb.config.update(data_args)
            wandb.config.update(training_args)

    # 加载模型
    if model_args.checkpoint_path and model_args.lora:
        adapter_config_path = os.path.join(model_args.checkpoint_path, 'adapter_config.json')
        if os.path.exists(adapter_config_path):
            logger.info(f"找到 LoRA adapter，从 {model_args.checkpoint_path} 加载模型")
            base_model = MMEBModel.load(model_args, is_trainable=True)
        else:
            logger.info("未找到 LoRA adapter，构建新模型")
            base_model = MMEBModel.build(model_args)
    else:
        base_model = MMEBModel.build(model_args)
    
    # 将模型转换为模型并行版本
    print_master("开始模型并行分配...")
    model = ModelParallelMMEBModel(
        encoder=base_model.encoder,
        pooling=base_model.pooling,
        normalize=base_model.normalize,
        temperature=base_model.temperature
    )
    model.eval_layers = base_model.eval_layers
    model.joint_training_layers = base_model.joint_training_layers
    
    # 分配模型层到不同的 GPU
    model = distribute_model_layers(model, world_size)
    
    # #region agent log
    # 验证层分配后的设备状态
    if hasattr(model, '_mp_layers'):
        for layer_idx, layer in enumerate(model._mp_layers):
            if list(layer.parameters()):
                layer_device = next(layer.parameters()).device
                # 确定这一层应该在哪个 GPU 上
                target_device = None
                for start_idx, end_idx, gpu_rank in model._mp_layer_ranges:
                    if start_idx <= layer_idx < end_idx:
                        target_device = gpu_rank
                        break
                if target_device is not None:
                    debug_log(
                        f"train_qwen3vl_mp.py:{570}",
                        f"验证层 {layer_idx} 的设备（分配后）",
                        {
                            "layer_idx": layer_idx,
                            "layer_device": str(layer_device),
                            "expected_device": f'cuda:{target_device}',
                            "match": str(layer_device) == f'cuda:{target_device}'
                        },
                        hypothesis_id="H2"
                    )
    # #endregion
    
    model_backbone = get_backbone_name(hf_config=model.config)
    setattr(model_args, 'model_backbone', model_backbone)
    setattr(training_args, 'model_backbone', model_backbone)
    print_rank(f'model_backbone: {model_backbone}')
    
    processor = load_processor(model_args, data_args)
    setattr(model, 'processor', processor)

    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_config = yaml.safe_load(yaml_file)
        train_dataset = init_mixed_dataset(dataset_config, model_args, data_args, training_args)
    train_collator = MultimodalDataCollator(processor, model_args, data_args, training_args)

    # 模型并行训练不使用 DDP 和 DataParallel
    original_local_rank = training_args.local_rank
    training_args.local_rank = -1
    
    trainer_cls = GradCacheLateProcessTrainer
    trainer = trainer_cls(
        model=model,
        processing_class=processor,
        args=training_args,
        model_args=model_args,
        train_dataset=train_dataset,
        data_collator=train_collator,
        max_length=data_args.max_len,
    )
    train_dataset.trainer = trainer
    
    # #region agent log
    # 验证 trainer 初始化后层的设备状态，并修复任何不匹配
    if hasattr(model, '_mp_layers'):
        for layer_idx, layer in enumerate(model._mp_layers):
            if list(layer.parameters()):
                layer_device = next(layer.parameters()).device
                # 确定这一层应该在哪个 GPU 上
                target_device = None
                for start_idx, end_idx, gpu_rank in model._mp_layer_ranges:
                    if start_idx <= layer_idx < end_idx:
                        target_device = gpu_rank
                        break
                if target_device is not None:
                    target_device_str = f'cuda:{target_device}'
                    if str(layer_device) != target_device_str:
                        # #region agent log
                        debug_log(
                            f"train_qwen3vl_mp.py:{595}",
                            f"检测到层 {layer_idx} 设备不匹配（trainer 初始化后），正在修复",
                            {
                                "layer_idx": layer_idx,
                                "layer_device": str(layer_device),
                                "expected_device": target_device_str
                            },
                            hypothesis_id="H2"
                        )
                        # #endregion
                        # 修复：将层移回正确的设备
                        layer = layer.to(target_device_str)
                        # 确保所有参数和缓冲区都在目标设备上
                        for param in layer.parameters():
                            if param.device != torch.device(target_device_str):
                                param.data = param.data.to(target_device_str)
                        for buffer in layer.buffers():
                            if buffer.device != torch.device(target_device_str):
                                buffer.data = buffer.data.to(target_device_str)
                        # 更新 layers 列表中的引用
                        model._mp_layers[layer_idx] = layer
                        # #region agent log
                        debug_log(
                            f"train_qwen3vl_mp.py:{620}",
                            f"层 {layer_idx} 设备修复完成",
                            {
                                "layer_idx": layer_idx,
                                "final_device": str(next(layer.parameters()).device) if list(layer.parameters()) else "unknown"
                            },
                            hypothesis_id="H2"
                        )
                        # #endregion
    # #endregion
    
    # 修复 norm 层的设备（trainer 初始化后可能被移动）
    encoder = model.encoder
    if hasattr(encoder, 'base_model'):
        encoder = encoder.base_model
    
    norm = None
    if hasattr(encoder, 'model') and hasattr(encoder.model, 'language_model'):
        if hasattr(encoder.model.language_model, 'norm'):
            norm = encoder.model.language_model.norm
    elif hasattr(encoder, 'language_model') and hasattr(encoder.language_model, 'norm'):
        norm = encoder.language_model.norm
    
    if norm is not None:
        target_norm_device = f'cuda:{model._mp_world_size-1}'
        norm_device = next(norm.parameters()).device if list(norm.parameters()) else None
        if norm_device != torch.device(target_norm_device):
            # #region agent log
            debug_log(
                f"train_qwen3vl_mp.py:{640}",
                "检测到 norm 层设备不匹配（trainer 初始化后），正在修复",
                {
                    "norm_device": str(norm_device),
                    "target_device": target_norm_device
                },
                hypothesis_id="H3"
            )
            # #endregion
            norm = norm.to(target_norm_device)
            for param in norm.parameters():
                if param.device != torch.device(target_norm_device):
                    param.data = param.data.to(target_norm_device)
            for buffer in norm.buffers():
                if buffer.device != torch.device(target_norm_device):
                    buffer.data = buffer.data.to(target_norm_device)
            # 更新 norm 层引用
            if hasattr(encoder, 'model') and hasattr(encoder.model, 'language_model'):
                encoder.model.language_model.norm = norm
            elif hasattr(encoder, 'language_model'):
                encoder.language_model.norm = norm
            # #region agent log
            debug_log(
                f"train_qwen3vl_mp.py:{655}",
                "norm 层设备修复完成",
                {
                    "final_device": str(next(norm.parameters()).device) if list(norm.parameters()) else "unknown"
                },
                hypothesis_id="H3"
            )
            # #endregion
    
    # 重写 _wrap_model 方法以禁用 DataParallel
    original_wrap_model = trainer._wrap_model
    def _wrap_model_no_dp(self, model, training=True, dataloader=None):
        return model
    trainer._wrap_model = _wrap_model_no_dp.__get__(trainer, type(trainer))

    print_master("开始模型并行训练...")
    print_master(f"模型层分配：每个 GPU 负责一部分层")
    print_master(f"已禁用 DataParallel，使用模型并行（模型分卡）")
    trainer.train(resume_from_checkpoint=resume_checkpoint_dir)
    
    # 恢复原始设置
    training_args.local_rank = original_local_rank
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
