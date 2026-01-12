# Adapted from train.py for Model Parallelism (模型分卡训练)
import logging
import os.path
import sys
import json

import torch
torch.autograd.set_detect_anomaly(True)
from numpy import ndarray
torch.serialization.add_safe_globals([ndarray])

logging.basicConfig(
    level=logging.INFO, format='[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # Ensures logs appear in stdout
)
logger = logging.getLogger(__name__)

# Debug logging configuration
DEBUG_LOG_PATH = "/home/infres/zzhu-24/PRIM/VLM2Vec/.cursor/debug.log"

def debug_log(location, message, data, hypothesis_id=None):
    """Write debug log to NDJSON file"""
    try:
        import time
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
        pass  # Silently fail if logging fails

import sys
import torch
import torch.distributed as dist
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
from src.model.processor import load_processor, get_backbone_name


def find_transformer_layers(model):
    """
    找到transformer层的位置
    支持不同的模型结构：Qwen2-VL, Qwen3-VL等
    """
    encoder = model.encoder
    
    # 处理LoRA包装的模型
    if hasattr(encoder, 'base_model'):
        encoder = encoder.base_model
    
    # 尝试不同的路径来找到layers
    layers = None
    layer_path = None
    
    # Qwen3-VL: model.language_model.layers 或 model.model.language_model.layers
    if hasattr(encoder, 'model') and hasattr(encoder.model, 'language_model'):
        if hasattr(encoder.model.language_model, 'layers'):
            layers = encoder.model.language_model.layers
            layer_path = 'model.language_model.layers'
    elif hasattr(encoder, 'language_model'):
        if hasattr(encoder.language_model, 'layers'):
            layers = encoder.language_model.layers
            layer_path = 'language_model.layers'
    # Qwen2-VL: model.model.layers 或 model.layers
    elif hasattr(encoder, 'model') and hasattr(encoder.model, 'layers'):
        layers = encoder.model.layers
        layer_path = 'model.layers'
    elif hasattr(encoder, 'layers'):
        layers = encoder.layers
        layer_path = 'layers'
    
    if layers is None:
        raise ValueError("无法找到transformer层。请检查模型结构。")
    
    return layers, layer_path


def distribute_model_layers(model, world_size):
    """
    将模型的transformer层分配到不同的GPU上
    每个GPU负责一部分层
    """
    layers, layer_path = find_transformer_layers(model)
    num_layers = len(layers)
    
    print_master(f"找到 {num_layers} 个transformer层，路径: {layer_path}")
    print_master(f"使用 {world_size} 个GPU进行模型并行")
    
    # 计算每个GPU负责的层范围
    layers_per_gpu = num_layers // world_size
    remainder = num_layers % world_size
    
    # 分配层到不同的GPU
    start_idx = 0
    for rank in range(world_size):
        # 前remainder个GPU多分配一层
        end_idx = start_idx + layers_per_gpu + (1 if rank < remainder else 0)
        
        print_master(f"GPU {rank}: 层 {start_idx} 到 {end_idx-1} (共 {end_idx-start_idx} 层)")
        
        # 将层移动到对应的GPU
        # 使用.to()方法会递归移动所有子模块和参数
        for i in range(start_idx, end_idx):
            target_device = f'cuda:{rank}'
            # 先检查层是否已经在目标设备上
            layer_params = list(layers[i].parameters())
            layer_device = layer_params[0].device if layer_params else None
            
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{96}",
                f"分配层 {i} 到 GPU {rank}",
                {
                    "layer_idx": i,
                    "target_device": target_device,
                    "initial_device": str(layer_device),
                    "num_params": len(layer_params),
                    "gpu_memory_before": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(world_size)} if torch.cuda.is_available() else {}
                },
                hypothesis_id="A"
            )
            # #endregion
            
            if layer_device != torch.device(target_device):
                layers[i] = layers[i].to(target_device)
                # 确保所有参数都在目标设备上（双重检查）
                param_moved = 0
                for param in layers[i].parameters():
                    if param.device != torch.device(target_device):
                        param.data = param.data.to(target_device)
                        param_moved += 1
                # 确保所有缓冲区也在目标设备上
                buffer_moved = 0
                for buffer in layers[i].buffers():
                    if buffer.device != torch.device(target_device):
                        buffer.data = buffer.data.to(target_device)
                        buffer_moved += 1
                # 清理GPU缓存
                torch.cuda.empty_cache()
                
                # #region agent log
                debug_log(
                    f"train_model_parallel.py:{111}",
                    f"层 {i} 移动完成",
                    {
                        "layer_idx": i,
                        "target_device": target_device,
                        "params_moved": param_moved,
                        "buffers_moved": buffer_moved,
                        "gpu_memory_after": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(world_size)} if torch.cuda.is_available() else {}
                    },
                    hypothesis_id="A"
                )
                # #endregion
            
            # 验证层确实在目标设备上
            final_device = next(layers[i].parameters()).device if list(layers[i].parameters()) else None
            if final_device != torch.device(target_device):
                raise RuntimeError(f"层 {i} 未能移动到目标设备 {target_device}，实际设备: {final_device}")
        
        start_idx = end_idx
    
    # 确保embedding层和输出层在正确的GPU上
    encoder = model.encoder
    if hasattr(encoder, 'base_model'):
        encoder = encoder.base_model
    
    # 移动embedding层到GPU 0
    # 处理不同的模型结构：Qwen2-VL, Qwen3-VL等
    if hasattr(encoder, 'model'):
        model_part = encoder.model
        # Qwen3-VL: model.language_model.embed_tokens
        if hasattr(model_part, 'language_model') and hasattr(model_part.language_model, 'embed_tokens'):
            embed_size = model_part.language_model.embed_tokens.weight.numel() * model_part.language_model.embed_tokens.weight.element_size() / 1024**3
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{132}",
                "移动embedding层到GPU 0",
                {
                    "embedding_size_gb": embed_size,
                    "gpu_0_memory_before": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
                },
                hypothesis_id="B"
            )
            # #endregion
            model_part.language_model.embed_tokens = model_part.language_model.embed_tokens.to('cuda:0')
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{132}",
                "embedding层移动完成",
                {
                    "gpu_0_memory_after": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
                },
                hypothesis_id="B"
            )
            # #endregion
        # Qwen2-VL: model.embed_tokens
        elif hasattr(model_part, 'embed_tokens'):
            embed_size = model_part.embed_tokens.weight.numel() * model_part.embed_tokens.weight.element_size() / 1024**3
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{135}",
                "移动embedding层到GPU 0",
                {
                    "embedding_size_gb": embed_size,
                    "gpu_0_memory_before": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
                },
                hypothesis_id="B"
            )
            # #endregion
            model_part.embed_tokens = model_part.embed_tokens.to('cuda:0')
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{135}",
                "embedding层移动完成",
                {
                    "gpu_0_memory_after": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
                },
                hypothesis_id="B"
            )
            # #endregion
    elif hasattr(encoder, 'embed_tokens'):
        embed_size = encoder.embed_tokens.weight.numel() * encoder.embed_tokens.weight.element_size() / 1024**3
        # #region agent log
        debug_log(
            f"train_model_parallel.py:{137}",
            "移动embedding层到GPU 0",
            {
                "embedding_size_gb": embed_size,
                "gpu_0_memory_before": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
            },
            hypothesis_id="B"
        )
        # #endregion
        encoder.embed_tokens = encoder.embed_tokens.to('cuda:0')
        # #region agent log
        debug_log(
            f"train_model_parallel.py:{137}",
            "embedding层移动完成",
            {
                "gpu_0_memory_after": torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
            },
            hypothesis_id="B"
        )
        # #endregion
    
    # 移动norm层到最后一个GPU
    norm_layer = None
    if hasattr(encoder, 'model'):
        model_part = encoder.model
        # Qwen3-VL: model.language_model.norm
        if hasattr(model_part, 'language_model') and hasattr(model_part.language_model, 'norm'):
            norm_layer = model_part.language_model.norm
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{267}",
                "找到norm层（Qwen3-VL）",
                {
                    "norm_device_before": str(next(norm_layer.parameters()).device) if list(norm_layer.parameters()) else None,
                    "target_device": f'cuda:{world_size-1}'
                },
                hypothesis_id="E"
            )
            # #endregion
            norm_layer = norm_layer.to(f'cuda:{world_size-1}')
            # 确保所有参数和缓冲区都在目标设备上
            for param in norm_layer.parameters():
                if param.device != torch.device(f'cuda:{world_size-1}'):
                    param.data = param.data.to(f'cuda:{world_size-1}')
            for buffer in norm_layer.buffers():
                if buffer.device != torch.device(f'cuda:{world_size-1}'):
                    buffer.data = buffer.data.to(f'cuda:{world_size-1}')
            model_part.language_model.norm = norm_layer
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{267}",
                "norm层移动完成（Qwen3-VL）",
                {
                    "norm_device_after": str(next(norm_layer.parameters()).device) if list(norm_layer.parameters()) else None,
                    "target_device": f'cuda:{world_size-1}'
                },
                hypothesis_id="E"
            )
            # #endregion
        # Qwen2-VL: model.norm
        elif hasattr(model_part, 'norm'):
            norm_layer = model_part.norm
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{270}",
                "找到norm层（Qwen2-VL）",
                {
                    "norm_device_before": str(next(norm_layer.parameters()).device) if list(norm_layer.parameters()) else None,
                    "target_device": f'cuda:{world_size-1}'
                },
                hypothesis_id="E"
            )
            # #endregion
            norm_layer = norm_layer.to(f'cuda:{world_size-1}')
            # 确保所有参数和缓冲区都在目标设备上
            for param in norm_layer.parameters():
                if param.device != torch.device(f'cuda:{world_size-1}'):
                    param.data = param.data.to(f'cuda:{world_size-1}')
            for buffer in norm_layer.buffers():
                if buffer.device != torch.device(f'cuda:{world_size-1}'):
                    buffer.data = buffer.data.to(f'cuda:{world_size-1}')
            model_part.norm = norm_layer
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{270}",
                "norm层移动完成（Qwen2-VL）",
                {
                    "norm_device_after": str(next(norm_layer.parameters()).device) if list(norm_layer.parameters()) else None,
                    "target_device": f'cuda:{world_size-1}'
                },
                hypothesis_id="E"
            )
            # #endregion
    elif hasattr(encoder, 'norm'):
        norm_layer = encoder.norm
        # #region agent log
        debug_log(
            f"train_model_parallel.py:{272}",
            "找到norm层（直接）",
            {
                "norm_device_before": str(next(norm_layer.parameters()).device) if list(norm_layer.parameters()) else None,
                "target_device": f'cuda:{world_size-1}'
            },
            hypothesis_id="E"
        )
        # #endregion
        norm_layer = norm_layer.to(f'cuda:{world_size-1}')
        # 确保所有参数和缓冲区都在目标设备上
        for param in norm_layer.parameters():
            if param.device != torch.device(f'cuda:{world_size-1}'):
                param.data = param.data.to(f'cuda:{world_size-1}')
        for buffer in norm_layer.buffers():
            if buffer.device != torch.device(f'cuda:{world_size-1}'):
                buffer.data = buffer.data.to(f'cuda:{world_size-1}')
        encoder.norm = norm_layer
        # #region agent log
        debug_log(
            f"train_model_parallel.py:{272}",
            "norm层移动完成（直接）",
            {
                "norm_device_after": str(next(norm_layer.parameters()).device) if list(norm_layer.parameters()) else None,
                "target_device": f'cuda:{world_size-1}'
            },
            hypothesis_id="E"
        )
        # #endregion
    
    # 存储层分配信息，用于前向传播
    model._mp_layer_ranges = []
    start_idx = 0
    for rank in range(world_size):
        end_idx = start_idx + layers_per_gpu + (1 if rank < remainder else 0)
        model._mp_layer_ranges.append((start_idx, end_idx, rank))
        start_idx = end_idx
    
    model._mp_world_size = world_size
    model._mp_layer_path = layer_path
    # 保存layers引用，确保前向传播时使用同一个对象
    model._mp_layers = layers
    
    print_master("模型层分配完成")
    return model


class ModelParallelMMEBModel(MMEBModel):
    """
    支持模型并行的MMEBModel
    重写encode_input方法以支持跨设备的前向传播
    """
    
    @property
    def device(self):
        """
        对于模型并行，返回第一个GPU（embedding层所在的设备）
        """
        return torch.device('cuda:0')
    
    def encode_input(self, input, selected_layers):
        """
        重写encode_input以支持模型并行
        数据在GPU间传递，每层在对应的GPU上计算
        """
        from src.model.processor import QWEN3_VL
        from transformers.modeling_outputs import BaseModelOutputWithPast
        
        model_backbone = get_backbone_name(hf_config=self.config)
        
        # 处理Qwen3-VL的特殊输入格式（与原始代码保持一致）
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
        else:
            # Qwen2-VL, Qwen2.5-VL等处理
            pass
        
        # 模型并行前向传播
        # 将输入移动到第一个GPU
        device = 'cuda:0'
        # #region agent log
        debug_log(
            f"train_model_parallel.py:{470}",
            "encode_input 开始，记录初始内存状态",
            {
                "gpu_memory_initial": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {},
                "input_keys": list(input.keys()),
                "pixel_values_shape": list(input.get('pixel_values', 'N/A').shape) if isinstance(input.get('pixel_values'), torch.Tensor) else str(type(input.get('pixel_values'))),
                "pixel_values_device": str(input.get('pixel_values', 'N/A').device) if isinstance(input.get('pixel_values'), torch.Tensor) else 'N/A'
            },
            hypothesis_id="B"
        )
        # #endregion
        input_on_device = {}
        for k, v in input.items():
            if isinstance(v, torch.Tensor):
                # #region agent log
                if k in ['pixel_values', 'pixel_values_videos']:
                    debug_log(
                        f"train_model_parallel.py:{475}",
                        f"移动大张量 {k} 到 GPU 0",
                        {
                            "key": k,
                            "shape": list(v.shape),
                            "dtype": str(v.dtype),
                            "size_mb": v.numel() * v.element_size() / 1024**2,
                            "source_device": str(v.device),
                            "target_device": device,
                            "gpu_memory_before_move": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                        },
                        hypothesis_id="B"
                    )
                # #endregion
                input_on_device[k] = v.to(device)
                # #region agent log
                if k in ['pixel_values', 'pixel_values_videos']:
                    debug_log(
                        f"train_model_parallel.py:{475}",
                        f"大张量 {k} 移动完成",
                        {
                            "key": k,
                            "gpu_memory_after_move": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                        },
                        hypothesis_id="B"
                    )
                # #endregion
            elif isinstance(v, list):
                # 处理列表类型的输入（如pixel_values可能是list）
                input_on_device[k] = v
            else:
                input_on_device[k] = v
        
        # 获取encoder和layers
        encoder = self.encoder
        if hasattr(encoder, 'base_model'):
            encoder = encoder.base_model
        
        # 获取embedding层和模型结构
        if hasattr(encoder, 'model'):
            model_part = encoder.model
            if hasattr(model_part, 'embed_tokens'):
                embed_tokens = model_part.embed_tokens
            elif hasattr(model_part, 'language_model') and hasattr(model_part.language_model, 'embed_tokens'):
                embed_tokens = model_part.language_model.embed_tokens
                model_part = model_part.language_model
            else:
                raise ValueError("无法找到embedding层")
        elif hasattr(encoder, 'embed_tokens'):
            embed_tokens = encoder.embed_tokens
            model_part = encoder
        else:
            raise ValueError("无法找到embedding层")
        
        # 获取layers - 使用保存的layers引用，确保使用已分配的层
        if hasattr(self, '_mp_layers'):
            layers = self._mp_layers
        else:
            # 如果没有保存的layers，则重新查找（向后兼容）
            layers, _ = find_transformer_layers(self)
        
        # 获取input_ids
        input_ids = input_on_device.get('input_ids')
        if input_ids is None:
            raise ValueError("需要input_ids进行前向传播")
        
        # 获取attention_mask和position_ids
        attention_mask = input_on_device.get('attention_mask')
        position_ids = input_on_device.get('position_ids')
        
        # 处理position_ids（Qwen模型可能需要特殊格式）
        model_backbone = get_backbone_name(hf_config=self.config)
        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=device).unsqueeze(0).expand_as(input_ids)
        
        # Qwen2-VL和Qwen3-VL可能需要3D position_ids（temporal, height, width）
        # 如果position_ids是2D，需要expand到3D
        if model_backbone in ['qwen2_vl', 'qwen2_5_vl', 'qwen3_vl'] and position_ids.dim() == 2:
            # 参考Qwen2-VL的实现：expand到3维
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        
        # Embedding
        hidden_states = embed_tokens(input_ids)
        
        # 计算position_embeddings（Qwen模型需要这个）
        # 获取rotary_emb（如果存在）
        rotary_emb = None
        
        # 对于Qwen3VL，rotary_emb可能在language_model中
        if model_backbone == 'qwen3_vl':
            # Qwen3VL结构: encoder.model.language_model 或 encoder.language_model
            if hasattr(encoder, 'model') and hasattr(encoder.model, 'language_model'):
                lang_model = encoder.model.language_model
                if hasattr(lang_model, 'rotary_emb'):
                    rotary_emb = lang_model.rotary_emb
            elif hasattr(encoder, 'language_model'):
                lang_model = encoder.language_model
                if hasattr(lang_model, 'rotary_emb'):
                    rotary_emb = lang_model.rotary_emb
        
        # 通用查找rotary_emb
        if rotary_emb is None:
            if hasattr(model_part, 'rotary_emb'):
                rotary_emb = model_part.rotary_emb
            elif hasattr(encoder, 'rotary_emb'):
                rotary_emb = encoder.rotary_emb
            elif hasattr(encoder, 'model') and hasattr(encoder.model, 'rotary_emb'):
                rotary_emb = encoder.model.rotary_emb
        
        position_embeddings = None
        if rotary_emb is not None:
            # 计算position_embeddings（在第一个GPU上）
            # rotary_emb需要hidden_states和position_ids
            try:
                position_embeddings = rotary_emb(hidden_states, position_ids)
                logger.info(f"成功计算position_embeddings，类型: {type(position_embeddings)}")
            except Exception as e:
                logger.error(f"计算position_embeddings失败: {e}")
                raise RuntimeError(f"无法计算position_embeddings，这是Qwen3VL的必需参数: {e}")
        else:
            # 对于Qwen3VL，position_embeddings是必需的，不能为None
            if model_backbone == 'qwen3_vl':
                raise RuntimeError(f"未找到rotary_emb，无法计算position_embeddings。这是Qwen3VL的必需参数。请检查模型结构。")
            else:
                logger.warning(f"未找到rotary_emb，无法计算position_embeddings。模型类型: {model_backbone}")
        
        # 存储所有层的hidden states（用于selected_layers）
        all_hidden_states = [hidden_states]
        
        # 逐层前向传播，在对应的GPU上计算
        for layer_idx, layer in enumerate(layers):
            # 确定这一层应该在哪个GPU上
            target_device = None
            for start_idx, end_idx, gpu_rank in self._mp_layer_ranges:
                if start_idx <= layer_idx < end_idx:
                    target_device = gpu_rank
                    break
            
            if target_device is None:
                raise ValueError(f"无法确定层 {layer_idx} 应该在哪个GPU上")
            
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{360}",
                f"层 {layer_idx} 前向传播开始",
                {
                    "layer_idx": layer_idx,
                    "target_device": target_device,
                    "hidden_states_device": str(hidden_states.device) if hasattr(hidden_states, 'device') else None,
                    "hidden_states_shape": list(hidden_states.shape) if hasattr(hidden_states, 'shape') else None,
                    "gpu_memory_before": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {},
                    "layer_device": str(next(layer.parameters()).device) if list(layer.parameters()) else None
                },
                hypothesis_id="C"
            )
            # #endregion
            
            # 移动hidden_states到目标GPU
            target_device_str = f'cuda:{target_device}'
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{610}",
                f"准备移动 hidden_states 到目标 GPU",
                {
                    "layer_idx": layer_idx,
                    "source_device": str(hidden_states.device) if hasattr(hidden_states, 'device') else None,
                    "target_device": target_device_str,
                    "hidden_states_shape": list(hidden_states.shape) if hasattr(hidden_states, 'shape') else None,
                    "hidden_states_size_mb": hidden_states.numel() * hidden_states.element_size() / 1024**2 if hasattr(hidden_states, 'numel') else None,
                    "gpu_memory_before_hidden_move": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                },
                hypothesis_id="D"
            )
            # #endregion
            hidden_states = hidden_states.to(target_device_str)
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{612}",
                f"hidden_states 移动完成",
                {
                    "layer_idx": layer_idx,
                    "final_device": str(hidden_states.device) if hasattr(hidden_states, 'device') else None,
                    "gpu_memory_after_hidden_move": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                },
                hypothesis_id="D"
            )
            # #endregion
            
            # 确保layer及其所有子模块都在目标设备上（双重检查）
            # 这很重要，因为某些子模块可能在之前的操作中被移动到了其他设备
            layer_device = next(layer.parameters()).device if list(layer.parameters()) else torch.device(target_device_str)
            if layer_device != torch.device(target_device_str):
                logger.warning(f"层 {layer_idx} 的设备 ({layer_device}) 与目标设备 ({target_device_str}) 不匹配，正在移动...")
                layer = layer.to(target_device_str)
            
            # 准备layer的输入 - 使用encoder的forward方法签名
            layer_kwargs = {
                'hidden_states': hidden_states,
                'attention_mask': attention_mask.to(target_device_str) if attention_mask is not None else None,
                'position_ids': position_ids.to(target_device_str),
                'output_hidden_states': True,
            }
            
            # 添加position_embeddings（Qwen3VL必需这个参数）
            # position_embeddings 是一个tuple (cos, sin)，需要移动到目标GPU
            if position_embeddings is not None:
                # #region agent log
                debug_log(
                    f"train_model_parallel.py:{631}",
                    f"准备移动 position_embeddings 到目标 GPU",
                    {
                        "layer_idx": layer_idx,
                        "is_tuple": isinstance(position_embeddings, tuple),
                        "source_device_0": str(position_embeddings[0].device) if isinstance(position_embeddings, tuple) else str(position_embeddings.device),
                        "target_device": target_device_str,
                        "gpu_memory_before_pos_move": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                    },
                    hypothesis_id="C"
                )
                # #endregion
                if isinstance(position_embeddings, tuple):
                    layer_kwargs['position_embeddings'] = (
                        position_embeddings[0].to(target_device_str),
                        position_embeddings[1].to(target_device_str)
                    )
                else:
                    layer_kwargs['position_embeddings'] = position_embeddings.to(target_device_str)
                # #region agent log
                debug_log(
                    f"train_model_parallel.py:{638}",
                    f"position_embeddings 移动完成",
                    {
                        "layer_idx": layer_idx,
                        "gpu_memory_after_pos_move": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                    },
                    hypothesis_id="C"
                )
                # #endregion
            else:
                # 对于Qwen3VL，position_embeddings是必需的，不能为None
                if model_backbone == 'qwen3_vl':
                    raise RuntimeError(f"position_embeddings为None，但Qwen3VL的layer需要这个参数。请确保正确计算了position_embeddings。")
                # 对于其他模型，不传递position_embeddings（如果为None）
            
            # 添加视觉相关的输入（如果存在）
            # 优化：如果已经在目标设备上，直接使用，避免不必要的复制
            for k in ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw']:
                if k in input_on_device and input_on_device[k] is not None:
                    if isinstance(input_on_device[k], torch.Tensor):
                        # #region agent log
                        debug_log(
                            f"train_model_parallel.py:{649}",
                            f"准备移动视觉输入 {k} 到目标 GPU",
                            {
                                "layer_idx": layer_idx,
                                "key": k,
                                "shape": list(input_on_device[k].shape),
                                "size_mb": input_on_device[k].numel() * input_on_device[k].element_size() / 1024**2,
                                "source_device": str(input_on_device[k].device),
                                "target_device": target_device_str,
                                "gpu_memory_before_vis_move": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                            },
                            hypothesis_id="B"
                        )
                        # #endregion
                        # 优化：如果已经在目标设备上，直接使用引用，避免复制
                        if input_on_device[k].device == torch.device(target_device_str):
                            layer_kwargs[k] = input_on_device[k]
                        else:
                            layer_kwargs[k] = input_on_device[k].to(target_device_str)
                        # #region agent log
                        debug_log(
                            f"train_model_parallel.py:{649}",
                            f"视觉输入 {k} 移动完成",
                            {
                                "layer_idx": layer_idx,
                                "key": k,
                                "gpu_memory_after_vis_move": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                            },
                            hypothesis_id="B"
                        )
                        # #endregion
                    else:
                        layer_kwargs[k] = input_on_device[k]
            
            # 调用layer
            try:
                # #region agent log
                debug_log(
                    f"train_model_parallel.py:{655}",
                    f"准备调用 layer {layer_idx}",
                    {
                        "layer_idx": layer_idx,
                        "target_device": target_device_str,
                        "layer_kwargs_keys": list(layer_kwargs.keys()),
                        "has_pixel_values": 'pixel_values' in layer_kwargs and layer_kwargs['pixel_values'] is not None,
                        "pixel_values_shape": list(layer_kwargs.get('pixel_values', 'N/A').shape) if isinstance(layer_kwargs.get('pixel_values'), torch.Tensor) else 'N/A',
                        "hidden_states_shape": list(layer_kwargs.get('hidden_states', 'N/A').shape) if isinstance(layer_kwargs.get('hidden_states'), torch.Tensor) else 'N/A',
                        "gpu_memory_before_layer_call": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {},
                        "gpu_memory_reserved": {f"cuda:{gpu}": torch.cuda.memory_reserved(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                    },
                    hypothesis_id="A"
                )
                # #endregion
                layer_outputs = layer(**layer_kwargs)
                # #region agent log
                debug_log(
                    f"train_model_parallel.py:{656}",
                    f"layer {layer_idx} 调用完成",
                    {
                        "layer_idx": layer_idx,
                        "gpu_memory_after_layer_call": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {},
                        "gpu_memory_reserved": {f"cuda:{gpu}": torch.cuda.memory_reserved(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                    },
                    hypothesis_id="A"
                )
                # #endregion
                # 处理不同的输出格式
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                elif hasattr(layer_outputs, 'hidden_states'):
                    hidden_states = layer_outputs.hidden_states
                elif hasattr(layer_outputs, 'last_hidden_state'):
                    hidden_states = layer_outputs.last_hidden_state
                else:
                    hidden_states = layer_outputs
                
                # 清理中间结果以节省内存（只保留hidden_states）
                # #region agent log
                debug_log(
                    f"train_model_parallel.py:{667}",
                    f"准备清理 layer {layer_idx} 的中间结果",
                    {
                        "layer_idx": layer_idx,
                        "gpu_memory_before_cleanup": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {},
                        "gpu_memory_reserved": {f"cuda:{gpu}": torch.cuda.memory_reserved(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                    },
                    hypothesis_id="A"
                )
                # #endregion
                del layer_outputs
                # 清理 layer_kwargs 中的大张量（特别是 pixel_values）
                for k in ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw']:
                    if k in layer_kwargs:
                        # #region agent log
                        if k in ['pixel_values', 'pixel_values_videos']:
                            debug_log(
                                f"train_model_parallel.py:{672}",
                                f"删除 layer_kwargs 中的 {k}",
                                {
                                    "layer_idx": layer_idx,
                                    "key": k,
                                    "gpu_memory_before_del": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                                },
                                hypothesis_id="B"
                            )
                        # #endregion
                        del layer_kwargs[k]
                del layer_kwargs
                # 强制同步并清空缓存，减少内存碎片
                if torch.cuda.is_available():
                    torch.cuda.synchronize(target_device_str)
                torch.cuda.empty_cache()
                # #region agent log
                debug_log(
                    f"train_model_parallel.py:{680}",
                    f"layer {layer_idx} 清理完成",
                    {
                        "layer_idx": layer_idx,
                        "gpu_memory_after_cleanup": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {},
                        "gpu_memory_reserved": {f"cuda:{gpu}": torch.cuda.memory_reserved(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                    },
                    hypothesis_id="A"
                )
                # #endregion
                
                # #region agent log
                debug_log(
                    f"train_model_parallel.py:{428}",
                    f"层 {layer_idx} 前向传播完成",
                    {
                        "layer_idx": layer_idx,
                        "target_device": target_device,
                        "hidden_states_device": str(hidden_states.device) if hasattr(hidden_states, 'device') else None,
                        "gpu_memory_after": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {},
                        "all_hidden_states_count": len(all_hidden_states) + 1
                    },
                    hypothesis_id="C"
                )
                # #endregion
            except Exception as e:
                logger.error(f"层 {layer_idx} 前向传播失败: {e}")
                # #region agent log
                debug_log(
                    f"train_model_parallel.py:{431}",
                    f"层 {layer_idx} 前向传播失败",
                    {
                        "layer_idx": layer_idx,
                        "target_device": target_device,
                        "error": str(e),
                        "gpu_memory": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                    },
                    hypothesis_id="C"
                )
                # #endregion
                raise
            
            all_hidden_states.append(hidden_states)
            
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{434}",
                f"层 {layer_idx} hidden_states已添加到列表",
                {
                    "layer_idx": layer_idx,
                    "all_hidden_states_count": len(all_hidden_states),
                    "gpu_memory": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                },
                hypothesis_id="D"
            )
            # #endregion
            
            # 如果不需要保留所有层的hidden states，可以只保留selected_layers的
            # 但为了兼容性，我们保留所有层的hidden states
        
        # 应用norm层（在最后一个GPU上）
        # 使用与distribute_model_layers相同的查找逻辑
        encoder = self.encoder
        if hasattr(encoder, 'base_model'):
            encoder = encoder.base_model
        
        norm = None
        if hasattr(encoder, 'model'):
            model_part_for_norm = encoder.model
            # Qwen3-VL: model.language_model.norm
            if hasattr(model_part_for_norm, 'language_model') and hasattr(model_part_for_norm.language_model, 'norm'):
                norm = model_part_for_norm.language_model.norm
            # Qwen2-VL: model.norm
            elif hasattr(model_part_for_norm, 'norm'):
                norm = model_part_for_norm.norm
        elif hasattr(encoder, 'norm'):
            norm = encoder.norm
        
        if norm is not None:
            target_norm_device = f'cuda:{self._mp_world_size-1}'
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{625}",
                "应用norm层前",
                {
                    "hidden_states_device": str(hidden_states.device) if hasattr(hidden_states, 'device') else None,
                    "norm_weight_device": str(next(norm.parameters()).device) if list(norm.parameters()) else None,
                    "target_device": target_norm_device,
                    "gpu_memory": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                },
                hypothesis_id="E"
            )
            # #endregion
            
            # 确保norm层在目标设备上（双重检查）
            norm_device = next(norm.parameters()).device if list(norm.parameters()) else None
            if norm_device != torch.device(target_norm_device):
                logger.warning(f"norm层设备 ({norm_device}) 与目标设备 ({target_norm_device}) 不匹配，正在移动...")
                norm = norm.to(target_norm_device)
                # 确保所有参数和缓冲区都在目标设备上
                for param in norm.parameters():
                    if param.device != torch.device(target_norm_device):
                        param.data = param.data.to(target_norm_device)
                for buffer in norm.buffers():
                    if buffer.device != torch.device(target_norm_device):
                        buffer.data = buffer.data.to(target_norm_device)
            
            hidden_states = hidden_states.to(target_norm_device)
            hidden_states = norm(hidden_states)
            all_hidden_states[-1] = hidden_states
            
            # #region agent log
            debug_log(
                f"train_model_parallel.py:{627}",
                "norm层应用完成",
                {
                    "hidden_states_device_after": str(hidden_states.device) if hasattr(hidden_states, 'device') else None,
                    "norm_weight_device_after": str(next(norm.parameters()).device) if list(norm.parameters()) else None,
                    "gpu_memory": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
                },
                hypothesis_id="E"
            )
            # #endregion
        
        # 提取selected_layers的hidden states并pooling
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
        
        # 清理 input_on_device 中的大张量，释放内存
        # #region agent log
        debug_log(
            f"train_model_parallel.py:{1000}",
            "encode_input 结束前清理大张量",
            {
                "gpu_memory_before_cleanup": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
            },
            hypothesis_id="B"
        )
        # #endregion
        for k in ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw']:
            if k in input_on_device:
                del input_on_device[k]
        # 强制同步并清空缓存
        if torch.cuda.is_available():
            for gpu in range(self._mp_world_size):
                torch.cuda.synchronize(f'cuda:{gpu}')
        torch.cuda.empty_cache()
        # #region agent log
        debug_log(
            f"train_model_parallel.py:{1015}",
            "encode_input 清理完成",
            {
                "gpu_memory_after_cleanup": {f"cuda:{gpu}": torch.cuda.memory_allocated(gpu) / 1024**3 for gpu in range(self._mp_world_size)} if torch.cuda.is_available() else {}
            },
            hypothesis_id="B"
        )
        # #endregion
        
        return all_outputs


def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
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

    # DEBUG PRINTS for Distributed Setup
    print("Distributed init debug info:")
    print(f"RANK: {os.environ.get('RANK')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")

    if torch.distributed.is_available():
        print(f"torch.distributed.is_initialized: {torch.distributed.is_initialized()}")
        if torch.distributed.is_initialized():
            print(f"torch.distributed.get_rank(): {torch.distributed.get_rank()}")
            print(f"torch.distributed.get_world_size(): {torch.distributed.get_world_size()}")

    # 获取world_size用于模型并行
    # 注意：模型并行不需要DDP，我们直接使用可见的GPU数量
    # 如果设置了CUDA_VISIBLE_DEVICES，使用该设置；否则使用所有可用GPU
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        world_size = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    else:
        world_size = torch.cuda.device_count()
    print_master(f"使用 {world_size} 个GPU进行模型并行（模型分卡）")

    # Check for existing checkpoints
    if training_args.resume_from == 'auto':
        resume_checkpoint_dir = find_latest_checkpoint(training_args.output_dir)
        if resume_checkpoint_dir:
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    elif training_args.resume_from.isdigit():
        resume_checkpoint_dir = os.path.join(training_args.output_dir, f'checkpoint-{training_args.resume_from}')
        if os.path.exists(resume_checkpoint_dir):
            logger.info(f"Resuming from checkpoint: {resume_checkpoint_dir}")
    else:
        resume_checkpoint_dir = None
        logger.info("No checkpoint found. Starting fresh training.")

    # Initialize WandB if enabled
    if 'wandb' in training_args.report_to:
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or (not torch.distributed.is_initialized()):
            print_rank('init wandb')
            wandb.init(project=training_args.project_name, name=training_args.run_name, mode="online")
            wandb.config.update(model_args)
            wandb.config.update(data_args)
            wandb.config.update(training_args)

    # If checkpoint_path is provided and contains LoRA adapter, use load() instead of build()
    # to load the trained LoRA weights instead of creating new random ones
    if model_args.checkpoint_path and model_args.lora:
        adapter_config_path = os.path.join(model_args.checkpoint_path, 'adapter_config.json')
        if os.path.exists(adapter_config_path):
            logger.info(f"Found LoRA adapter in checkpoint_path, loading model with trained weights from {model_args.checkpoint_path}")
            base_model = MMEBModel.load(model_args, is_trainable=True)
        else:
            logger.info(f"No LoRA adapter found in checkpoint_path, building new model")
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
    
    # 分配模型层到不同的GPU
    model = distribute_model_layers(model, world_size)
    
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

    # 注意：模型并行训练不使用DDP和DataParallel
    # 设置local_rank为-1以禁用DDP
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
    
    # 重写_wrap_model方法以禁用DataParallel（模型并行与DataParallel冲突）
    # 保存原始方法以便需要时恢复
    original_wrap_model = trainer._wrap_model
    def _wrap_model_no_dp(self, model, training=True, dataloader=None):
        # 对于模型并行，不包装模型，直接返回
        # 这样可以避免DataParallel自动包装模型
        return model
    trainer._wrap_model = _wrap_model_no_dp.__get__(trainer, type(trainer))

    print_master("开始模型并行训练...")
    print_master(f"模型层分配：每个GPU负责一部分层")
    print_master(f"已禁用DataParallel，使用模型并行（模型分卡）")
    trainer.train(resume_from_checkpoint=resume_checkpoint_dir)
    
    # 恢复原始设置（如果需要）
    training_args.local_rank = original_local_rank
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
