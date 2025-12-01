import os
from typing import Dict
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
from src.model.processor import QWEN2_5_VL_TOKENSELECTION
from src.arguments import ModelArguments, TrainingArguments
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL, \
    backbone2model, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, E5_V, QWEN2_VL_TAIL

from src.arguments import ModelArguments
from src.model.processor import LLAVA_NEXT, QWEN2_VL, PHI3V, get_backbone_name, print_master, QWEN2_5_VL, INTERNVIDEO2, \
    QWEN2_VL_TOKENSELECTION, backbone2model, GME, VLM_IMAGE_TOKENS, LamRA, LamRA_QWEN2_5, COLPALI
from src.model.baseline_backbone.colpali import ColPali
from src.model.baseline_backbone.gme.gme_inference import GmeQwen2VL
from src.model.baseline_backbone.lamra.lamra_inference import LamRAQwen2VL
from src.model.baseline_backbone.lamra.lamra_qwen25_inference import LamRAQwen25VL
from src.model.baseline_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.model.baseline_backbone.llava_next import LlavaNextForConditionalGeneration

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']

class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'last',
                 normalize: bool = False,
                 temperature: float = 0.02
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def encode_input(self, input, chosen_layer):
        if getattr(self, "model_backbone", None) == INTERNVIDEO2:
            if "input_ids" in input.keys():
                # text side
                text_output = self.encoder.get_text_encoder()(
                    input["input_ids"],
                    attention_mask=input["attention_mask"],
                    return_dict=True,
                    mode="text",
                )
                text_embeds = text_output.last_hidden_state
                pooled_text_embeds = text_embeds[:, 0]
                pooled_output = self.encoder.text_proj(pooled_text_embeds)
                pooled_output /= pooled_output.norm(dim=-1, keepdim=True)
                return pooled_output
            else:
                _, vfeat = self.encoder.encode_vision(input["pixel_values"], test=True)
                vfeat = self.encoder.vision_proj(vfeat)
                vfeat /= vfeat.norm(dim=-1, keepdim=True)
                return vfeat
        elif getattr(self, "model_backbone", None) in [GME, LamRA, LamRA_QWEN2_5]:
            # pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            texts = [text.replace(VLM_IMAGE_TOKENS[QWEN2_VL] + '\n', '') for text in input["texts"]] # we are actually passing video queries so this should not happen
            images = []
            for imgs in input['images']:
                # if multi images are given, select the middle frame only
                if isinstance(imgs, list):
                    imgs = imgs[len(imgs) // 2]
                    assert not isinstance(imgs, list) # make sure we have extracted the middle frame and it is no longer a list
                    images.append(imgs)
                else:
                    images.append(imgs)
            try:
                self.encoder.set_chosen_layer(chosen_layer)
            except AttributeError:
                print("Layer choice not implemented.")
            try:
                self.encoder.set_pooling(self.pooling)
            except AttributeError:
                print("Pooling not implemented.")
            pooled_output = self.encoder.get_fused_embeddings(texts=texts, images=images)
            return pooled_output
        elif getattr(self, "model_backbone", None) == COLPALI:
            try:
                self.encoder.set_chosen_layer(chosen_layer)
            except AttributeError:
                print("Layer choice not implemented.")
            try:
                self.encoder.set_pooling(self.pooling)
            except AttributeError:
                print("Pooling not implemented.")
            pooled_output = self.encoder(**input, return_dict=True, output_hidden_states=True)
            return pooled_output
        elif getattr(self, "model_backbone", None) == LLAVA_NEXT:
            input['pixel_values'] = input['pixel_values'].squeeze(dim=1)
            input['image_sizes'] = input['image_sizes'].squeeze(dim=1)
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[chosen_layer]
            pooled_output = self._pooling(hidden_states, input['attention_mask'])
            return pooled_output
        else:
            hidden_states = self.encoder(**input, return_dict=True, output_hidden_states=True)
            hidden_states = hidden_states.hidden_states[chosen_layer]
            # print_master("HIDDEN_STATES:")
            # print_master(hidden_states.shape)
            pooled_output = self._pooling(hidden_states, input['attention_mask'])
            # print_master("POOLED_OUTPUT:")
            # print_master(pooled_output.shape)
            return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                # print_master("LEFT_PADDING")
                # Get the vectors at the last position
                reps = last_hidden_state[torch.arange(batch_size), -1, :]
            else:
                # Calculate last 1 position in the original tensor
                eos_indices = attention_mask.sum(dim=1) - 1
                # Get the vectors at the last 1 position of each attention mask
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), eos_indices]
        elif self.pooling == 'average':
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            batch_size = last_hidden_state.shape[0]
            if left_padding:
                reps = last_hidden_state[torch.arange(batch_size)].mean(dim=1)
            else:
                eos_indices = attention_mask.sum(dim=1)
                reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), :eos_indices].mean(dim=1)
        else:
            raise NotImplementedError
        if self.normalize:
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    @classmethod
    def build(cls, model_args: ModelArguments, **kwargs):
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        model_backbone = get_backbone_name(hf_config=config)
        print_master(f'Loading backbone [{model_backbone}] from {model_args.model_name}')

        config.delete_L = model_args.delete_L
        config.delete_n = model_args.delete_n

        # Loading the base model
        if model_backbone == PHI3V:
            config._attn_implementation = "eager"
            config.padding_side = "right"
            config.use_cache = False
            base_model = Phi3VForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone == LLAVA_NEXT:
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_backbone in [QWEN2_VL, QWEN2_5_VL]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False
            if model_args.plus_one_token:
                base_model = backbone2model['qwen2_vl_tail'].from_pretrained(
                    model_args.model_name,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                # torch.nn.init.normal_(base_model.tail_emb, mean=0.0, std=1e-4)
                eos_token_id = config.eos_token_id
                with torch.no_grad():
                    emb = base_model.get_input_embeddings().weight
                    base_model.tail_emb.copy_(emb[eos_token_id])
                    # Frozen the tail embedding to eos embedding, for debug
                    base_model.tail_emb.requires_grad = False
            else:
                base_model = backbone2model[model_backbone].from_pretrained(
                    model_args.model_name,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )


        elif model_backbone in [QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION]:
            config._attn_implementation = "flash_attention_2"
            config.padding_side = "left"
            config.use_cache = False

            from src.utils import parse_layer_type
            lm_qwen_layer = 28
            vis_qwen_layer = 32
            lm_skip_layer = parse_layer_type(model_args.lm_skip_layer, lm_qwen_layer)
            vis_skip_layer = parse_layer_type(model_args.vis_skip_layer, vis_qwen_layer)

            base_model = backbone2model[model_backbone].from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                lm_skip_layer=lm_skip_layer,
                vis_skip_layer=vis_skip_layer,
            )
        else:
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_args.model_name, **kwargs, config=config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        if model_args.lora:
            print_master(f'Loading lora adapter from {base_model}')
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            if model_args.tail_token_train_only:
                for name, param in lora_model.named_parameters():
                    param.requires_grad = False
                
                # Set LoRA weights to zero
                for name, module in lora_model.named_modules():
                    if hasattr(module, "lora_A"):
                        for k, p in module.lora_A.items():
                            p.weight.data.zero_()
                    if hasattr(module, "lora_B"):
                        for k, p in module.lora_B.items():
                            p.weight.data.zero_()
                print_master("All LoRA parameters are set to 0.")

            if hasattr(lora_model.base_model, "tail_emb"):
                lora_model.base_model.tail_emb.requires_grad = True # prevent from being set to False in get_peft_model

            # if model_args.plus_one_token:
                # lora_model = TailTokenDetachPrefixWrapper(lora_model, merged=False, freeze_text_embeddings=True, tail_token_train_only=model_args.tail_token_train_only, tail_gradient_flow_only=model_args.tail_gradient_flow_only)
            

            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )



        model.qry_chosen_layer = model_args.qry_chosen_layer
        model.tgt_chosen_layer = model_args.tgt_chosen_layer

        
        return model


    @classmethod
    def load(cls, model_args: ModelArguments, is_trainable=True, **kwargs):
        # Loading the base model
        model_name_or_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        if not hasattr(model_args, "model_backbone") or not model_args.model_backbone:
            model_backbone = get_backbone_name(hf_config=config, model_type=model_args.model_type)
            setattr(model_args, 'model_backbone', model_backbone)
        print_master(f'Loading backbone [{model_args.model_backbone}] from {model_name_or_path}')
        if model_args.model_backbone in {LLAVA_NEXT, QWEN2_VL, QWEN2_5_VL, QWEN2_VL_TOKENSELECTION, QWEN2_5_VL_TOKENSELECTION, E5_V}:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config._attn_implementation = model_args.attn_implementation
            config.vision_config._attn_implementation = model_args.attn_implementation

            config.delete_L = model_args.delete_L
            config.delete_n = model_args.delete_n
            
            base_model = backbone2model[model_args.model_backbone].from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                config=config
            )

        elif model_args.model_backbone == PHI3V:
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            config.padding_side = "right"
            base_model = Phi3VForCausalLM.from_pretrained(model_args.model_name, **kwargs, config=config,
                                                          torch_dtype=torch.bfloat16, trust_remote_code=True)
            base_model.padding_side = "right"
        elif model_args.model_backbone == INTERNVIDEO2:
            print_master(f'Loading backbone [{model_args.model_backbone}] from {"src/model/vlm_backbone/internvideo2/"}')
            config = AutoConfig.from_pretrained("src/model/vlm_backbone/internvideo2/",
                                                trust_remote_code=True)
            base_model = backbone2model[model_args.model_backbone].from_pretrained("src/model/vlm_backbone/internvideo2/", config=config,
                                                                                   trust_remote_code=True)
        elif model_args.model_backbone == GME:
            base_model = GmeQwen2VL(model_args.model_name, processor=kwargs['processor'])
            config._attn_implementation = model_args.attn_implementation
            config.vision_config._attn_implementation = model_args.attn_implementation
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == LamRA:
            base_model = LamRAQwen2VL(model_args.model_name)
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == LamRA_QWEN2_5:
            base_model = LamRAQwen25VL(model_args.model_name)
            setattr(base_model, 'config', config)
        elif model_args.model_backbone == COLPALI:
            config._attn_implementation = model_args.attn_implementation
            base_model = ColPali.from_pretrained(model_args.model_name)
            setattr(base_model, 'config', config)
        else:
            # Loading external base model from HF
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            base_model = cls.TRANSFORMER_CLS.from_pretrained(
                model_name_or_path, **kwargs, config=config,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True)

        # Building the model on top of the base
        if model_args.lora:
            print_master(f'Loading LoRA from {model_name_or_path}')
            lora_config = LoraConfig.from_pretrained(model_name_or_path)
            lora_model = PeftModel.from_pretrained(base_model, model_name_or_path, config=lora_config, is_trainable=is_trainable)
            lora_model.load_adapter(model_name_or_path, lora_model.active_adapter, is_trainable=is_trainable)
            if not is_trainable:
                lora_model = lora_model.merge_and_unload()
            # for name, param in lora_model.named_parameters():
            #     print_master(name)
            #     print_master(param.data)
            # print_master(lora_model)
                                
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

            
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature
            )

        model.model_backbone = model_args.model_backbone
        model.qry_chosen_layer = model_args.qry_chosen_layer
        model.tgt_chosen_layer = model_args.tgt_chosen_layer
        # print_master(model.encoder)
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(self, qry: Dict[str, Tensor] = None, tgt: Dict[str, Tensor] = None, *args, **kwargs):

        qry_reps = self.encode_input(qry, self.qry_chosen_layer) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt, self.tgt_chosen_layer) if tgt else None # (bsz_per_device, dim)

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry_reps = self._dist_gather_tensor(qry_reps)
            all_tgt_reps = self._dist_gather_tensor(tgt_reps)
        else:
            all_qry_reps = qry_reps
            all_tgt_reps = tgt_reps

        scores = self.compute_similarity(all_qry_reps, all_tgt_reps)
        scores = scores.view(all_qry_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (all_qry_reps.size(0) // all_tgt_reps.size(0))
        loss = self.cross_entropy(scores / self.temperature, target)
        if self.is_ddp:
            loss = loss * self.world_size
        return loss

    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))