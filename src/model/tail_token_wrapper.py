

import torch
from torch import nn
from typing import Optional
from typing import Optional, Any, Tuple


from src.utils import print_master

class TailTokenDetachPrefixWrapper(nn.Module):
    """
    A simplified wrapper for a Hugging Face VLM/LLM model.
    - Adds one learnable tail token at the end of text embeddings.
    - Prefix (all other tokens) are treated as 'dead': no gradient flows through them.
    - Model parameters remain trainable, gradient flows through the tail path.

    No KV cache is used here.
    """

    def __init__(self, base_model: nn.Module, tail_embedding=None, merged=False, freeze_text_embeddings: bool = True, tail_token_train_only: bool = False, tail_gradient_flow_only: bool = False):
        super().__init__()
        self.base = base_model
        self.merged = merged
        print_master(self.base)
        if self.merged: # eval stage
            self.hidden_size = self.base.model.embed_tokens.weight.shape[-1]
        else: # train stage
            self.hidden_size = self.base.base_model.model.model.embed_tokens.weight.shape[-1]

            if tail_token_train_only:
                for n, p in self.base.named_parameters():
                    p.requires_grad = False
                    print_master("Freezing all base model parameters except tail token embedding.")

            if tail_gradient_flow_only:
                for i, block in enumerate(self.base.base_model.model.model.layers):
                    self.base.base_model.model.model.layers[i] = TailIsolatedBlock(block)
                print_master("Gradient will only flow through the tail token embedding.")

            if not freeze_text_embeddings:
                for p in self.base.base_model.model.model.embed_tokens.parameters():
                    p.requires_grad = True
            
        
        if tail_embedding is not None:
            self.tail_token = tail_embedding
        else: # random initialization
            self.tail_token = nn.Parameter(0.01 * torch.randn(1, 1, self.hidden_size))
            self.tail_token.requires_grad = True       


    @property
    def device(self):
        return next(self.base.parameters()).device

    @property
    def config(self):
        return getattr(self.base, "config", None)



    def forward(self, **kwargs):
        # 1) Build text embeddings if input_ids provided

        if kwargs.get("inputs_embeds", None) is None:
            print_master("input embeds is None and input ids are:")
            print_master(kwargs.get("input_ids", None))
            inputs_embeds = self.base.base_model.model.model.embed_tokens(kwargs["input_ids"]) if not self.merged else self.base.model.embed_tokens(kwargs["input_ids"])
            del kwargs["input_ids"]
        else:
            print_master("input embeds is not None")
            inputs_embeds = kwargs["inputs_embeds"]

        B = inputs_embeds.shape[0]

        # 2) Detach prefix embeddings (no gradient through them)
        # prefix_embeds = inputs_embeds.detach()
        prefix_embeds = inputs_embeds
        # 3) Append trainable tail token (this one carries gradient)
        tail = self.tail_token.expand(B, 1, -1).to(prefix_embeds.device, dtype=prefix_embeds.dtype)
        all_embeds = torch.cat([prefix_embeds, tail], dim=1)

        kwargs["inputs_embeds"] = all_embeds

        # print_master("ALL_EMBEDS:")
        # print_master(all_embeds.shape)

        # 4) Extend attention mask
        attention_mask = kwargs.get("attention_mask", None)
        if attention_mask is not None:
            add = torch.ones(B, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, add], dim=1)
        kwargs["attention_mask"] = attention_mask

        if "output_hidden_states" not in kwargs or kwargs["output_hidden_states"] is None:
            kwargs["output_hidden_states"] = getattr(self.config, "output_hidden_states", False)
        if "return_dict" not in kwargs or kwargs["return_dict"] is None:
            kwargs["return_dict"] = getattr(self.config, "use_return_dict", True)

        # print for every step one of the model weights to verify training
        # print_master("Model weight sample:")
        # print_master(self.base.base_model.model.model.layers[-1].block.self_attn.q_proj.weight.flatten()[0].item())
        # print_master(self.base.base_model.model.model.layers[-1].self_attn.q_proj.lora_A.default.weight.flatten())
        # print_master("Tail token:")
        # print_master(self.tail_token.flatten())

        
        # 5) Forward through base model (normal)
        outputs = self.base(**kwargs)
        # print_master("OUTPUTS:")
        # print_master(len(outputs.hidden_states))
        # print_master(outputs.hidden_states[0].shape)
        return outputs


# class TailIsolatedBlock(nn.Module):
#     def __init__(self, block):
#         super().__init__()
#         self.block = block

#     def forward(self, hidden_states, **kwargs):
#         prefix, tail = hidden_states[:, :-1, :], hidden_states[:, -1:, :]
#         prefix = prefix.detach()   # stop gradient flow through prefix
#         hidden_states = torch.cat([prefix, tail], dim=1)
#         return self.block(hidden_states, **kwargs)

class Customed_Forward_Backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Create and save a mask that only allows gradient on the last token
        grad_mask = torch.zeros_like(x)
        grad_mask[:, -1:, :] = 1.0
        ctx.save_for_backward(grad_mask)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (grad_mask,) = ctx.saved_tensors
        return grad_output * grad_mask


class TailIsolatedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, hidden_states, **kwargs):
        # Apply the custom forward-backward operation
        hidden_states = Customed_Forward_Backward.apply(hidden_states)
        return self.block(hidden_states, **kwargs)

