

import torch
from torch import nn
from typing import Optional
from typing import Optional, Any, Tuple


from src.utils import print_master

class TailTokenWrapper(nn.Module):
    """
    Wrap a Hugging Face vision-language model (e.g., Qwen2VLModel).
    This wrapper appends a learnable token embedding at the end of the text sequence.
    If `freeze_base=True`, only the new token embedding is trainable.

    Key assumptions:
    - The wrapped model exposes `.model.embed_tokens` for text embeddings.
    - The caller will pool the last position (which now corresponds to the tail token).
    """

    def __init__(self, base_model: nn.Module, hidden_size: int, freeze_base: bool = True):
        super().__init__()
        self.base = base_model
        # Trainable tail token [1, 1, D]
        self.tail_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = True

    @property
    def device(self):
        return next(self.base.parameters()).device

    @property
    def config(self):
        # Forward config to make wrapper compatible with PEFT / Hugging Face ecosystem
        return getattr(self.base, "config", None)

    def __getattr__(self, name):
        """Forward attribute access to the base model when not found."""
        if name in ("base", "tail_token"):
            return super().__getattr__(name)
        try:
            return getattr(self.base, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs
    ):

        # 1) Build embeddings if input_ids are provided
        if inputs_embeds is None:
            inputs_embeds = self.base.model.embed_tokens(input_ids)

        # 2) Append the tail token
        B = inputs_embeds.size(0)
        tail = self.tail_token.expand(B, 1, -1).to(inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = torch.cat([inputs_embeds, tail], dim=1)

        # 3) Extend attention mask
        if attention_mask is not None:
            add = torch.ones(B, 1, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, add], dim=1)


        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict=return_dict if return_dict is not None else self.config.use_return_dict,

        # 4) Pass through the base model
        outputs = self.base.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        return outputs


class TailTokenDetachPrefixWrapper(nn.Module):
    """
    A simplified wrapper for a Hugging Face VLM/LLM model.
    - Adds one learnable tail token at the end of text embeddings.
    - Prefix (all other tokens) are treated as 'dead': no gradient flows through them.
    - Model parameters remain trainable, gradient flows through the tail path.

    No KV cache is used here.
    """

    def __init__(self, base_model: nn.Module, tail_embedding=None, merged=False, freeze_text_embeddings: bool = True):
        super().__init__()
        self.base = base_model
        self.merged = merged
        print_master(self.base)
        if self.merged: 
            self.hidden_size = self.base.model.embed_tokens.weight.shape[-1]
        else:        
            self.hidden_size = self.base.base_model.model.model.embed_tokens.weight.shape[-1]

            if not freeze_text_embeddings:
                for p in self.base.base_model.model.model.embed_tokens.parameters():
                    p.requires_grad = True
            else:
                for p in self.base.base_model.model.model.embed_tokens.parameters():
                    p.requires_grad = False
        
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

    # def __getattr__(self, name):
    #     if name in ("base", "tail_token"):
    #         return super().__getattr__(name)
    #     try:
    #         return getattr(self.base, name)
    #     except AttributeError:
    #         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    # # debug
    # def forward(self, **kwargs):
    #     if "input_ids" in kwargs and "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is None:
    #         inputs_embeds = self.base.base_model.model.model.embed_tokens(kwargs["input_ids"])
    #         del kwargs["input_ids"]
    #         kwargs["inputs_embeds"] = inputs_embeds
    #     if "output_hidden_states" not in kwargs or kwargs["output_hidden_states"] is None:
    #         kwargs["output_hidden_states"] = getattr(self.config, "output_hidden_states", False)
    #     if "return_dict" not in kwargs or kwargs["return_dict"] is None:
    #         kwargs["return_dict"] = getattr(self.config, "use_return_dict", True)

    #     outputs = self.base(**kwargs)

    #     return outputs


    def forward(self, **kwargs):
        # 1) Build text embeddings if input_ids provided

        if "input_ids" in kwargs and "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is None:
            inputs_embeds = self.base.base_model.model.model.embed_tokens(kwargs["input_ids"]) if not self.merged else self.base.model.embed_tokens(kwargs["input_ids"])
            del kwargs["input_ids"]

            B = kwargs["inputs_embeds"].shape[0]

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
            if attention_mask is not None:
                add = torch.ones(B, 1, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, add], dim=1)
            kwargs["attention_mask"] = attention_mask

        if "output_hidden_states" not in kwargs or kwargs["output_hidden_states"] is None:
            kwargs["output_hidden_states"] = getattr(self.config, "output_hidden_states", False)
        if "return_dict" not in kwargs or kwargs["return_dict"] is None:
            kwargs["return_dict"] = getattr(self.config, "use_return_dict", True)

        
        # 5) Forward through base model (normal)
        outputs = self.base(**kwargs)
        # print_master("OUTPUTS:")
        # print_master(len(outputs.hidden_states))
        # print_master(outputs.hidden_states[0].shape)
        return outputs


class TailIsolatedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, **kwargs):
        prefix, tail = hidden_states[:, :-1, :], hidden_states[:, -1:, :]
        prefix = prefix.detach()   # stop gradient flow through prefix
        hidden_states = torch.cat([prefix, tail], dim=1)
        return self.block(hidden_states, **kwargs)
