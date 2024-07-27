from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gemma2 import ModelArgs, TransformerBlock, RMSNorm
from .base import IdentityBlock

@dataclass
class ModelArgs(ModelArgs):
    start_layer: int = 0
    end_layer: int = 46

class GemmaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.start_layer = args.start_layer
        self.end_layer = args.end_layer
        assert self.vocab_size > 0
        if self.start_layer == 0 or self.end_layer == self.num_hidden_layers:
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

        self.layers = []
        for i in range(self.num_hidden_layers):
            if self.start_layer <= i < self.end_layer:
                self.layers.append(TransformerBlock(args=args))
            else:
                self.layers.append(IdentityBlock())

        if self.end_layer == self.num_hidden_layers:
            self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        if self.start_layer == 0:
            h = self.embed_tokens(inputs)
            h = h * (self.args.hidden_size**0.5)
        else:
            h = inputs
        

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        if self.end_layer == self.num_hidden_layers:
            h = self.norm(h)
        return h

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.start_layer = args.start_layer
        self.end_layer = args.end_layer
        self.final_logit_softcapping = args.final_logit_softcapping
        self.model = GemmaModel(args)
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)

        if self.end_layer == self.args.num_hidden_layers:
            out = self.model.embed_tokens.as_linear(out)
            out = mx.tanh(out / self.final_logit_softcapping)
            out = out * self.final_logit_softcapping
            return out
        else:
            return out

    def sanitize(self, weights):
        total_layers = len(self.layers)
        shard_state_dict = {}
        for key, value in weights.items():
            if "self_attn.rotary_emb.inv_freq" in key:
                continue
            if key.startswith('model.layers.'):
                layer_num = int(key.split('.')[2])
                if self.start_layer <= layer_num < self.end_layer:
                    shard_state_dict[key] = value
            elif (self.start_layer == 0 or self.end_layer == total_layers)  and key.startswith('model.embed_tokens'):
                shard_state_dict[key] = value
            elif self.end_layer == total_layers and (key.startswith('model.norm') or key.startswith('lm_head')):
                shard_state_dict[key] = value
        
        return shard_state_dict
    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads