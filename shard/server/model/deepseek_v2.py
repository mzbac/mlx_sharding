from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import KVCache
from mlx_lm.models.deepseek_v2 import ModelArgs, DeepseekV2DecoderLayer
from .base import IdentityBlock

@dataclass
class ModelArgs(ModelArgs):
    start_layer: int = 0
    end_layer: int = 27

class DeepseekV2Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_hidden_layers = config.num_hidden_layers
        self.start_layer = config.start_layer
        self.end_layer = config.end_layer
        self.vocab_size = config.vocab_size
        if self.start_layer == 0:
            self.embed_tokens = nn.Embedding(
                config.vocab_size, config.hidden_size)

        self.layers = []
        for i in range(self.num_hidden_layers):
            if self.start_layer <= i < self.end_layer:
                self.layers.append(DeepseekV2DecoderLayer(config, i))
            else:
                self.layers.append(IdentityBlock())

        if self.end_layer == self.num_hidden_layers:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        if self.start_layer == 0:
            h = self.embed_tokens(x)
        else:
            h = x

        mask = None
        T = h.shape[1]
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        if self.end_layer == self.num_hidden_layers:
            h = self.norm(h)
        return h


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.start_layer = config.start_layer
        self.end_layer = config.end_layer
        self.model = DeepseekV2Model(config)
        if self.end_layer == self.args.num_hidden_layers:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[KVCache] = None,
    ):
        out = self.model(inputs, cache)
        if self.end_layer == self.args.num_hidden_layers:
            return self.lm_head(out)
        return out

    def sanitize(self, weights):
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            for n, m in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")]:
                for k in ["weight", "scales", "biases"]:
                    if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(self.args.n_routed_experts)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{
                            m}.{k}"] = mx.stack(to_join)
        return weights

    @ property
    def layers(self):
        return self.model.layers

    @ property
    def head_dim(self):
        return (
            self.args.qk_nope_head_dim + self.args.qk_rope_head_dim,
            self.args.v_head_dim,
        )

    @ property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
