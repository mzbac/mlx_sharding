from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_additive_causal_mask
from mlx_lm.models.llama import TransformerBlock, ModelArgs
from .base import IdentityBlock


@dataclass
class ModelArgs(ModelArgs):
    start_layer: int = 0
    end_layer: int = 32


class LlamaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.start_layer = args.start_layer
        self.end_layer = args.end_layer
        assert self.vocab_size > 0
        if self.start_layer == 0:
            self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = []
        for i in range(self.num_hidden_layers):
            if self.start_layer <= i < self.end_layer:
                self.layers.append(TransformerBlock(args=args))
            else:
                self.layers.append(IdentityBlock())

        if self.end_layer == self.num_hidden_layers:
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        if self.start_layer == 0:
            h = self.embed_tokens(inputs)
        else:
            h = inputs

        mask = None
        if h.shape[1] > 1:
            mask = create_additive_causal_mask(
                h.shape[1], cache[0].offset if cache is not None else 0
            )
            mask = mask.astype(h.dtype)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        if self.end_layer == self.num_hidden_layers:
            h = self.norm(h)
        return h


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.start_layer = args.start_layer
        self.end_layer = args.end_layer
        self.model = LlamaModel(args)
        if self.end_layer == self.args.num_hidden_layers:
            if not args.tie_word_embeddings:
                self.lm_head = nn.Linear(
                    args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        if self.end_layer == self.args.num_hidden_layers:
            if self.args.tie_word_embeddings:
                out = self.model.embed_tokens.as_linear(out)
            else:
                out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        return {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads
