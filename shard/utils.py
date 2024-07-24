from importlib import import_module
import glob
import json
import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Generator, Optional, Tuple, List
from mlx_lm.models.base import KVCache
from mlx_lm.sample_utils import top_p_sampling
from mlx_lm.utils import apply_repetition_penalty, get_model_path
import numpy as np
from .grpc import mlx_tensor_pb2

MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
}

def _get_classes(config: dict):
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = import_module(f".model.{model_type}", package="shard.server")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        print(msg)
        raise ValueError(msg)

    return arch.Model, arch.ModelArgs


def load_model(path_or_hf_repo: str, start_layer: int = None, end_layer: int = None):
    path = get_model_path(path_or_hf_repo)
    with open(path / "config.json", "r") as f:
        config = json.load(f)
        if start_layer is not None and end_layer is not None:
            config['start_layer'] = start_layer
            config['end_layer'] = end_layer
    weight_files = glob.glob(str(path / "*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {path}")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    model_class, model_args_class = _get_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
    total_layers = len(model.layers)

    if start_layer is not None and end_layer is not None:
        shard_state_dict = {}
        for key, value in weights.items():
            if key.startswith('model.layers.'):
                layer_num = int(key.split('.')[2])
                if start_layer <= layer_num < end_layer:
                    shard_state_dict[key] = value
            elif start_layer == 0 and key.startswith('model.embed_tokens'):
                shard_state_dict[key] = value
            elif end_layer == total_layers and (key.startswith('model.norm') or key.startswith('lm_head')):
                shard_state_dict[key] = value
        
        weights = shard_state_dict

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    if (quantization := config.get("quantization", None)) is not None:
        def class_predicate(p, m):
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in weights
        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )
    model.load_weights(list(weights.items()))
    model.eval()
    return model


def send_tensor(stub, tensor: np.ndarray):
    tensor_bytes = tensor.tobytes()
    tensor_message = mlx_tensor_pb2.Tensor(
        tensor_data=tensor_bytes,
        shape=list(tensor.shape),
        dtype=str(tensor.dtype)
    )
    response = stub.SendTensor(tensor_message)
    return response


def response_to_mlx_array(response):
    try:
        tensor_data = response.tensor_data
        shape = response.shape
        dtype_str = response.dtype
        dtype_map = {
            'float32': np.float32,
            'int32': np.int32,
            'float64': np.float64,
            'int64': np.int64,
            "float16": np.float16,
        }
        np_dtype = dtype_map.get(dtype_str, np.float32)
        np_array = np.frombuffer(tensor_data, dtype=np_dtype).reshape(shape)
        mx_dtype = getattr(mx, dtype_str, mx.float32)
        tensor = mx.array(np_array, dtype=mx_dtype)
        return tensor
    except Exception as e:
        return None
    

def create_generate_step_with_grpc(grpc_stubs: List):
    def generate_step(
        prompt: mx.array,
        model: nn.Module,
        temp: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        top_p: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        
        for stub in grpc_stubs:
            reset_response = stub.ResetCache(mlx_tensor_pb2.ResetCacheRequest())
            print("ResetCache Response:", reset_response.message)

        def sample(logits: mx.array) -> Tuple[mx.array, float]:
            if logit_bias:
                indices = mx.array(list(logit_bias.keys()))
                values = mx.array(list(logit_bias.values()))
                logits[:, indices] += values
            logprobs = logits - mx.logsumexp(logits)
            if temp == 0:
                token = mx.argmax(logits, axis=-1)
            else:
                if top_p > 0 and top_p < 1.0:
                    token = top_p_sampling(logits, top_p, temp)
                else:
                    token = mx.random.categorical(logits * (1 / temp))
            return token, logprobs

        y = prompt
        if hasattr(model, "make_cache"):
            cache = model.make_cache()
        else:
            kv_heads = (
                [model.n_kv_heads] * len(model.layers)
                if isinstance(model.n_kv_heads, int)
                else model.n_kv_heads
            )
            cache = [KVCache(model.head_dim, n) for n in kv_heads]

        repetition_context = prompt.tolist()
        if repetition_context_size:
            repetition_context = repetition_context[-repetition_context_size:]

        def _step(y):
            nonlocal repetition_context
            output = model(y[None], cache=cache)
            if output.dtype == mx.bfloat16:
                output = output.astype(mx.float16)

            for stub in grpc_stubs:
                response = send_tensor(stub, np.array(output))
                output = response_to_mlx_array(response.tensor)
            
            logits = output[:, -1, :]
            if repetition_penalty:
                logits = apply_repetition_penalty(
                    logits, repetition_context, repetition_penalty
                )
                y, logprobs = sample(logits)
                repetition_context.append(y.item())
            else:
                y, logprobs = sample(logits)
            if repetition_context_size:
                if len(repetition_context) > repetition_context_size:
                    repetition_context = repetition_context[-repetition_context_size:]
            return y, logprobs.squeeze(0)

        y, logprobs = _step(y)
        mx.async_eval(y)
        while True:
            next_y, next_logprobs = _step(y)
            mx.async_eval(next_y)
            yield y.item(), logprobs
            y, logprobs = next_y, next_logprobs

    return generate_step