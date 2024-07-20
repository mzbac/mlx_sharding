from typing import Tuple
import grpc
import mlx_tensor_pb2
import mlx_tensor_pb2_grpc
from transformers import AutoTokenizer
import numpy as np
import mlx.core as mx
from server.server import load_model
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.models.base import KVCache

tokenizer = AutoTokenizer.from_pretrained("shard_0")
model = load_model("shard_0")

prompt = "how to write quicksort in python"

# update the address and port of the first shard
channel_1 = grpc.insecure_channel('localhost:49200')
stub_1 = mlx_tensor_pb2_grpc.MLXTensorServiceStub(channel_1)

# update the address and port of the second shard
channel_2 = grpc.insecure_channel('localhost:51998')
stub_2 = mlx_tensor_pb2_grpc.MLXTensorServiceStub(channel_2)


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

        np_array = np.frombuffer(tensor_data, dtype=np_dtype)

        np_array = np_array.reshape(shape)

        mx_dtype = getattr(mx, dtype_str, mx.float32)
        tensor = mx.array(np_array, dtype=mx_dtype)

        return tensor
    except Exception as e:
        return None


def generate_step(
    prompt,
    model,
):
    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        token = mx.argmax(logits, axis=-1)

        return token

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

    def _step(y):
        output = model(y[None], cache=cache)
        # Send the output tensor to the shard 1 server
        response = send_tensor(stub_1, np.array(output))
        output = response_to_mlx_array(response.tensor)
        # Send the output tensor to the shard 2 server assume shard 2 is the last shard
        response = send_tensor(stub_2, np.array(output))
        logits = response_to_mlx_array(response.tensor)

        logits = logits[:, -1, :]

        y = sample(logits)
        return y

    y = _step(y)

    mx.async_eval(y)
    while True:
        next_y = _step(y)
        mx.async_eval(next_y)
        yield y.item()
        y = next_y


def stream_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    **kwargs,
):
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()
    for token, n in zip(
        generate_step(prompt_tokens, model, **kwargs),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)

        # Yield the last segment if streaming
        yield detokenizer.last_segment

    detokenizer.finalize()
    yield detokenizer.last_segment


for t in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(t, end="", flush=True)
print()
