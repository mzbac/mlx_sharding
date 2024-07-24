import argparse
from typing import Tuple
import grpc
from shard.grpc import mlx_tensor_pb2, mlx_tensor_pb2_grpc
from transformers import AutoTokenizer
import numpy as np
import mlx.core as mx
from shard.utils import load_model
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.models.base import KVCache
import time


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text using a specified model.")
    parser.add_argument("--model", type=str, default="shard_0",
                        help="Path or name of the model to use")
    parser.add_argument("--prompt", type=str, default="how to write quicksort in python",
                        help="Prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--server_address", type=str, default="localhost:50908",
                        help="Address of the gRPC server")
    return parser.parse_args()


def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model)

    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    channel_options = [
        ('grpc.max_metadata_size', 32 * 1024 * 1024),
        ('grpc.max_send_message_length', 128 * 1024 * 1024),
        ('grpc.max_receive_message_length', 128 * 1024 * 1024),
    ]
    channel_1 = grpc.insecure_channel(
        args.server_address, options=channel_options)
    stub_1 = mlx_tensor_pb2_grpc.MLXTensorServiceStub(channel_1)

    reset_response = stub_1.ResetCache(mlx_tensor_pb2.ResetCacheRequest())
    print("ResetCache Response:", reset_response.message)

    for t in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens, stub=stub_1):
        print(t, end="", flush=True)
    print()


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


def generate_step(prompt, model, stub):
    def sample(logits: mx.array) -> Tuple[mx.array, float]:
        return mx.argmax(logits, axis=-1)

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
        if output.dtype == mx.bfloat16:
            output = output.astype(mx.float16)
        response = send_tensor(stub, np.array(output))
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


def stream_generate(model, tokenizer, prompt: str, max_tokens: int = 100, stub=None, **kwargs):
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    tic = time.perf_counter()
    detokenizer.reset()
    for token, n in zip(
        generate_step(prompt_tokens, model, stub, **kwargs),
        range(max_tokens),
    ):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            tic = time.perf_counter()
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)
        yield detokenizer.last_segment

    token_count = n + 1
    detokenizer.finalize()
    yield detokenizer.last_segment
    gen_time = time.perf_counter() - tic
    print("=" * 10)
    if token_count == 0:
        print("No tokens generated for this prompt")
        return
    prompt_tps = prompt_tokens.size / prompt_time
    gen_tps = (token_count - 1) / gen_time
    print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
    print(f"Generation: {gen_tps:.3f} tokens-per-sec")


if __name__ == "__main__":
    main()
