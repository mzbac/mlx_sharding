import argparse
from typing import Tuple, List
import grpc
from shard.grpc import mlx_tensor_pb2, mlx_tensor_pb2_grpc
from transformers import AutoTokenizer
import mlx.core as mx
from shard.utils import load_model, response_to_mlx_array, send_tensor
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.models.base import KVCache
import time

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate text using a specified model.")
    parser.add_argument("--model", type=str, default="shard_0", help="Path or name of the model to use")
    parser.add_argument("--prompt", type=str, default="how to write quicksort in python", help="Prompt for text generation")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--server_address", type=str, default="localhost:50051", help="Comma-separated addresses of the gRPC servers")
    parser.add_argument("--start_layer", type=int, default=None, help="Start layer for dynamic sharding")
    parser.add_argument("--end_layer", type=int, default=None, help="End layer for dynamic sharding")
    return parser.parse_args()

def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, start_layer=args.start_layer, end_layer=args.end_layer)

    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    channel_options = [
        ('grpc.max_metadata_size', 32 * 1024 * 1024),
        ('grpc.max_send_message_length', 128 * 1024 * 1024),
        ('grpc.max_receive_message_length', 128 * 1024 * 1024),
    ]
    
    server_addresses = args.server_address.split(',')
    stubs = []
    for address in server_addresses:
        channel = grpc.insecure_channel(address.strip(), options=channel_options)
        stub = mlx_tensor_pb2_grpc.MLXTensorServiceStub(channel)
        stubs.append(stub)

    for stub in stubs:
        reset_response = stub.ResetCache(mlx_tensor_pb2.ResetCacheRequest())
        print(f"ResetCache Response for {stub}: {reset_response.message}")

    for t in stream_generate(model, tokenizer, prompt, max_tokens=args.max_tokens, stubs=stubs):
        print(t, end="", flush=True)
    print()

def generate_step(prompt, model, stubs: List[mlx_tensor_pb2_grpc.MLXTensorServiceStub]):
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
        
        for i, stub in enumerate(stubs):
            response = send_tensor(stub, output)
            output = response_to_mlx_array(response.tensor)
            if i == len(stubs) - 1:  # Last stub
                logits = output[:, -1, :]
                y = sample(logits)
                return y
        
        raise ValueError("No valid response from any stub")

    y = _step(y)
    mx.async_eval(y)
    while True:
        next_y = _step(y)
        mx.async_eval(next_y)
        yield y.item()
        y = next_y

def stream_generate(model, tokenizer, prompt: str, max_tokens: int = 100, stubs=None, **kwargs):
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    tic = time.perf_counter()
    detokenizer.reset()
    for token, n in zip(
        generate_step(prompt_tokens, model, stubs, **kwargs),
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