import glob
from importlib import import_module
import json
import grpc
from concurrent import futures
import numpy as np
import mlx_tensor_pb2
import mlx_tensor_pb2_grpc
import mlx.core as mx
from mlx_lm.utils import get_model_path
import mlx.nn as nn
import argparse
from mlx_lm.models.base import KVCache


MODEL = None
CACHE = None

MODEL_REMAPPING = {
    "mistral": "llama",  # mistral is compatible with llama
    "phi-msft": "phixtral",
}


def _get_classes(config: dict):
    model_type = config["model_type"]
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    try:
        arch = import_module(f".model.{model_type}", package="server")
    except ImportError:
        msg = f"Model type {model_type} not supported."
        print(msg)
        raise ValueError(msg)

    return arch.Model, arch.ModelArgs


def load_model(path_or_hf_repo: str):
    path = get_model_path(path_or_hf_repo)
    with open(path / "config.json", "r") as f:
        config = json.load(f)
    weight_files = glob.glob(str(path / "*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {path}")
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))
    model_class, model_args_class = _get_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)
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


def reset_cache():
    global CACHE
    if hasattr(MODEL, "make_cache"):
        CACHE = MODEL.make_cache()
    else:
        kv_heads = (
            [MODEL.n_kv_heads] * len(MODEL.layers)
            if isinstance(MODEL.n_kv_heads, int)
            else MODEL.n_kv_heads
        )
        CACHE = [KVCache(MODEL.head_dim, n) for n in kv_heads]
    print("Cache has been reset")


class MLXTensorServicer(mlx_tensor_pb2_grpc.MLXTensorServiceServicer):
    def SendTensor(self, request, context):
        try:
            print("Received tensor request")
            dtype_str = request.dtype
            dtype_map = {
                'float32': np.float32,
                'int32': np.int32,
                'float64': np.float64,
                'int64': np.int64,
                "float16": np.float16,
            }
            np_dtype = dtype_map.get(dtype_str, np.float32)
            np_array = np.frombuffer(request.tensor_data, dtype=np_dtype)
            mx_dtype = getattr(mx, dtype_str, mx.float32)
            tensor = mx.array(np_array, dtype=mx_dtype)
            tensor = mx.reshape(tensor, request.shape)
            print(f"Received tensor with shape: {
                  tensor.shape} and dtype: {tensor.dtype}")

            if MODEL is not None:
                processed_tensor = MODEL(tensor, cache=CACHE)
                print(f"Processed tensor with shape: {
                      processed_tensor.shape} and dtype: {processed_tensor.dtype}")
                processed_np = np.array(processed_tensor)
                processed_bytes = processed_np.tobytes()
                response_tensor = mlx_tensor_pb2.Tensor(
                    tensor_data=processed_bytes,
                    shape=list(processed_np.shape),
                    dtype=str(processed_np.dtype)
                )
                return mlx_tensor_pb2.TensorResponse(
                    success=True,
                    message="Tensor processed successfully",
                    tensor=response_tensor)
            else:
                return mlx_tensor_pb2.TensorResponse(
                    success=False,
                    message="Model not loaded",
                    tensor=None
                )
        except Exception as e:
            print(f"Error processing tensor: {e}")
            return mlx_tensor_pb2.TensorResponse(success=False, message=str(e))

    def ResetCache(self, request, context):
        try:
            reset_cache()
            return mlx_tensor_pb2.ResetCacheResponse(
                success=True,
                message="Cache reset successfully"
            )
        except Exception as e:
            print(f"Error resetting cache: {e}")
            return mlx_tensor_pb2.ResetCacheResponse(
                success=False,
                message=f"Error resetting cache: {str(e)}"
            )


def serve(model_path):
    global MODEL
    MODEL = load_model(model_path)
    reset_cache()
    server_options = [
        ('grpc.max_metadata_size', 32 * 1024 * 1024),
        ('grpc.max_send_message_length', 128 * 1024 * 1024),
        ('grpc.max_receive_message_length', 128 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(
        max_workers=10), options=server_options)
    mlx_tensor_pb2_grpc.add_MLXTensorServiceServicer_to_server(
        MLXTensorServicer(), server)

    port = server.add_insecure_port('[::]:0')
    server.start()
    print(f"Server started, listening on 0.0.0.0:{port}")
    server.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLX Tensor Server")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model or HuggingFace repo")
    args = parser.parse_args()

    serve(args.model)
