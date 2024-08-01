
import grpc
from concurrent import futures
from ..grpc import mlx_tensor_pb2, mlx_tensor_pb2_grpc
from ..utils import bytes_to_tensor, load_model, tensor_to_bytes
import mlx.core as mx
from mlx_lm.models.base import KVCache

MODEL = None
CACHE = None

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
            tensor = bytes_to_tensor(request.tensor_data, request.dtype)
            tensor = mx.reshape(tensor, request.shape)
            print(f"Received tensor with shape: {
                  tensor.shape} and dtype: {tensor.dtype}")

            if MODEL is not None:
                processed_tensor = MODEL(tensor, cache=CACHE)
                print(f"Processed tensor with shape: {
                      processed_tensor.shape} and dtype: {processed_tensor.dtype}")
                processed_bytes = tensor_to_bytes(processed_tensor)
                response_tensor = mlx_tensor_pb2.Tensor(
                    tensor_data=processed_bytes,
                    shape=list(processed_tensor.shape),
                    dtype=str(processed_tensor.dtype)
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


def serve(model_path, start_layer=None, end_layer=None):
    global MODEL
    MODEL = load_model(model_path, start_layer=start_layer, end_layer=end_layer)
    reset_cache()
    server_options = [
        ('grpc.max_metadata_size', 32 * 1024 * 1024),
        ('grpc.max_send_message_length', 1280 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1280 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(
        max_workers=10), options=server_options)
    mlx_tensor_pb2_grpc.add_MLXTensorServiceServicer_to_server(
        MLXTensorServicer(), server)

    port = server.add_insecure_port('[::]:0')
    server.start()
    print(f"Server started, listening on 0.0.0.0:{port}")
    if start_layer is not None or end_layer is not None:
        print(f"Model loaded with layers {start_layer or 0} to {end_layer or 'end'}")
    server.wait_for_termination()

