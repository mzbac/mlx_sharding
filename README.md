# MLX Pipeline Parallelism Demo

This project demonstrates how to implement pipeline parallelism for large language models using MLX. It includes tools for sharding a model, serving shards across multiple machines, and generating text using the distributed model. Additionally, it now features an OpenAI API-compatible server for easier integration and usage.

## Demo Video

To see the distributed inference in action, check out our demo video:

[Sharding DeepSeek-Coder-V2-Lite-Instruct Demo](https://www.youtube.com/watch?v=saOboSfP76o)

## Educational Purpose

This repository is designed for educational purposes to illustrate how pipeline parallelism can be implemented in MLX. It provides a basic framework for:

1. Sharding a large language model
2. Distributing model shards across multiple machines
3. Implementing a simple pipeline for text generation
4. Serving the model through an OpenAI API-compatible interface

While not optimized for production use, this demo serves as a starting point for understanding and experimenting with pipeline parallelism in machine learning workflows.

## Setup and Usage

### 1. Shard the Model

Use `sharding_weight.py` to split your model into multiple shards:

```bash
python sharding_weight.py --model "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx" --output_dir shard_0 --start_layer 0 --end_layer 14 --total_layers 27
python sharding_weight.py --model "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx" --output_dir shard_1 --start_layer 14 --end_layer 27 --total_layers 27
# Repeat for additional shards as needed
```

This will create directories (`shard_0`, `shard_1`, `shard_2`, etc.) containing the sharded model weights and configurations.

### 2. Distribute Shards

Copy the shard directories to their respective machines:

- Local machine (for generate script): Keep `shard_0`
- Machine 1: Copy `shard_1`
- Machine 2: Copy `shard_2`

### 3. Start the Servers

On each remote machine, start a server instance for its respective shard. For example, to start the server for shard 1:

```bash
python -m shard.main --model mzbac/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx-shard-1
```

Note the IP address and port printed by the server.

### 4. Generate Text

#### Using the generate script

Run the generate script on the local machine with the desired arguments:

```bash
python generate.py --model mzbac/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx-shard-0 --server_address <remote_ip>:<port> --prompt "Your prompt here" --max_tokens 512
```

#### Using the OpenAI API-compatible server

To use the OpenAI API-compatible server:

1. Start the server:

   ```bash
   python -m shard.openai_api --model /path/to/your/model --llm-shard-addresses localhost:50051,<remote_ip1>:<port1>,<remote_ip2>:<port2>
   ```

2. Use the API endpoints:
   - `/v1/completions`: Text completion endpoint
   - `/v1/chat/completions`: Chat completion endpoint

Example usage:

```bash
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

## Limitations and Considerations

1. **Fixed Shard Configuration**: The generate script is currently set up for three shards (one local, two remote). To use more shards, you'll need to modify the script to include additional gRPC channels and adjust the `generate_step` function accordingly.

2. **Network Dependency**: The performance of this pipeline parallelism implementation is heavily dependent on network speed and latency between machines.

3. **Error Handling**: The current implementation has basic error handling. In a production environment, you'd want to implement more robust error handling and recovery mechanisms.

4. **Security**: This demo uses insecure gRPC channels. For any real-world application, implement proper security measures.

5. **Shard_0 Dependency**: The generate script relies on `shard_0` for initial processing. Ensure it's available on the local machine running the script.

## Extending the System

To extend the system for more than three shards:

1. Create additional shards using `sharding_weight.py`.
2. Set up more server instances, one for each new shard.
3. In `generate.py`, add more gRPC channels for the new shards.
4. Modify the `generate_step` function to pass data through all shards in the correct order.

## Requirements

- Python 3.x
- MLX library
- gRPC and related dependencies
- NumPy
- Transformers library
- Sufficient RAM on each machine to load and process its model shard
