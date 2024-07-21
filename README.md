# MLX Pipeline Parallelism Demo

This project demonstrates how to implement pipeline parallelism for large language models using MLX. It includes tools for sharding a model, serving shards across multiple machines, and generating text using the distributed model.

## Demo Video

To see the distributed inference in action, check out our demo video:

[MLX Pipeline Parallelism Inference Demo](https://www.youtube.com/watch?v=AgiqBfpkslI)

**Note on Performance:** In this demo, the inference speed is slower than optimal due to two main factors:

1. Network Bottleneck: The setup is using a WiFi network, which introduces latency in communication between shards.
2. Hardware Bottleneck: One of the machines used is a low spec Mac M1 Pro, which significantly bottlenecks the whole inference process.

## Educational Purpose

This repository is designed for educational purposes to illustrate how pipeline parallelism can be implemented in MLX. It provides a basic framework for:

1. Sharding a large language model
2. Distributing model shards across multiple machines
3. Implementing a simple pipeline for text generation

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
python -m server.server --model mzbac/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx-shard-1
```

Note the IP address and port printed by the server.

### 4. Configure the Generate Script

The `generate.py` script accepts command-line arguments for flexibility. Key arguments include:

- `--model`: Specifies the path or name of the model to use (local shard)
- `--server_address`: Specifies the address of the remote shard server
- `--prompt`: Sets the prompt for text generation
- `--max_tokens`: Sets the maximum number of tokens to generate

### 5. Generate Text

Run the generate script on the local machine with the desired arguments. For example:

```bash
python generate.py --model mzbac/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx-shard-0 --server_address <remote_ip>:<port> --prompt "Your prompt here" --max_tokens 512
```

This command:

- Uses the local shard `mzbac/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx-shard-0`
- Connects to a remote shard server at `<remote_ip>:<port>` (replace with the actual IP address and port of your remote server)
- Generates text based on the given prompt
- Limits the generation to a maximum of 512 tokens

The specified model (shard-0) handles the initial processing locally, while the remote shard (shard-1) processes on its machine as configured in the server setup.

You can adjust the `--prompt` and `--max_tokens` arguments as needed for your specific use case.

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
