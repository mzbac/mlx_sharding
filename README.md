# MLX Sharding

This project demonstrates how to implement pipeline parallelism for large language models using MLX. It includes tools for sharding a model, serving shards across multiple machines, and generating text using the distributed model. Additionally, it features an OpenAI API-compatible server for easier integration and usage.

## Demo Video

To see the distributed inference in action, check out our demo video:

[Sharding DeepSeek-Coder-V2-Lite-Instruct Demo](https://www.youtube.com/watch?v=saOboSfP76o)

## Quick Start

### Installation

Install the package using pip:

```bash
pip install mlx-sharding
```

### Running the Servers

1. For the shard node:

   ```bash
   mlx-sharding-server --model mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx --start-layer 14 --end-layer 27
   ```

2. For the primary node:

   ```bash
   mlx-sharding-api --model mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx --start-layer 0 --end-layer 14 --llm-shard-addresses <your shard node address>
   ```

   Replace `<your shard node address>` with the actual address of your shard node (e.g., `localhost:50051`).

## Educational Purpose

This repository is designed for educational purposes to illustrate how pipeline parallelism can be implemented in MLX. It provides a basic framework for:

1. Sharding a large language model
2. Distributing model shards across multiple machines
3. Implementing a simple pipeline for text generation
4. Serving the model through an OpenAI API-compatible interface

While not optimized for production use, this demo serves as a starting point for understanding and experimenting with pipeline parallelism in machine learning workflows.

## Setup and Usage

### 1. Model Preparation

You have two main options for preparing and using the model:

#### Option A: Pre-Sharding the Model

If you prefer to pre-shard the model, use `sharding_weight.py`:

```bash
python sharding_weight.py --model "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx" --output_dir shard_0 --start_layer 0 --end_layer 14 --total_layers 27
python sharding_weight.py --model "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx" --output_dir shard_1 --start_layer 14 --end_layer 27 --total_layers 27
# Repeat for additional shards as needed
```

#### Option B: Dynamic Sharding

You can let the system dynamically load and shard the weights when starting the server. This option doesn't require pre-sharding.

### 2. Distribute Shards (If Using Option A)

If you've pre-sharded the model, copy the shard directories to their respective machines. Skip this step for Option B.

### 3. Start the Servers

Start server instances based on your chosen approach:

#### For Pre-Sharded Model (Option A)

On each machine with a shard, start a server instance. For example:

```bash
python -m shard.main --model mzbac/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx-shard-1
```

#### For Dynamic Sharding (Option B)

Start the server with specific layer ranges:

```bash
python -m shard.main --model "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx" --start-layer 0 --end-layer 14
```

Note the IP address and port printed by each server.

### 4. Generate Text

#### Using the generate script

For a dynamically sharded setup:

```bash
python generate.py --model "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx" --start_layer 0 --end_layer 14 --server_address <remote_ip1>:<port1>,<remote_ip2>:<port2> --prompt "Your prompt here" --max_tokens 512
```

For a pre-sharded setup:

```bash
python generate.py --model mzbac/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx-shard-0 --server_address <remote_ip1>:<port1>,<remote_ip2>:<port2> --prompt "Your prompt here" --max_tokens 512
```

#### Using the OpenAI API-compatible server

1. Start the server:

   For dynamic sharding:

   ```bash
   python -m shard.openai_api --model "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx" --llm-shard-addresses localhost:50051,<remote_ip1>:<port1>,<remote_ip2>:<port2> --start-layer 0 --end-layer 14
   ```

   For pre-sharded model:

   ```bash
   python -m shard.openai_api --model mzbac/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx-shard-0 --llm-shard-addresses localhost:50051,<remote_ip1>:<port1>,<remote_ip2>:<port2>
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

### 5. Web User Interface

This project now includes a web-based user interface for easy interaction with the model. To use the UI:

1. Ensure the OpenAI API-compatible server is running (as described in step 4).

2. Navigate to `http://localhost:8080` (or the appropriate host and port if you've configured it differently) in your web browser.

3. Use the interface to input prompts, adjust parameters, and view the model's responses.

The UI provides a user-friendly way to interact with the model, making it easier to experiment with different inputs and settings without needing to use command-line tools or write code.

## Limitations and Considerations

1. **Network Dependency**: The performance of this pipeline parallelism implementation is heavily dependent on network speed and latency between machines.

2. **Error Handling**: The current implementation has basic error handling. In a production environment, you'd want to implement more robust error handling and recovery mechanisms.

3. **Security**: This demo uses insecure gRPC channels. For any real-world application, implement proper security measures.

4. **Shard Configuration**: Ensure that when using multiple shards, the layer ranges are set correctly to cover the entire model without overlap.

## Extending the System

To extend the system for more shards:

1. If pre-sharding, create additional shards using `sharding_weight.py`.
2. Set up more server instances, one for each new shard.
3. In `generate.py` or when using the OpenAI API server, include all shard addresses.
4. Adjust the layer ranges accordingly when using dynamic sharding.

## Requirements

- Python 3.x
- MLX library
- gRPC and related dependencies
- NumPy
- Transformers library
- Sufficient RAM on each machine to load and process its model shard

## Acknowledgments

- MLX team for providing the framework
- Exo(<https://github.com/exo-explore/exo>) that I heavily inspired from for their implementation of pipeline parallelism
