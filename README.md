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
python sharding_weight.py --model_dir <path_to_full_model> --output_dir shard_0 --start_layer 0 --end_layer 2
python sharding_weight.py --model_dir <path_to_full_model> --output_dir shard_1 --start_layer 2 --end_layer 4
python sharding_weight.py --model_dir <path_to_full_model> --output_dir shard_2 --start_layer 4 --end_layer 6
# Repeat for additional shards as needed
```

This will create directories (`shard_0`, `shard_1`, `shard_2`, etc.) containing the sharded model weights and configurations.

### 2. Distribute Shards

Copy the shard directories to their respective machines:

- Local machine (for generate script): Keep `shard_0`
- Machine 1: Copy `shard_1`
- Machine 2: Copy `shard_2`

### 3. Start the Servers

On each remote machine, start a server instance for its respective shard:

Machine 1:

```bash
python -m server.server --model /path/to/shard_1
```

Machine 2:

```bash
python -m server.server --model /path/to/shard_2
```

Note the IP addresses and ports printed by each server.

### 4. Configure the Generate Script

Update `generate.py` with the correct IP addresses and ports for each shard:

```python
channel_1 = grpc.insecure_channel('machine1_ip:port')  # For shard_1
channel_2 = grpc.insecure_channel('machine2_ip:port')  # For shard_2
# Add more channels for additional shards if needed
```

The generate script uses `shard_0` locally for handling the initial prompt to tensor conversion and tokenization:

```python
tokenizer = AutoTokenizer.from_pretrained("/path/to/shard_0")
model = load_model("/path/to/shard_0")
```

### 5. Generate Text

Run the generate script on the local machine:

```bash
python generate.py
```

This will use the distributed model to generate text based on the given prompt, with `shard_0` handling the initial processing locally, and `shard_1` and `shard_2` processing on their respective remote machines.

## Limitations and Considerations

1. **Cache Cleanup**: The current implementation does not clean up the cache between generations. To start fresh, you need to restart the server processes.

2. **Fixed Shard Configuration**: The generate script is currently set up for three shards (one local, two remote). To use more shards, you'll need to modify the script to include additional gRPC channels and adjust the `generate_step` function accordingly.

3. **Network Dependency**: The performance of this pipeline parallelism implementation is heavily dependent on network speed and latency between machines.

4. **Error Handling**: The current implementation has basic error handling. In a production environment, you'd want to implement more robust error handling and recovery mechanisms.

5. **Security**: This demo uses insecure gRPC channels. For any real-world application, implement proper security measures.

6. **Shard_0 Dependency**: The generate script relies on `shard_0` for initial processing. Ensure it's available on the local machine running the script.

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
