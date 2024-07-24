from .server.server import serve
import argparse

def main():
    parser = argparse.ArgumentParser(description="MLX Tensor Server")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model or HuggingFace repo")
    parser.add_argument("-s", "--start-layer", type=int, default=None,
                        help="Start layer index for model sharding (optional)")
    parser.add_argument("-e", "--end-layer", type=int, default=None,
                        help="End layer index for model sharding (optional)")
    args = parser.parse_args()

    serve(args.model, start_layer=args.start_layer, end_layer=args.end_layer)

if __name__ == "__main__":
    main()