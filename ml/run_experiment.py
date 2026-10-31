"""Command-line wrapper around the same train_model API used by Strands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="m-eurosat")
    parser.add_argument("--config", required=True, help="Path to a JSON configuration file.")
    parser.add_argument("--model", help="Override config model with a registered model ID.")
    parser.add_argument("--epochs", type=int, help="Override config epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override config batch size.")
    parser.add_argument("--checkpoint-dir", help="Directory containing exact external checkpoints.")
    parser.add_argument(
        "--image-size",
        type=int,
        help="Explicit square resize override; omitted preserves native resolution.",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.model:
        config["model"] = args.model
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.image_size is not None:
        config["image_size"] = args.image_size
    result = train_model(args.dataset_id, config)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()

