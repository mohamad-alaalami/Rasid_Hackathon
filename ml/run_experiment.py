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
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = train_model(args.dataset_id, config)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()

