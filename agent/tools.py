"""Safe Strands tools exposing the registry-driven ML engine."""

from strands import tool
import torch

from ml import train_model as run_training
from ml.inspect_dataset import inspect_dataset as inspect_ml_dataset
from ml.registry import get_dataset_spec, list_dataset_specs


@tool
def agent_capabilities() -> dict:
    """Return GoTrain's implemented capabilities and registered datasets."""
    return {
        "agent": "GoTrain",
        "status": "prototype",
        "available": [
            "Claude Sonnet via Amazon Bedrock",
            "Strands tool calling",
            "GPU inspection",
            "registry-driven classification and segmentation",
            "validation-only model selection",
        ],
        "datasets": list_dataset_specs(),
        "pending": ["automatic multi-run comparison", "final one-time test evaluation"],
    }


@tool
def inspect_gpu() -> dict:
    """Check GPU availability, name, memory, and CUDA version."""
    if not torch.cuda.is_available():
        return {"gpu_available": False, "message": "CUDA GPU is not available."}
    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "gpu_available": True,
        "gpu_name": torch.cuda.get_device_name(device),
        "total_memory_gb": round(total_bytes / (1024 ** 3), 2),
        "free_memory_gb": round(free_bytes / (1024 ** 3), 2),
        "cuda_version": torch.version.cuda,
    }


@tool
def list_supported_datasets() -> dict:
    """List datasets whose contracts are registered and safe for training."""
    return {"datasets": list_dataset_specs()}


@tool
def inspect_dataset(dataset_id: str) -> dict:
    """Inspect one registered dataset's train/validation data without using test data."""
    return inspect_ml_dataset(dataset_id=dataset_id)


@tool
def train_registered_model(
    dataset_id: str,
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.0003,
    weight_decay: float = 0.0001,
    pretrained: bool = True,
    seed: int = 42,
) -> dict:
    """Train one safe experiment for a registered classification or segmentation dataset.

    Inspect the dataset first. The registry selects the compatible task type, model,
    channels, class count, and metric. This tool never accesses the test split.
    """
    spec = get_dataset_spec(dataset_id)
    if not 1 <= epochs <= 20:
        return {"status": "rejected", "message": "epochs must be between 1 and 20."}
    if not 1 <= batch_size <= 64:
        return {"status": "rejected", "message": "batch_size must be between 1 and 64."}
    return run_training(
        dataset_id=dataset_id,
        config={
            "model": spec.default_model,
            "pretrained": pretrained,
            "input_mode": "registry",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "device": "auto",
            "mixed_precision": True,
        },
    )
