from strands import tool
import torch

from ml import train_model as run_training
from ml.inspect_dataset import inspect_dataset as inspect_ml_dataset


# ============================================
# TOOL 1: AGENT CAPABILITIES
# ============================================

@tool
def agent_capabilities() -> dict:
    """Return the current capabilities of the GoTrain AI agent."""

    return {
        "agent": "GoTrain",
        "status": "prototype",
        "available": [
            "Claude Sonnet via Amazon Bedrock",
            "Strands tool calling",
            "GPU hardware inspection",
            "m-eurosat dataset inspection",
            "ResNet-18 training and validation"
        ],
        "pending": [
            "Automatic experiment comparison and improvement",
            "Segmentation model training",
            "Final test-split evaluation"
        ]
    }


# ============================================
# TOOL 2: GPU INSPECTION
# ============================================

@tool
def inspect_gpu() -> dict:
    """Check GPU availability, name, memory, and CUDA version."""

    if not torch.cuda.is_available():
        return {
            "gpu_available": False,
            "message": "CUDA GPU is not available."
        }

    device = torch.cuda.current_device()

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)

    return {
        "gpu_available": True,
        "gpu_name": torch.cuda.get_device_name(device),
        "total_memory_gb": round(total_bytes / (1024 ** 3), 2),
        "free_memory_gb": round(free_bytes / (1024 ** 3), 2),
        "cuda_version": torch.version.cuda
    }


# ============================================
# TOOL 3: DATASET INSPECTION
# ============================================

@tool
def inspect_m_eurosat() -> dict:
    """Inspect the installed m-eurosat training and validation data.

    Use this tool before training to confirm split sizes, band shapes,
    reflectance ranges, RGB indices, and label representation. This tool is
    read-only and never accesses the test split.
    """

    return inspect_ml_dataset()


# ============================================
# TOOL 4: CONTROLLED MODEL TRAINING
# ============================================

@tool
def train_m_eurosat_model(
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.0003,
    weight_decay: float = 0.0001,
    pretrained: bool = True,
    seed: int = 42,
) -> dict:
    """Train and validate one ResNet-18 experiment on m-eurosat.

    Use this tool only after inspect_gpu and inspect_m_eurosat succeed. It
    trains exclusively on the official training split, evaluates exclusively
    on the validation split, saves the best checkpoint, and returns structured
    metrics. It never accesses the test split. Start with one epoch as a smoke
    test. Do not repeat an identical configuration without a reason.
    """

    if epochs > 20:
        return {
            "status": "rejected",
            "error_type": "EXPERIMENT_LIMIT_EXCEEDED",
            "message": "epochs must be 20 or fewer for an agent experiment."
        }
    if batch_size > 64:
        return {
            "status": "rejected",
            "error_type": "GPU_SAFETY_LIMIT_EXCEEDED",
            "message": "batch_size must be 64 or fewer on the Tesla T4."
        }

    return run_training(
        dataset_id="m-eurosat",
        config={
            "model": "resnet18",
            "pretrained": pretrained,
            "input_mode": "rgb",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "device": "auto",
            "mixed_precision": True,
        },
    )
