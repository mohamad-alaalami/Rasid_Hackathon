from strands import tool
import torch


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
            "GPU hardware inspection"
        ],
        "pending": [
            "Dataset inspection",
            "GPU model training",
            "Model evaluation",
            "Automatic experiment improvement"
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
