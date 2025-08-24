import torch


def ensure_cuda():
    if torch.cuda.is_available():
        return "cuda"
    else:
        raise ValueError("Cuda not available.")
