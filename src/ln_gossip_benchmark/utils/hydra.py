import warnings
from pathlib import Path

import torch


def decode_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_device(device):
    if device is not None:
        device = device.lower()

    assert device in [None, "cpu", "cuda"], (
        "device must be one of [None, 'cpu', 'cuda']"
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "Warning: 'cuda' device is not available, switching to 'cpu' device"
        )
        device = "cpu"
    return torch.device(device)
