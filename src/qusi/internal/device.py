import torch
from torch.types import Device


def get_device() -> Device:
    """
    Gets the available device for PyTorch to run on.

    :return: The device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device
