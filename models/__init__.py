from . import *
import torch

# Define any package-level variables or configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

__all__ = [
    "lstm_model",
    'autoencoder',
    'ocsvm',
    'lof',
]