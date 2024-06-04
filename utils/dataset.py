import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd


class DroneDataset(Dataset):
    """
    DroneDataset is a class that extends torch.utils.data.Dataset. 
    It is used to create a dataset for training our model.

    Parameters
    
    data : np.ndarray
        The data to be used for training the model. The data should be in the shape of (num_samples, feature_dim)
    seq_length : int
        The length of the sequence to be used for training the model. Default is 10.
    ----------
    Dataset : torch.utils.data.Dataset
    """
    def __init__(self, data: np.ndarray, seq_length: int=10):
        self.seq_length = seq_length
        self.data, self.target = DroneDataset._seq_generate(seq_length=self.seq_length, df=data)
        
    def __repr__(self) -> str:
        return f"DroneDataset(seq_length={self.seq_length}, num_samples={len(self.data)}, datashape={self.data[0].shape}, targetshape={self.target[0].shape})"
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y
    
    def __len__(self):
        return len(self.data)
    
    def _seq_generate(seq_length: int=10, df: np.ndarray=np.zeros(1)) -> tuple:   
        data = []
        target = []

        for i in range(len(df) - seq_length):
            seq = df[i:i+seq_length] # shape = (seq_length, 6)
            label = df[i+seq_length]  # Predict next state [x, y, vx, vy, ax, ay]
            data.append(seq)
            target.append(label)
        return (data, target)