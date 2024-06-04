#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

import sys
sys.path.append("../") # add parent folder to sys.path


# # Prepare Data

# In[2]:


from models.lstm_model import DetectionLSTM
from utils.dataset import DroneDataset
from utils.preprocessing import data_preprocessing

datapath = "../data/drive-me-not/trace1.csv" 
df = data_preprocessing(filepath=datapath)
df_np = df.to_numpy()

data = DroneDataset(df_np, 10)


# # Train the Model

# In[5]:


# trainer
from utils.trainer import Trainer

LEARNING_RATE = 0.001
BATCH_SIZE = 64

dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

model = DetectionLSTM(input_size=6, 
                      hidden_size=25, 
                      num_layers=2, 
                      output_size=6, 
                      batch_first=True)

criteria = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

trainer = Trainer(model, dataloader, nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001))
trainer.train(epochs=100)


# # Make predictions

# In[ ]:




