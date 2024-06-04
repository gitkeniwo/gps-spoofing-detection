import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from distance import DistanceNN
criterion = nn.MSELoss()
loaded_model = DistanceNN()
loaded_model.load_state_dict(torch.load('distance_model.pth'))
loaded_model.eval()  # Set the model to evaluation mode

data=pd.read_csv('/Users/liguangyu/Downloads/gps-spoofing-detection-cellular-master/trace4_cell.csv')
distance=data['distance'].values[:1000]
dbm=data['dBm'].values[:1000]
dbm=np.float32(dbm)
distance=np.float32(distance)
dBm = dbm.reshape(-1, 1)
distance = distance.reshape(-1, 1)

# Split the data into training and testing sets
dBm_train, dBm_test, distance_train, distance_test = train_test_split(dBm, distance, test_size=0.2, random_state=42)

# Standardize the data (mean=0, std=1)
scaler = StandardScaler()
dBm_train = scaler.fit_transform(dBm_train)
dBm_test = scaler.transform(dBm_test)

# Convert data to PyTorch tensors
dBm_train = torch.tensor(dBm_train)
distance_train = torch.tensor(distance_train)
dBm_test = torch.tensor(dBm_test)
distance_test = torch.tensor(distance_test)

# Evaluate the loaded model
with torch.no_grad():  # No need to track gradients during evaluation
    test_outputs = loaded_model(dBm_test)
    test_loss = criterion(test_outputs, distance_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Make predictions with the loaded model
predictions = test_outputs.numpy()
target=distance_test.numpy()
print(np.quantile(predictions-target, 0.9))
print(np.quantile(predictions-target, 0.5))


with torch.no_grad():  # No need to track gradients during evaluation
    train_outputs = loaded_model(dBm_train)
    train_loss = criterion(train_outputs, distance_train)
    print(f'Test Loss: {train_loss.item():.4f}')
predictions = train_outputs.numpy()
target=distance_train.numpy()
print(np.quantile(predictions-target, 0.9))
print(np.quantile(predictions-target, 0.5))