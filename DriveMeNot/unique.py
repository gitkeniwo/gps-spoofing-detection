import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
data=pd.read_csv('/Users/liguangyu/Downloads/gps-spoofing-detection-cellular-master/trace4_cell.csv')
distance=data['distance'].values
dbm=data['dBm'].values
dbm=np.float32(dbm)
distance=np.float32(distance)
dBm = dbm.reshape(-1, 1)
distance = distance.reshape(-1, 1)
cell=40283
data=data[data['CID']==cell]
data=data[data['dBm']<15]
data=data[['dBm','distance']]
data.to_csv('unique.csv')
import pandas as pd
import matplotlib.pyplot as plt
data.plot.scatter(x='dBm', y='distance')
plt.title('Scatter plot of strength vs distance')
plt.show()