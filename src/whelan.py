#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')


# # Load data

# ## Load data - Demonstration

# In[2]:


df = pd.read_csv('../data/drive-me-not/trace1.csv')
# benign_flight.info()

#filter out the anchor points
df = df[df['Anchor_Number'] == 0]
df


# In[3]:


import pandasql as ps

# Filter out the rows whose change of position is not reflected in the coordinates
stmt = """SELECT * 
FROM df
WHERE Time in (
    SELECT min(Time) 
    FROM df
    GROUP BY GPS_lat, GPS_long 
    )
"""

df = ps.sqldf(stmt, locals())


# In[4]:


from utils.preprocessing import zero_one_normalization

# compute velocity
df['vx'] = df.GPS_long.diff() / df.Time.diff()
df['vy'] = df.GPS_lat.diff() / df.Time.diff()
df.dropna(inplace=True)

# compute acceleration
df['ax'] = df.vx.diff() / df.Time.diff()
df['ay'] = df.vy.diff() / df.Time.diff()
df.dropna(inplace=True)

# 0-1 normalization

for col in ['vx', 'vy', 'ax', 'ay']:
    df[col] = zero_one_normalization(df[col])

selected_attributes = ['GPS_lat', 'GPS_long', 'Time', 'vx', 'vy', 'ax', 'ay', 'dBm']
df = df[selected_attributes]
df


# In[7]:


from utils.visualization import plot_trace

plot_trace(df, mode="velocity")


# ## Pipelining the dataloader

# In[2]:


from utils.preprocessing import data_preprocessing

traces = ['../data/drive-me-not/trace'+ str(i) + '.csv' for i in range(1, 9)]

traces_df = []
for trace in traces:
    traces_df.append(data_preprocessing(trace))
    
for df in traces_df:
    print(df.shape)


# # PCA preprocessing

# In[3]:


# pca
from utils.preprocessing import pca_transform, add_traces

N_COMPONENTS = 3

pca_dfs = [add_traces(df=pca_transform(df, n_components=N_COMPONENTS), num=i+1) for i, df in enumerate(traces_df)]

pca_dfs = pd.concat(pca_dfs)
pca_dfs['trace'] = pca_dfs['trace'].astype(int).astype(str)
pca_dfs.reset_index(drop=True, inplace=True)
pca_dfs


# In[4]:


from utils.visualization import plot_pca

plot_pca(pca_dfs, n_components=N_COMPONENTS)


# # One-class Classification

# ## OCSVM

# In[314]:


from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV
import math

def optimize_OneClassSVM(X, n):
    
    print('Searching for optimal hyperparameters...')
    nu = np.linspace(start=1e-5, stop=1e-2, num=n)
    gamma = np.linspace(start=1e-6, stop=1e-3, num=n)
    opt_diff = 1.0
    opt_nu = None
    opt_gamma = None
    
    for i in range(len(nu)):
        for j in range(len(gamma)):
            classifier = OneClassSVM(kernel="rbf", nu=nu[i], gamma=gamma[j])
            classifier.fit(X)
            label = classifier.predict(X)
            
            p = 1 - float(sum(label == 1.0)) / len(label)
            
            diff = math.fabs(p - nu[i]) # difference between the predicted and expected error rate
            
            if diff < opt_diff: # update the optimal hyperparameters
                opt_diff = diff
                opt_nu = nu[i]
                opt_gamma = gamma[j]
                
    print("Found: nu = %d, gamma = %f" % (opt_nu, opt_gamma))
    return opt_nu, opt_gamma

df_train = pca_dfs[['pca-one', 'pca-two', 'pca-three']]
nu_opt, gamma_opt = optimize_OneClassSVM(df_train, 20)



# ## Local Outlier Factor (LOF)

# In[ ]:





# # Autoencoder

# In[ ]:




