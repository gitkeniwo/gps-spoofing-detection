# self-defined autoencoder model

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

class Autoencoder(nn.Module):
    """
    Autoencoder class

    Parameters
    ----------
    nn : _type_
        _description_
    """
    
    def __init__(self, input_size, first_layer_size, hidden_size, last_layer_size, output_size):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, first_layer_size),
            nn.ReLU(True),
            nn.Linear(first_layer_size, hidden_size),
            nn.ReLU(True)
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, last_layer_size),
            nn.ReLU(True),
            nn.Linear(last_layer_size, output_size),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
    
def predict_spoofed(y_pred, spoofed_ae, T) -> list:
    """ 
    Predict spoofed data using the autoencoder model, over a threshold T.
    """
    
    spoofed_pred = []
    
    for y_p, s_t in zip(y_pred, spoofed_ae):
        
        mse = mean_squared_error(y_p, s_t)
        if mse > T:
            # larger than threshold, spoofed
            spoofed_pred.append(-1)
        else:
            # smaller than threshold, benign
            spoofed_pred.append(1)
            
    return spoofed_pred