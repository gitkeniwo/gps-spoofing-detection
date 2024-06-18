import torch
import torch.nn as nn

INPUT_SIZE = 6
# Input
HIDDEN_SIZE = 50
# Number of features in LSTM Hidden state h

NUM_LAYERS = 2
# Number of recurrent layers.
OUTPUT_SIZE = 6

BATCH_FIRST = True
'''
batch_first – If True, then the input and output tensors are 
provided as (batch, seq, feature) instead of (seq, batch, feature). 
Note that this does not apply to hidden or cell states. 
See the Inputs/Outputs sections below for details. Default: False
'''


class DetectionLSTM(nn.Module):
    """
    DetectionLSTM is our LSTM model for generation of time series trace data.
    """
    
    def __init__(self, 
                 input_size=INPUT_SIZE, 
                 hidden_size=HIDDEN_SIZE, 
                 num_layers=NUM_LAYERS, 
                 output_size=OUTPUT_SIZE, 
                 batch_first=BATCH_FIRST, 
                 **kwargs):
        super(DetectionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        forward is the method that is called when the model forward-propagates.

        Parameters
        ----------
        x : _type_
            _description_

        Returns
        -------
        output : Tensor
            Prediction of the model.
        """
  
        h0, c0 = self.get_hidden_states(x)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        # [:, -1, :] 代表取最后一个时间步的输出
        return out
    
    def get_hidden_states(self, x):
        """
        get_hidden_states initializes the hidden states of the LSTM model with zeros.

        Parameters
        ----------
        x : Tensor
            model input.

        Returns
        -------
        Tuple(Tensor, Tensor)
            hidden states of the LSTM model.
        """
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        return (h0, c0)