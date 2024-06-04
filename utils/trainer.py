import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

class Trainer:
    """
     Trainer class
    """
    
    def __init__(self, model: nn.Module, dataloader: DataLoader, criterion, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, epochs):
        """Execute train process, using tqdm for progress bar."""
        
        for epoch in range(epochs):
            
            running_loss = 0.0
            
            for i, data in tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                
                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                
                # Convert to float
                inputs, labels = inputs.float(), labels.float()
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Backward and optimize
                loss.backward()
                
                # Update model parameters
                self.optimizer.step()
                
                running_loss += loss.item()
                
            print(f"Epoch {epoch+1}, loss: {running_loss/len(self.dataloader)}")
            