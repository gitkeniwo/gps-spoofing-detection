import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import wandb

class Trainer:
    """
     Trainer class
    """
    
    def __init__(self, model: nn.Module, dataloader: DataLoader, criterion, optimizer, if_wandb=False, ):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        
        if if_wandb:
            wandb.init(project="pytorch-template")
            wandb.config.update({"model": model.__class__.__name__,
                                "optimizer": optimizer.__class__.__name__,
                                "criterion": criterion.__class__.__name__,
                                "learning_rate": optimizer.param_groups[0]['lr'],
                                "batch_size": dataloader.batch_size,
                                "dataset": "drone-dataset",
                                "epochs": 100})
            
            # if wandb is enabled, log the model 
            wandb.watch(self.model)

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
                
                # wandb logging
                if wandb.run:
                    wandb.log({"loss": loss.item()})
                
            print(f"Epoch {epoch+1}, loss: {running_loss/len(self.dataloader)}")
         
        if wandb.run:   
            wandb.finish()
        
            