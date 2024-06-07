import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import wandb

class Trainer:
    """
     Trainer class
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion, optimizer, if_wandb=False, ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        
        if if_wandb:
            wandb.init(project="pytorch-template")
            wandb.config.update({"model": model.__class__.__name__,
                                "optimizer": optimizer.__class__.__name__,
                                "criterion": criterion.__class__.__name__,
                                "learning_rate": optimizer.param_groups[0]['lr'],
                                "batch_size": self.train_loader.batch_size,
                                "dataset": "drone-dataset",
                                "epochs": 100})
            
            # if wandb is enabled, log the model 
            wandb.watch(self.model)
            

    def train_epoch(self):
        """Execute train process for one epoch."""
        
        running_loss = 0.0
        
        for i, data in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            
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

        print(f"Train Loss: {running_loss}")
        return running_loss
                
    def validate_epoch(self):
        """Execute validation process for one epoch."""
        # forward pass
        self.model.eval()
        
        val_loss = 0.0
        
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(self.val_loader), total=len(self.val_loader)):
                inputs, labels = data
                inputs, labels = inputs.float(), labels.float()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                
                if wandb.run:
                    wandb.log({"val_loss": val_loss})    
                
        print(f"Val Loss: {loss.item()}")
        
        # wandb logging 
                
        self.model.train()
        return val_loss

    def train(self, epochs):
        """Execute train process for multiple epochs."""
        
        train_loss = []
        val_loss = []
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss.append(self.train_epoch())
            val_loss.append(self.validate_epoch())
            
            # visualize loss
            
        self.visualize_loss(train_loss, val_loss)
        print("Finished Training")
        
        if wandb.run:
            wandb.finish()
            
    def visualize_loss(self, train_loss, val_loss):
        """Visualize loss using matplotlib."""
        
        import matplotlib.pyplot as plt
        
        plt.plot(train_loss, label="train_loss")
        plt.plot(val_loss, label="val_loss")
        plt.legend()
        plt.show()
        plt.savefig(f"../outputs/img/train_val_loss_{self.model.__class__.__name__}.png")
        
        
    
    def save_model(self, path):
        """Save model to the given path."""
        
        torch.save(self.model.state_dict(), path)
        

        
            
            
        
            