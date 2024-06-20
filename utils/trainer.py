import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
import wandb
import numpy as np

import os

from functools import partial
from ray import tune, train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.train import Checkpoint, get_checkpoint
import ray.cloudpickle as pickle
from models import autoencoder

class Trainer:
    """
     Trainer class
    """
    
    def __init__(self, model: nn.Module, 
                 train_loader: DataLoader, val_loader: DataLoader, 
                 criterion, optimizer, 
                 if_wandb=False, 
                 wandb_project_name="default"+str(np.random.randint(0, 1000))
                 ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        
        if if_wandb:
            wandb.init(project=wandb_project_name)
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
        
        # print progress bar using tqdm, only when epoch % 10 == 0
        
        for i, data in tqdm.tqdm(enumerate(self.train_loader), 
                                 total=len(self.train_loader),
                                 leave=True,
                                 ):
            
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
            for i, data in tqdm.tqdm(enumerate(self.val_loader), 
                                     total=len(self.val_loader),
                                     leave=True,
                                     ):
                inputs, labels = data
                inputs, labels = inputs.float(), labels.float()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                
                if wandb.run:
                    wandb.log({"val_loss": val_loss})    
                
        
        
        # wandb logging 
            
        self.model.train()
        
        return val_loss


    def train(self, epochs, path="../outputs/img/train_val_loss.png"):
        """Execute train process for multiple epochs."""
        
        train_loss = []
        val_loss = []
        
        for epoch in range(epochs):
            
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss.append(self.train_epoch())
            val_loss.append(self.validate_epoch())
            
            # visualize loss
            
        self.visualize_loss(train_loss, val_loss, path=path)
        print("Finished Training")
        
        if wandb.run:
            wandb.finish()
            
        return train_loss[-1],  val_loss[-1]
            
            
    def visualize_loss(self, train_loss, val_loss, path: str=f"../outputs/img/train_val_loss.png"):
        """Visualize loss using seaborn """
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_theme(context="notebook", style="darkgrid")
        plt.plot(train_loss, label="train_loss")
        plt.plot(val_loss, label="val_loss")
        plt.legend()
        plt.savefig(path)
        plt.show()
        
    def save_model(self, path):
        """Save model to the given path."""
        
        self.model.eval()
        
        torch.save(self.model.state_dict(), path)
     











def ray_trainer(config, df_train_ae, df_test_ae):
    
    model = autoencoder.Autoencoder(hidden_size=config['hidden_size'])
    
    
    
    
    from torch.utils.data import TensorDataset
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(df_train_ae, df_train_ae, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # turn to tensor
    autoencoder_X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    autoencoder_y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    autoencoder_X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    autoencoder_y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
    autoencoder_X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    autoencoder_y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32)

    # create TensorDataset
    autoencoder_train_dataset = TensorDataset(autoencoder_X_train_tensor, autoencoder_y_train_tensor)
    autoencoder_test_dataset = TensorDataset(autoencoder_X_test_tensor, autoencoder_y_test_tensor)
    autoencoder_val_dataset = TensorDataset(autoencoder_X_val_tensor, autoencoder_y_val_tensor)
    
    train_loader = DataLoader(autoencoder_train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(autoencoder_test_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(autoencoder_val_dataset, batch_size=config['batch_size'], shuffle=True)


    criterion = torch.nn.MSELoss()
    #adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    

        
    def train_epoch():
        
        running_loss = 0.0
        
        # print progress bar using tqdm, only when epoch % 10 == 0
        
        for i, data in tqdm.tqdm(enumerate(train_loader), 
                                    total=len(train_loader),
                                    leave=True,
                                    ):
            
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # Convert to float
            inputs, labels = inputs.float(), labels.float()
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            running_loss += loss.item()         


        print(f"Train Loss: {running_loss}")
    
        return running_loss
             
                
    def validate_epoch():
        """Execute validation process for one epoch."""
        # forward pass
        model.eval()
        
        val_loss = 0.0
        
        with torch.no_grad():
            for i, data in tqdm.tqdm(enumerate(val_loader), 
                                        total=len(val_loader),
                                        leave=True,
                                        ):
                inputs, labels = data
                inputs, labels = inputs.float(), labels.float()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()

            
        model.train()
        
        return val_loss
    
    
    train_loss = []
    val_loss = []
    
    epochs = 100
    
    for epoch in range(epochs):
        
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss.append(train_epoch())
        val_loss.append(validate_epoch())

    print("Finished Training")    
    return {"train_loss": train_loss[-1], "val_loss": val_loss[-1]}
    

































# class TuneTrainer(tune.Trainable):
#     """
#     TuneTrainer class, subclass of tune.Trainable
#     """
#     def setup(self, config):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
#         # Initialize your model, criterion, and optimizer with config
#         self.model = (config).to(self.device)
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = torch.optim.Adam(
#             self.model.parameters(), lr=config["lr"]
#         )

#         # Load your datasets and create DataLoaders
#         # Note: Adjust these lines according to your dataset
#         train_dataset = YourDataset(train=True)
#         val_dataset = YourDataset(train=False)
        
#         self.train_loader = DataLoader(
#             train_dataset, batch_size=int(config["batch_size"]), shuffle=True
#         )
#         self.val_loader = DataLoader(
#             val_dataset, batch_size=int(config["batch_size"])
#         )

#     def step(self):
#         train_loss = self.train_epoch()
#         val_loss = self.validate()
        
#         # Report metrics to Tune
#         return {"val_loss": val_loss, "train_loss": train_loss}

#     def train_epoch(self):
#         self.model.train()
#         total_loss = 0.0
        
#         for batch in self.train_loader:
#             inputs, targets = batch
#             inputs, targets = inputs.to(self.device), targets.to(self.device)
            
#             self.optimizer.zero_grad()
#             outputs = self.model(inputs)
#             loss = self.criterion(outputs, targets)
#             loss.backward()
#             self.optimizer.step()
            
#             total_loss += loss.item()
        
#         return total_loss / len(self.train_loader)

#     def validate(self):
#         self.model.eval()
#         total_loss = 0.0
        
#         with torch.no_grad():
#             for batch in self.val_loader:
#                 inputs, targets = batch
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
                
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, targets)
#                 total_loss += loss.item()
        
#         return total_loss / len(self.val_loader)

#     def save_checkpoint(self, checkpoint_dir):
#         checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
#         torch.save(self.model.state_dict(), checkpoint_path)
#         return checkpoint_path

#     def load_checkpoint(self, checkpoint_path):
#         self.model.load_state_dict(torch.load(checkpoint_path))     



# def tune_hyperparameters(config: dict={
#                             "lr": tune.loguniform(1e-4, 1e-1),
#                             "batch_size": tune.choice([16, 32, 64, 128]),
#                             "hidden_size": tune.choice([3, 6, 9]),
#                         }, 
#                         num_samples=10, max_num_epochs=10, gpus_per_trial=1):
#     """
#     tune_hyperparameters _summary_

#     Parameters
#     ----------
#     num_samples : int, optional
#         _description_, by default 10
#     max_num_epochs : int, optional
#         _description_, by default 10
#     gpus_per_trial : int, optional
#         _description_, by default 1

#     Returns
#     -------
#     _type_
#         _description_
#     """
    
#     config = config

#     # scheduler is responsible for early stopping
#     scheduler = ASHAScheduler(
#         metric="val_loss",
#         mode="min",
#         max_t=max_num_epochs,
#         grace_period=1,
#         reduction_factor=2
#     )

#     # progress reporter is responsible for reporting the progress
#     reporter = CLIReporter(
#         metric_columns=["train_loss", "val_loss", "training_iteration"]
#     )

#     result = tune.run(
#         TuneTrainer,
#         resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter,
#         name="tune_model_hyperparameters"
#     )

#     best_trial = result.get_best_trial("val_loss", "min", "last")
#     print(f"Best trial config: {best_trial.config}")
#     print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")

#     # Load the best model
#     best_checkpoint_dir = best_trial.checkpoint.value
#     model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "model.pth"))
#     model = YourModel(best_trial.config)
#     model.load_state_dict(model_state)

#     return model