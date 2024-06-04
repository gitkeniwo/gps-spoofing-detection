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

# Define the neural network model
class DistanceNN(nn.Module):
    def __init__(self):
        super(DistanceNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x=torch.exp(self.fc4(x))
        return x
if __name__ == "__main__":
    # # Initialize the model, loss function, and optimizer
    model = DistanceNN()
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 500
    batch_size = 32

    # Convert to DataLoader
    train_dataset = torch.utils.data.TensorDataset(dBm_train, distance_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        # print(epoch)
        c=0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
            loss.backward()
            optimizer.step()
            c+=loss.item()
        if (epoch+1) % 10 == 0:
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {c:.4f}')

    # Evaluate the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients during evaluation
        test_outputs = model(dBm_test)
        test_loss = criterion(test_outputs, distance_test)
        print(f'Test Loss: {test_loss.item():.4f}')
    torch.save(model.state_dict(), 'distance_model.pth')
    # Make predictions
    predictions = test_outputs.numpy()
    target=distance_test.numpy()
    # print(predictions[:50]-target[:50])

    # model = DistanceNN()
    with torch.no_grad():  # No need to track gradients during evaluation
        test_outputs = model(dBm_test)
        test_loss = criterion(test_outputs, distance_test)
        print(f'Test Loss: {test_loss.item():.4f}')
        predictions = test_outputs.numpy()
        target=distance_test.numpy()
        print(np.quantile(np.abs(predictions-target), 0.9))
        print(np.quantile(np.abs(predictions-target), 0.5))


    with torch.no_grad():  # No need to track gradients during evaluation
        train_outputs = model(dBm_train)
        train_loss = criterion(train_outputs, distance_train)
        print(f'Train Loss: {train_loss.item():.4f}')
        predictions = train_outputs.numpy()
        target=distance_train.numpy()
        print(np.quantile(np.abs(predictions-target), 0.9))
        print(np.quantile(np.abs(predictions-target), 0.5))





    # print(target[:5])
    # # Generate synthetic data for demonstration
    # np.random.seed(42)
    # X = np.random.rand(1000, 10).astype(np.float32)  # 1000 samples, 10 features
    # y = np.random.rand(1000, 1).astype(np.float32)   # 1000 target values

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Standardize the data (mean=0, std=1)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # # Convert data to PyTorch tensors
    # X_train = torch.tensor(X_train)
    # y_train = torch.tensor(y_train)
    # X_test = torch.tensor(X_test)
    # y_test = torch.tensor(y_test)

    # # Define the neural network model
    # class SimpleNN(nn.Module):
    #     def __init__(self):
    #         super(SimpleNN, self).__init__()
    #         self.fc1 = nn.Linear(10, 64)
    #         self.fc2 = nn.Linear(64, 64)
    #         self.fc3 = nn.Linear(64, 1)
        
    #     def forward(self, x):
    #         x = torch.relu(self.fc1(x))
    #         x = torch.relu(self.fc2(x))
    #         x = self.fc3(x)
    #         return x

    # # Initialize the model, loss function, and optimizer
    # model = SimpleNN()
    # criterion = nn.MSELoss()  # Mean Squared Error for regression
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Training the model
    # num_epochs = 50
    # batch_size = 32

    # # Convert to DataLoader
    # train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # for epoch in range(num_epochs):
    #     for inputs, targets in train_loader:
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()
        
    #     if (epoch+1) % 10 == 0:
    #         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # # Evaluate the model
    # model.eval()  # Set the model to evaluation mode
    # with torch.no_grad():  # No need to track gradients during evaluation
    #     test_outputs = model(X_test)
    #     test_loss = criterion(test_outputs, y_test)
    #     print(f'Test Loss: {test_loss.item():.4f}')

    # # Make predictions
    # predictions = test_outputs.numpy()
    # print(predictions[:5])
