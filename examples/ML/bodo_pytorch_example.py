import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import bodo

def init_process_group():
    dist.init_process_group(backend="mpi")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

@bodo.jit
def load_and_preprocess_data():
    # Load Data
    hvfhv_dataset = "s3://bodo-example-data/nyc-taxi/fhvhv_tripdata/"
    df = pd.read_parquet(hvfhv_dataset)
    
    # Define feature columns and target column
    features = ["PULocationID", "DOLocationID", "trip_time", "base_passenger_fare", "tolls", "sales_tax", "tips", "driver_pay"]
    target = "trip_miles"
    
    # Scale numerical features
    scaler = StandardScaler()
    scaler = scaler.fit(df[features].values)
    df[features] = scaler.transform(df[features].values)
    # Split the dataset into train and test sets
    #X_train, X_test, y_train, y_test = train_test_split(df[features].values, df[target].values, test_size=0.2, random_state=42)
    # Use that for quick testing
    X_train = df[0:100][features].values
    X_test = df[101:150][features].values
    y_train = df[0:100][target].values
    y_test = df[101:150][target].values
    
    return X_train, X_test, y_train, y_test

# Define a simple neural network
class TaxiDemandNN(nn.Module):
    def __init__(self, input_size):
        super(TaxiDemandNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    rank, world_size = init_process_group()
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Initialize model, loss function, and optimizer
    model = TaxiDemandNN(input_size=X_train.shape[1])
    model = torch.nn.parallel.DistributedDataParallel(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if rank == 0 and epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        test_loss = criterion(y_pred_test, y_test_tensor)
        if rank == 0:
            print(f"Test Loss: {test_loss.item():.4f}")
    
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()



# Another test
# def main():
#     # Initialize MPI-based distributed training
#     rank, world_size = init_process_group()

#     # Parameters
#     num_samples = 100000
#     num_features = 100
#     batch_size = 256
#     num_epochs = 5

#     data, labels = load_and_preprocess_data()

#     # Step 2: PyTorch model, loss, optimizer
#     model = SimpleModel(input_size=num_features, output_size=2)
#     model = torch.nn.parallel.DistributedDataParallel(model)  # Wrap model for distributed training

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.01)

#     # Step 3: Training loop
#     for epoch in range(num_epochs):
#         print(f"Rank {rank}, Epoch {epoch + 1}/{num_epochs}")

#         # Create DataLoader for local data
#         dataset = CustomDataset(local_data, local_labels)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#         for batch_data, batch_labels in dataloader:
#             # Forward pass
#             outputs = model(batch_data)
#             loss = criterion(outputs, batch_labels)

#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()

#             # Optimizer step
#             optimizer.step()

#         if rank == 0:
#             print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

#     # Clean up
#     dist.destroy_process_group()

