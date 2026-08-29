import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import pandas as pd
import os
import bodo
import torch.multiprocessing as mp
from mpi4py import MPI
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bodo.libs.distributed_api import (
    create_subcomm_mpi4py,
    get_host_ranks,
    get_nodes_first_ranks,
)


def init_process_group():
    dist.init_process_group(backend="mpi", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def get_gpu_ranks(num_gpus_in_node):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    host_ranks = get_host_ranks()
    nodes_first_ranks = get_nodes_first_ranks()
    if rank in nodes_first_ranks:
        subcomm = create_subcomm_mpi4py(nodes_first_ranks)
        num_gpus_per_node = subcomm.gather(num_gpus_in_node)
        if rank == 0:
            gpu_ranks = []
            for i, ranks in enumerate(host_ranks.values()):
                n_gpus = num_gpus_per_node[i]
                if n_gpus == 0:
                    continue
                cores_per_gpu = len(ranks) // n_gpus
                for local_rank, global_rank in enumerate(ranks):
                    if local_rank % cores_per_gpu == 0:
                        my_gpu = local_rank / cores_per_gpu
                        if my_gpu < n_gpus:
                            gpu_ranks.append(global_rank)
            comm.bcast(gpu_ranks)
    if rank != 0:
        gpu_ranks = comm.bcast(None)
    return gpu_ranks


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


class TaxiDemandNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def batch_generator(X, y, batch_size, gpu_ranks):
    for start_idx in range(0, len(X), batch_size):
        end_idx = start_idx + batch_size
        if len(gpu_ranks) > 0:
            batch_X = bodo.rebalance(X[start_idx:end_idx], dests=list(gpu_ranks), parallel=True)
            batch_y = bodo.rebalance(y[start_idx:end_idx], dests=list(gpu_ranks), parallel=True)
        yield batch_X, batch_y


def main():
    rank, _ = init_process_group()
    num_gpus = torch.cuda.device_count()
    gpu_ranks = get_gpu_ranks(num_gpus)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    model = TaxiDemandNN(input_size=X_train.shape[1]).to(f"cuda:{local_rank % num_gpus}" if rank in gpu_ranks else "cpu")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank % num_gpus] if rank in gpu_ranks else None)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    batch_size = 32
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        for batch_X, batch_y in batch_generator(X_train, y_train, batch_size, gpu_ranks):
            batch_X_tensor = torch.tensor(batch_X, dtype=torch.float32).to(model.device)
            batch_y_tensor = torch.tensor(batch_y, dtype=torch.float32).view(-1, 1).to(model.device)
            y_pred = model(batch_X_tensor)
            loss = criterion(y_pred, batch_y_tensor)
            loss.backward()
            optimizer.step()

        if rank == 0 and epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Evaluation after training
    model.eval()
    with torch.no_grad():
        batch_X_tensor = torch.tensor(X_test, dtype=torch.float32).to(model.device)
        batch_y_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(model.device)
        y_pred = model(batch_X_tensor)
        loss = criterion(y_pred, batch_y_tensor)
        if rank == 0:
            print(f"Test Loss: {loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

