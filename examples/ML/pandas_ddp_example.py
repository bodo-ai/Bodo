import bodo.pandas as pd
from bodo.mpi4py import MPI
import bodo.ai
import torch
import torch.distributed as dist
import torch.distributed.checkpoint
import tqdm
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
import os

LR = 16e-6
EPOCHS = 5
NUM_CLASSES = 5
SEQ_LENGTH = 512
BATCH_SIZE = 32
CHECKPOINT_DIR = "./checkpoint_dir"

class PandasDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, device: torch.device | None = None):
        self.df = df
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        batch_texts = row["label"]
        batch_texts = torch.tensor(batch_texts, device=self.device)

        batch_y = row["tokenized"]
        input_ids = torch.tensor(batch_y["input_ids"], device=self.device)
        attention_mask = torch.tensor(batch_y["attention_mask"], device=self.device)

        return input_ids, attention_mask, batch_texts

class BodoDistributedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: PandasDataset, worker_ranks: list[int], shuffle=True):
        assert isinstance(dataset, PandasDataset), (
            "BodoDistributedSampler only works with PandasDataset"
        )
        self.dataset = dataset
        self.worker_ranks = worker_ranks
        self.shuffle = shuffle
        # Create a subcomm of worker ranks
        world_group = MPI.COMM_WORLD.Get_group()
        self.worker_group = world_group.Incl(worker_ranks)
        world_group.Free()
        self.worker_subcomm = MPI.COMM_WORLD.Create(self.worker_group)
        self.seed = 0

    def __del__(self):
        if hasattr(self, "worker_group") and self.worker_group != MPI.GROUP_NULL:
            self.worker_group.Free()
        if hasattr(self, "worker_subcomm") and self.worker_subcomm != MPI.COMM_NULL:
            self.worker_subcomm.Free()

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # Shuffle the indices if required
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
            self.seed += 1  # Change seed for next epoch

        # Ensure all ranks have the same number of samples
        max_sample_len = self.worker_subcomm.allreduce(len(indices), op=MPI.MAX)
        indices += indices[: (max_sample_len - len(indices))]
        return iter(indices)

    def __len__(self):
        return len(self.dataset)

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def process_dataset(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    labels = {'business':0,
        'entertainment':1,
        'sport':2,
        'tech':3,
        'politics':4
        }

    # Remove duplicates and extra whitespace
    df.drop_duplicates(subset=["text"])
    df["text"] = df.text.str.lower().str.replace(r"\s+", " ", regex=True)

    def map_tokenizer(x):
        tokenized = tokenizer(x, max_length=SEQ_LENGTH, truncation=True, padding='max_length')
        return {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask}

    df["label"] = df.category.map(labels)
    df["tokenized"] = df.text.map(map_tokenizer, engine='python')
    return df


def prepare_datasets(tokenizer):
    test_df = pd.read_parquet("test.parquet")
    train_df = pd.read_parquet("train.parquet")
    val_df = pd.read_parquet("val.parquet")

    test_df = process_dataset(test_df, tokenizer)
    train_df = process_dataset(train_df, tokenizer)
    val_df = process_dataset(val_df, tokenizer)

    # remove test examples from train
    train_df = train_df[~train_df.text.isin(test_df.text)]

    return train_df, val_df, test_df


def validation(model, val_loader, loss_fn):
    model.eval()
    rank = dist.get_rank()
    device = next(model.parameters()).device
    total_ddp_acc_loss_train = torch.zeros(3).to(device)

    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green"
        )
    with torch.no_grad():
        for input_id, mask, val_label in val_loader:
            output = model(input_id, mask)

            batch_loss = loss_fn(output, val_label.long())
            
            # Track local loss and accuracy stats
            acc = (output.argmax(dim=1) == val_label).sum().item()
            total_ddp_acc_loss_train[0] += acc
            total_ddp_acc_loss_train[1] += batch_loss.item()
            total_ddp_acc_loss_train[2] += val_label.size(0)

            if rank==0:
                inner_pbar.update(1)

    # Compute global loss and accuracy stats
    dist.all_reduce(total_ddp_acc_loss_train, op=dist.ReduceOp.SUM)
    val_accuracy = total_ddp_acc_loss_train[0] / total_ddp_acc_loss_train[2]
    avg_val_loss = total_ddp_acc_loss_train[1] / total_ddp_acc_loss_train[2]

    if rank == 0:
        inner_pbar.close()
        print(
                f"Loss: \t{avg_val_loss:.4f} Accuracy: \t{val_accuracy:.4f}"
            )

    return val_accuracy, avg_val_loss


def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.train()
    rank = dist.get_rank()
    device = next(model.parameters()).device
    total_ddp_acc_loss_train = torch.zeros(3).to(device)

    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue"
        )

    for input_id, mask, train_label in train_loader:

        output = model(input_id, mask)

        batch_loss = loss_fn(output, train_label.long())

        # Track local loss and accuracy stats
        acc = (output.argmax(dim=1) == train_label).sum().item()
        total_ddp_acc_loss_train[0] += acc
        total_ddp_acc_loss_train[1] += batch_loss.item()
        total_ddp_acc_loss_train[2] += train_label.size(0)

        model.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if rank==0:
            inner_pbar.update(1)
    
    # Compute global loss and accuracy stats
    dist.all_reduce(total_ddp_acc_loss_train, op=dist.ReduceOp.SUM)
    train_accuracy = total_ddp_acc_loss_train[0] / total_ddp_acc_loss_train[2]
    avg_train_loss = total_ddp_acc_loss_train[1] / total_ddp_acc_loss_train[2]

    if rank == 0:
        inner_pbar.close()
        print(
                f" Loss: \t{avg_train_loss:.4f} Accuracy: \t{train_accuracy:.4f}"
            )

    return train_accuracy, avg_train_loss


def train_main(train_df, val_df, test_df):
    model = bodo.ai.prepare_model(BertClassifier())
    if model:
        device = next(model.parameters()).device
    else:
        device = None
    gpu_ranks = bodo.get_gpu_ranks()

    # Rebalance data if using accelerators onto 
    # acclerator ranks
    accelerators_used = len(gpu_ranks) != 0
    if accelerators_used:
        train_df = bodo.rebalance(train_df, dests=gpu_ranks, random=True, parallel=True)
        val_df = bodo.rebalance(val_df, dests=gpu_ranks, random=True, parallel=True)
        test_df = bodo.rebalance(test_df, dests=gpu_ranks, random=True, parallel=True)


    train_dataset = PandasDataset(train_df, device)
    val_dataset = PandasDataset(val_df, device)
    test_dataset = PandasDataset(test_df, device)
    train_sampler = BodoDistributedSampler(
        train_dataset,
        worker_ranks=gpu_ranks
        if accelerators_used
        else list(range(MPI.COMM_WORLD.Get_size())),
    )
    val_sampler = BodoDistributedSampler(
        val_dataset,
        worker_ranks=gpu_ranks
        if accelerators_used
        else list(range(MPI.COMM_WORLD.Get_size())),
        shuffle=False,
    )
    test_sampler = BodoDistributedSampler(
        test_dataset,
        worker_ranks=gpu_ranks
        if accelerators_used
        else list(range(MPI.COMM_WORLD.Get_size())),
        shuffle=False,
    )
    if model == None:
        return
    pytorch_rank = dist.get_rank()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):

        if pytorch_rank == 0:
            print(f"Train Epoch: \t{epoch}")
        train_one_epoch(model, train_loader, loss_fn, optimizer)

        if pytorch_rank == 0:
            print("Validation: ")
        validation(model, val_loader, loss_fn)

        # Checkpoint every epoch
        base_model = (model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model)
        torch.distributed.checkpoint.state_dict_saver.save(
            {"model_state_dict": base_model.state_dict()},
            checkpoint_id=CHECKPOINT_DIR
        )
    
    if pytorch_rank == 0:
        print("Test: ")
    validation(model, test_loader, loss_fn)



if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_df, val_df, test_df = prepare_datasets(tokenizer)
    train_df.execute_plan()
    val_df.execute_plan()
    test_df.execute_plan()
    bodo.ai.torch_train(train_main, train_df, val_df, test_df)
    os._exit(0)

