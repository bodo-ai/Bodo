import os

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertModel, BertTokenizer


class PandasDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels : pd.Series = df["label"]
        self.tokenized : pd.Series = df["tokenized"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        batch_texts = self.labels.iloc[idx]
        batch_y = self.tokenized.iloc[idx]

        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
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

    df.drop_duplicates(subset=["text"])
    df["text"] = df.text.str.lower().str.replace(r"\s+", " ", regex=True)

    def map_tokenizer(x):
        tokenized = tokenizer(x, max_length=10, truncation=True, padding='max_length')
        return {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask}

    df["label"] = df.label.map(labels)
    df["tokenized"] = df.text.map(map_tokenizer)
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


def ddp_validation(model, rank, world_size, val_loader, loss_fn):
    model.eval()
    device = torch.device(f"cuda:{rank}")
    total_ddp_acc_loss_train = torch.zeros(3).to(device)

    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for val_input, val_label in val_loader:
            val_label = val_label.to(device)
            mask = val_input['attention_mask'].to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = loss_fn(output, val_label.long())

            acc = (output.argmax(dim=1) == val_label).sum().item()
            total_ddp_acc_loss_train[0] += acc
            total_ddp_acc_loss_train[1] += batch_loss.item()
            total_ddp_acc_loss_train[2] += val_label.size(0)

            if rank==0:
                inner_pbar.update(1)

    dist.all_reduce(total_ddp_acc_loss_train, op=dist.ReduceOp.SUM)
    val_accuracy = total_ddp_acc_loss_train[0] / total_ddp_acc_loss_train[2]
    avg_val_loss = total_ddp_acc_loss_train[1] / total_ddp_acc_loss_train[2]

    if rank == 0:
        inner_pbar.close()
        print(
                f"Validation Loss: \t{avg_val_loss:.4f} Accuracy: \t{val_accuracy:.4f}"
            )

    return val_accuracy, avg_val_loss


def ddp_train_one_epoch(model, rank, train_loader, loss_fn, optimizer, epoch, sampler=None):
    model.train()
    device = torch.device(f"cuda:{rank}")
    total_ddp_acc_loss_train = torch.zeros(3).to(device)

    if sampler:
        sampler.set_epoch(epoch)

    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )

    for train_input, train_label in train_loader:

        train_label = train_label.to(device)
        mask = train_input['attention_mask'].to(device)
        input_id = train_input['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask)

        batch_loss = loss_fn(output, train_label.long())

        acc = (output.argmax(dim=1) == train_label).sum().item()
        total_ddp_acc_loss_train[0] += acc
        total_ddp_acc_loss_train[1] += batch_loss.item()
        total_ddp_acc_loss_train[2] += train_label.size(0)

        model.zero_grad()
        batch_loss.backward()
        optimizer.step()
        if rank==0:
            inner_pbar.update(1)

    dist.all_reduce(total_ddp_acc_loss_train, op=dist.ReduceOp.SUM)
    train_accuracy = total_ddp_acc_loss_train[0] / total_ddp_acc_loss_train[2]
    avg_train_loss = total_ddp_acc_loss_train[1] / total_ddp_acc_loss_train[2]

    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{avg_train_loss:.4f} Accuracy: \t{train_accuracy:.4f}"
            )

    return train_accuracy, avg_train_loss


def ddp_main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    LR = 1e-6
    EPOCHS = 1

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_df, val_df, _test_df = prepare_datasets(tokenizer)
    train_dataset = PandasDataset(train_df)
    val_dataset = PandasDataset(val_df)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=2, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=2, sampler=val_sampler)

    model = BertClassifier().to(device)
    ddp_model = DDP(model)

    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # TODO: checkpointing?
        ddp_train_one_epoch(ddp_model, rank, train_loader, loss_fn, optimizer, epoch, train_sampler)
        ddp_validation(ddp_model, rank, world_size, val_loader, loss_fn)

    # TODO: Evaluate test dataset at the end of training

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(ddp_main, args=(world_size,), nprocs=world_size, join=True)

