import bodo.pandas as pd
import bodo.ai
import torch
import torch.distributed as dist
import torch.distributed.checkpoint
import tqdm
from torch import nn
from torch.optim import Adam
from transformers import BertModel, BertTokenizer
import os

LR = 16e-6
EPOCHS = 5
NUM_CLASSES = 5
SEQ_LENGTH = 512
BATCH_SIZE = 32
CHECKPOINT_DIR = "./checkpoint_dir"

class BertDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        batch_texts = row["label"]
        batch_texts = torch.tensor(batch_texts)

        batch_y = row["tokenized"]
        input_ids = torch.tensor(batch_y["input_ids"])
        attention_mask = torch.tensor(batch_y["attention_mask"])

        return input_ids, attention_mask, batch_texts

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


def preprocess_datasets(tokenizer):
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
        print(f"Training Steps: {len(train_loader)}")


    for input_id, mask, train_label in train_loader:
        input_id = input_id.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        train_label = train_label.to(device, non_blocking=True)

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


    train_loader = bodo.ai.prepare_dataset(train_df, BATCH_SIZE, dataset_func=BertDataset)
    val_loader = bodo.ai.prepare_dataset(val_df, BATCH_SIZE, dataset_func=BertDataset)
    test_loader = bodo.ai.prepare_dataset(test_df, BATCH_SIZE, dataset_func=BertDataset)


    if model == None:
        return
    pytorch_rank = dist.get_rank()

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
        torch.distributed.checkpoint.save(
            {"model_state_dict": base_model.state_dict()},
            checkpoint_id=CHECKPOINT_DIR
        )
    
    if pytorch_rank == 0:
        print("Test: ")
    validation(model, test_loader, loss_fn)



if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_df, val_df, test_df = preprocess_datasets(tokenizer)
    bodo.ai.torch_train(train_main, train_df, val_df, test_df)
    os._exit(0)

