import pandas as pd
import bodo.ai
import torch
import torch.distributed as dist
import tqdm
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BertModel, BertTokenizer

LR = 1e-6
EPOCHS = 1
NUM_CLASSES = 5
SEQ_LENGTH = 512

class PandasDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.labels : pd.Series = df["label"]
        self.tokenized : pd.Series = df["tokenized"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        batch_texts = self.labels.iloc[idx]
        batch_y = self.tokenized.iloc[idx]

        return batch_y, batch_texts


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

    df.drop_duplicates(subset=["text"])
    df["text"] = df.text.str.lower().str.replace(r"\s+", " ", regex=True)

    def map_tokenizer(x):
        tokenized = tokenizer(x, max_length=SEQ_LENGTH, truncation=True, padding='max_length', return_tensors="pt")
        return {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask}

    df["label"] = df.category.map(labels)
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


def ddp_validation(model, val_loader, loss_fn):
    model.eval()
    rank = dist.get_rank()
    device = next(model.parameters()).device
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


def ddp_train_one_epoch(model, train_loader, loss_fn, optimizer, epoch):
    model.train()
    rank = dist.get_rank()
    device = next(model.parameters()).device
    total_ddp_acc_loss_train = torch.zeros(3).to(device)

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


def train_main(train_df, val_df, test_df):
    model = bodo.ai.prepare_model(BertClassifier())
    if model == None:
        return

    train_dataset = PandasDataset(train_df)
    val_dataset = PandasDataset(val_df)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=2, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=2, sampler=val_sampler)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        # TODO: checkpointing?
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        ddp_train_one_epoch(model, train_loader, loss_fn, optimizer, epoch)
        ddp_validation(model, val_loader, loss_fn)

    # TODO: Evaluate test dataset at the end of training
    


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_df, val_df, test_df = prepare_datasets(tokenizer)
    bodo.ai.torch_train(train_main, train_df, val_df, test_df)

