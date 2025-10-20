import bodo.pandas as pd
import bodo.ai
import torch
import torch.distributed as dist
import torch.distributed.checkpoint
import tqdm
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_cosine_schedule_with_warmup
import os

LR = 2e-5
EPOCHS = 5
NUM_CLASSES = 5
SEQ_LENGTH = 512
BATCH_SIZE = 2
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


def process_dataset(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    labels = {'business':0,
        'entertainment':1,
        'sport':2,
        'tech':3,
        'politics':4
        }

    # Remove duplicates and extra whitespace
    df = df.drop_duplicates()
    df["text"] = df.text.str.lower().str.replace(r"\s+", " ", regex=True)

    def map_tokenizer(x):
        tokenized = tokenizer(x, max_length=SEQ_LENGTH, truncation=True, padding='max_length')
        return {"input_ids": tokenized.input_ids, "attention_mask": tokenized.attention_mask}

    df["label"] = df.category.map(labels)
    df["tokenized"] = df.text.map(map_tokenizer, engine='python')
    return df


def preprocess_datasets(tokenizer):
    test_df = pd.read_parquet("test.parquet")
    train_df = pd.read_parquet("train.parquet")[:10]
    val_df = pd.read_parquet("val.parquet")

    test_df = process_dataset(test_df, tokenizer)
    train_df = process_dataset(train_df, tokenizer)
    val_df = process_dataset(val_df, tokenizer)

    # remove test examples from train
    train_df = train_df[~train_df.text.isin(test_df.text)]

    return train_df, val_df, test_df


def validation(model, val_loader):
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
            input_id = input_id.to(device)
            mask = mask.to(device)
            val_label = val_label.to(device)

            output = model(input_id, token_type_ids=None,
                                    attention_mask=mask,
                                    labels=val_label)

            batch_loss = output.loss
            
            # Track local loss and accuracy stats
            acc = (output.logits.argmax(dim=1) == val_label).sum().item()
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


def train_one_epoch(model, train_loader, optimizer, scheduler):
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
        input_id = input_id.to(device)
        mask = mask.to(device)
        train_label = train_label.to(device)
        optimizer.zero_grad()

        output = model(input_id, token_type_ids=None, 
                                 attention_mask=mask,
                                 labels=train_label)

        batch_loss = output.loss

        # Track local loss and accuracy stats
        acc = (output.logits.argmax(dim=1) == train_label).sum().item()
        total_ddp_acc_loss_train[0] += acc
        total_ddp_acc_loss_train[1] += batch_loss.item()
        total_ddp_acc_loss_train[2] += train_label.size(0)

        batch_loss.backward()
        optimizer.step()
        scheduler.step()
        scheduler.step()
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

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = NUM_CLASSES, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    model = bodo.ai.prepare_model(model)

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
    total_steps = EPOCHS * len(train_loader)
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=10,
                                            num_training_steps=total_steps)

    for epoch in range(EPOCHS):

        if pytorch_rank == 0:
            print(f"Train Epoch: \t{epoch}")
        train_one_epoch(model, train_loader, optimizer, scheduler)

        if pytorch_rank == 0:
            print("Validation: ")
        validation(model, val_loader)

        # Checkpoint every epoch
        base_model = (model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model)
        torch.distributed.checkpoint.save(
            {"model_state_dict": base_model.state_dict()},
            checkpoint_id=CHECKPOINT_DIR
        )
    
    if pytorch_rank == 0:
        print("Test: ")
    validation(model, test_loader)



if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_df, val_df, test_df = preprocess_datasets(tokenizer)
    bodo.ai.torch_train(train_main, train_df, val_df, test_df)
    os._exit(0)

