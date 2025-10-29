import bodo.pandas as pd

# Use pandas for timestamp manipulations
import pandas
import bodo.pandas as pd
import bodo.ai
import torch
import torch.distributed as dist
import torch.distributed.checkpoint
import tqdm
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    get_peft_model_state_dict
)

# --- Configuration for the filter ---
RECENT_DAYS = 1

# --- Configuration for data loading ---
USER_TABLE = "chat_analytics.user_messages"
BOT_TABLE = "chat_analytics.bot_messages"
S3_LOCATION = "s3://bodo-iceberg-training-demo"

# --- Model Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"

LR = 2e-5
EPOCHS = 1
BATCH_SIZE = 2
CHECKPOINT_DIR = "./llama3_lora_checkpoint_dir"

def load_data():
    # Load Data from the iceberg table in S3
    print(f"--- Loading data from {S3_LOCATION} ---")
    print(f"Filtering for 'liked' messages in the last {RECENT_DAYS} day(s)...")
    
    user_df = pd.read_iceberg(USER_TABLE, location=S3_LOCATION)
    bot_df = pd.read_iceberg(BOT_TABLE, location=S3_LOCATION)
    
    # Define the "recent" cutoff time
    cutoff_date = pandas.Timestamp.now() - pandas.Timedelta(days=RECENT_DAYS)
    
    # Filter bot messages for "liked" feedback and recent timestamps
    liked_bot_messages_df = bot_df[
        (bot_df["feedback_status"] == "liked")
        & (bot_df["response_timestamp"] >= cutoff_date)
    ]
    
    # Use pd.merge() to join the two dataframes
    joined_df = pd.merge(
        user_df, liked_bot_messages_df, on=["conversation_id", "message_number"]
    )
    
    # Select just the relevant columns
    output_columns = ["message_text", "response_text"]
    final_df = joined_df[output_columns]
    return final_df


class LlamaDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_ids = torch.tensor(row["input_tokenized"])
        output_ids = torch.tensor(row["output_tokenized"])
        example = torch.cat([input_ids, output_ids])
        labels = torch.cat([torch.full_like(input_ids, -100), output_ids])
        attention_mask = torch.ones_like(example)

        return example, labels, attention_mask

    def __getitems__(self, idxs):
        input_ids = []
        labels = []
        attention_masks = []
        max_len = 0
        for idx in idxs:
            row = self.df.iloc[idx]
            input_id = torch.tensor(row["input_tokenized"])
            output_id = torch.tensor(row["output_tokenized"])
            example_len = input_id.size(0) + output_id.size(0)
            if example_len > max_len:
                max_len = example_len
        for idx in idxs:
            row = self.df.iloc[idx]
            input_id = torch.tensor(row["input_tokenized"])
            output_id = torch.tensor(row["output_tokenized"])
            example = torch.cat([input_id, output_id])
            pad_len = max_len - example.size(0)
            label = torch.cat([torch.full_like(input_id, -100), output_id])
            attention_mask = torch.ones_like(example)
            if pad_len > 0:
                pad_tensor = torch.full((pad_len,), fill_value=0, dtype=torch.long)
                example = torch.cat([example, pad_tensor])
                label = torch.cat([label, torch.full_like(pad_tensor, -100)])
                attention_mask = torch.cat([attention_mask, pad_tensor])

            input_ids.append(example)
            labels.append(label)
            attention_masks.append(attention_mask)
        return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_masks)

def train_one_epoch(model, train_loader, optimizer, scheduler):
    model.train()
    rank = dist.get_rank()
    device = next(model.parameters()).device
    total_ddp_acc_loss_train = torch.zeros(2).to(device)

    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue"
        )
        print(f"Training Steps: {len(train_loader)}")


    for input_ids, labels, mask in train_loader:
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        output = model(input_ids, labels=labels, attention_mask=mask)

        batch_loss = output.loss

        # Track local loss and accuracy stats
        total_ddp_acc_loss_train[0] += batch_loss.item() * BATCH_SIZE
        total_ddp_acc_loss_train[1] += BATCH_SIZE

        batch_loss.backward()
        optimizer.step()
        scheduler.step()
        if rank==0:
            inner_pbar.update(1)
    
    # Compute global loss and accuracy stats
    dist.all_reduce(total_ddp_acc_loss_train, op=dist.ReduceOp.SUM)
    avg_train_loss = total_ddp_acc_loss_train[0] / total_ddp_acc_loss_train[1]

    if rank == 0:
        inner_pbar.close()
        print(
                f" Loss: \t{avg_train_loss:.4f}"
            )

    return avg_train_loss


def train_main(train_df):

    # Load tokenizer here to get pad_token_id ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Llama models don't have a default pad token. Set it to EOS.
    tokenizer.pad_token = tokenizer.eos_token

    # --- CHANGED: Load Llama 3.1 for Sequence Classification ---
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16, # Use bfloat16 for memory efficiency
        pad_token_id=tokenizer.pad_token_id, # Set pad token ID in model config
    )

    # --- ADDED: LoRA Configuration ---
    print("Applying LoRA configuration...")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM", # Specify task type for classification
        r=16,                # Rank of the LoRA matrices (default 8 or 16)
        lora_alpha=32,       # Alpha scaling factor (often 2x rank)
        lora_dropout=0.1,    # Dropout
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"]
    )
    model = get_peft_model(model, peft_config)
    if bodo.get_rank() == 0: 
        model.print_trainable_parameters()
    # --- END ADDED ---

    model = bodo.ai.prepare_model(model)

    if model:
        device = next(model.parameters()).device
    else:
        device = None
    gpu_ranks = bodo.get_gpu_ranks()

    # Rebalance data (same as before)
    accelerators_used = len(gpu_ranks) != 0
    if accelerators_used:
        train_df = bodo.rebalance(train_df, dests=gpu_ranks, random=True, parallel=True)

    train_loader = bodo.ai.prepare_dataset(train_df, BATCH_SIZE, dataset_func=LlamaDataset, pin_memory=True)


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


        # --- CHANGED: Checkpoint only the LoRA adapter weights ---
        base_model = (model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model)
        
        # Get only the trainable (adapter) parameters
        adapter_state_dict = get_peft_model_state_dict(base_model)
        
        torch.distributed.checkpoint.save(
            {"model_state_dict": adapter_state_dict}, # Save only the adapter
            checkpoint_id=CHECKPOINT_DIR
        )
        
    # Save in peft-preferred format on rank 0
    if pytorch_rank == 0:
       base_model.save_pretrained(CHECKPOINT_DIR) # Saves adapter_config.json etc.
    



if __name__ == "__main__":
    # --- CHANGED: Load Llama 3.1 Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    data_df = load_data()
    train_df = pd.DataFrame({"input_tokenized": data_df["message_text"].ai.tokenize(tokenizer), "output_tokenized": data_df["response_text"].ai.tokenize(tokenizer)})

    bodo.ai.torch_train(train_main, train_df)
