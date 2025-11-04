import bodo.pandas as pd

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
CUTOFF_DATE = pd.Timestamp("2025-10-29 00:00:00")

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
    print(f"Filtering for 'liked' messages since {CUTOFF_DATE}...")
    
    user_df = pd.read_iceberg(USER_TABLE, location=S3_LOCATION)
    bot_df = pd.read_iceberg(BOT_TABLE, location=S3_LOCATION)

    # Filter bot messages for "liked" feedback and recent timestamps
    liked_bot_messages_df = bot_df[
        (bot_df["feedback_status"] == "liked")
        & (bot_df["response_timestamp"] >= CUTOFF_DATE)
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
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df = df
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        self.template = "User: {user_message}\nBot: {bot_response}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        user_message = row["message_text"]
        bot_response = row["response_text"]
        # Create the prompt for the model
        prompt = self.template.format(
            user_message=user_message, bot_response=bot_response
        )
        # Tokenize the prompt
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        example = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        # For causal LM, labels are the same as input_ids
        labels = example.clone()

        return example, labels, attention_mask

    def __getitems__(self, idxs):
        prompts = []
        for idx in idxs:
            row = self.df.iloc[idx]
            user_message = row["message_text"]
            bot_response = row["response_text"]
            prompt = self.template.format(
                user_message=user_message, bot_response=bot_response
            )
            prompts.append(prompt)
        encoding = self.tokenizer(
            prompts,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        labels = input_ids.clone()
        return input_ids, labels, attention_mask


def train_one_epoch(model, train_loader, optimizer, scheduler):
    model.train()
    prev_cache = model.config.use_cache
    model.config.use_cache = False
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
        mask = mask.to(device, non_blocking=True)

        output = model(input_ids, labels=labels, attention_mask=mask)

        batch_loss = output.loss

        # Track local loss and accuracy stats
        total_ddp_acc_loss_train[0] += batch_loss.item() * input_ids.size(0)
        total_ddp_acc_loss_train[1] += input_ids.size(0)

        optimizer.zero_grad()
        batch_loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

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
    model.config.use_cache = prev_cache

    return avg_train_loss


def train_main(train_df):

    # Load tokenizer here to get pad_token_id ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Llama models don't have a default pad token. Set it to EOS.
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16, # Use bfloat16 for memory efficiency
        pad_token_id=tokenizer.pad_token_id, # Set pad token ID in model config
    )

    print("Applying LoRA configuration...")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM", # Specify task type for classification
        r=16,                # Rank of the LoRA matrices (default 8 or 16)
        lora_alpha=32,       # Alpha scaling factor (often 2x rank)
        lora_dropout=0.1,    # Dropout
        target_modules=["q_proj", "k_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    if bodo.get_rank() == 0: 
        model.print_trainable_parameters()

    model = bodo.ai.prepare_model(model)

    if model:
        device = next(model.parameters()).device
    else:
        device = None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset_func = lambda df: LlamaDataset(df, tokenizer)
    train_loader = bodo.ai.prepare_dataset(train_df, BATCH_SIZE, dataset_func=dataset_func, pin_memory=True)


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


        # Checkpoint only the LoRA adapter weights using a distributed checkpoint
        # for each epoch
        base_model = (model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model)
        
        # Get only the trainable (adapter) parameters
        adapter_state_dict = get_peft_model_state_dict(base_model)
        
        torch.distributed.checkpoint.save(
            {"model_state_dict": adapter_state_dict}, # Save only the adapter
            checkpoint_id=CHECKPOINT_DIR
        )
        
    # Save in peft-preferred format on rank 0 to allow easy loading later
    if pytorch_rank == 0:
       base_model.save_pretrained(CHECKPOINT_DIR) # Saves adapter_config.json etc.
    



if __name__ == "__main__":

    train_df = load_data()
    bodo.ai.torch_train(train_main, train_df)
