"""
This script does the following:

Dataset Loading: Uses the Hugging Face datasets library to load The Pile. It caches the dataset for reuse. Can make this just on disk.
Text Cleaning: Removes unnecessary whitespace and newline/tab characters.
Deduplication: Deduplicates text entries using MD5 hashes.
Tokenization: Uses a pretrained GPT-Neo tokenizer to tokenize the dataset into input IDs and attention masks, truncating or padding sequences to a fixed length. Can change to any other.
Saving Processed Data: Saves the processed and tokenized dataset in JSONL format, suitable for ML training pipelines.
"""


import hashlib
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from dataclasses import dataclass
import bodo
import time


@dataclass
class Config:
    dataset_name: str = "monology/pile-uncopyrighted"
    cache_dir: str = "./data"
    tokenizer_name: str = "EleutherAI/gpt-neo-2.7B"
    output_file: str = "./processed_dataset.jsonl"
    max_seq_length: int = 512


@bodo.wrap_python(bodo.types.string_type)
def hash_text(text):
    """Hash function for deduplication"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


@bodo.wrap_python(bodo.types.string_type)
def clean_text(text):
    text = text.replace('\n', ' ').replace('\t', ' ')
    # Remove extra spaces
    text = ' '.join(text.split())
    return text


tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token
tuple_arr_type = bodo.typeof((np.array([1, 2]), np.array([3, 4])))


@bodo.wrap_python(tuple_arr_type)
def run_tokenizer(text):
    tokenized = tokenizer(text, truncation=True, max_length=Config.max_seq_length, padding="max_length")
    return (np.array(tokenized["input_ids"]), np.array(tokenized["attention_mask"]))


@bodo.jit
def tokenize_data(row):
    text = row.text
    t = run_tokenizer(text)
    return pd.Series([text, t[0], t[1]], index=["text", "input_ids", "attention_mask"])


@bodo.jit
def preprocess_pile(df, out_file):
    t0 = time.time()
    df["text"] = df["text"].map(clean_text)
    df["text_hash"] = df["text"].map(hash_text)
    df = df.drop_duplicates(subset=["text_hash"])
    df = df.drop("text_hash", axis=1)
    processed_data = df.apply(tokenize_data, axis=1)
    processed_data.to_json(out_file, orient="records", lines=True)
    print("Execution Time:", time.time() - t0)


if __name__ == "__main__":
    print("Loading dataset...")
    dataset = load_dataset(Config.dataset_name, split="train", cache_dir=Config.cache_dir)
    df = dataset.to_pandas()
    preprocess_pile(df, Config.output_file)
