import bodo.pandas as pd
from transformers import AutoTokenizer

model_name = "LLMDH/pleias_350m_ocr"
model_len = 2048
split_encoded_path = None # Path to save the tokenized output, must be a location all nodes can access
assert split_encoded_path is not None, "Please set the split_encoded_path variable to a valid path for saving the tokenized output."

tokenizer = None

def tokenize(row):
    # Tokenizer needs to be re-instantiated per worker in distributed execution
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Special token sequences marking different sections of the input
    text_start = [2, 65522]
    text_end = [65523]
    ocr_correction_start = [65528]

    # Extra tokens used reduce available model context
    extra_tokens = len(text_start) + len(text_end) + len(ocr_correction_start)

    # Tokenize the input text from the current row
    row_encoded = tokenizer.encode(row.text)

    # Limit each chunk to half the model context minus reserved special tokens
    num_row_tokens = (model_len // 2) - extra_tokens

    # Split the tokenized row into multiple chunks
    split_row_encoded = [
        row_encoded[i : i + num_row_tokens]
        for i in range(0, len(row_encoded), num_row_tokens)
    ]

    # Add section markers around each chunk (drop the stop token from the tokenizer output since it's in the text_start)
    split_row_encoded = [
        text_start + (split_row_encoded_chunk[1:] if split_row_encoded_chunk[0] == 2 else split_row_encoded_chunk) + text_end + ocr_correction_start
        for split_row_encoded_chunk in split_row_encoded
    ]
    return split_row_encoded


if __name__ == "__main__":
    prompts = pd.read_parquet(
        "hf://datasets/LLMDH/English-PD-bad-OCR/**/*.parquet"
    )
    prompts["split_encoded_prompts"] = prompts.apply(tokenize, axis=1)
    prompts.to_parquet(split_encoded_path)
