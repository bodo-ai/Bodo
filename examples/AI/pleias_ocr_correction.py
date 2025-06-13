import bodo
import bodo.pandas as pd
from pandas import ArrowDtype
import pyarrow as pa
import os
from transformers import AutoTokenizer

batch_size = 350
model_name = "LLMDH/pleias_350m_ocr"
model_len = 2048
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

prompts = pd.read_parquet(
    "hf://datasets/LLMDH/English-PD-bad-OCR/**/*.parquet"
)


def tokenize(row):
    # Tokenizer needs to be re-instantiated per worker in distributed execution
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
        text_start + split_chunk[1:] + text_end + ocr_correction_start
        for split_chunk in split_row_encoded
    ]
    return split_row_encoded

def ocr_correction(prompts):
    from vllm import LLM, SamplingParams, TokensPrompt

    gpu_ranks = bodo.libs.distributed_api.get_gpu_ranks()
    assert bodo.get_rank() in gpu_ranks, (
        "Only use 1 bodo worker per GPU in the cluster, set workers with BODO_NUM_WORKERS environment variable"
    )
    llm = LLM(
        model=model_name,
        dtype="auto",
        # Bodo workers have issues with async runtime
        disable_async_output_proc=True,
        max_model_len=model_len,
    )
    sampling_params = SamplingParams(
        repetition_penalty=1,
        stop_token_ids=[2],  # 2 is <|endoftext|>
        max_tokens=model_len,
    )

    # Organize the split prompts into batches that fit within the batch size
    batches = [[]]
    prompt_idx = 0
    # Maps batch index and batch entry index to the original prompt indices
    batch_to_prompt_idx = [[]]
    while prompt_idx < len(prompts):
        split_encoded_prompts = prompts["split_encoded_prompts"][prompt_idx]

        split_prompt_idx = 0
        while split_prompt_idx < len(split_encoded_prompts):
            encoded_prompt = split_encoded_prompts[split_prompt_idx]
            gen_prompt = TokensPrompt(prompt_token_ids=encoded_prompt)

            # Append to current batch or start a new one
            if len(batches[-1]) < batch_size:
                batches[-1].append(gen_prompt)
                batch_to_prompt_idx[-1].append(prompt_idx)
            else:
                batches.append([gen_prompt])
                batch_to_prompt_idx.append([prompt_idx])
            split_prompt_idx += 1
        prompt_idx += 1

    text_results = pd.Series([""] * len(prompts), dtype=ArrowDtype(pa.large_string()))
    for i, batch in enumerate(batches):
        batch_result = llm.generate(prompts=batch, sampling_params=sampling_params)
        for j, result in enumerate(batch_result):
            text = "".join([output.text for output in result.outputs])
            # Add to the corresponding original prompt index (supports multiple fragments per row)
            text_results[batch_to_prompt_idx[i][j]] += text
        print(f"Finished batch {i + 1} of {len(batches)}")
    return text_results


prompts["split_encoded_prompts"] = prompts.apply(tokenize, axis=1)
prompts["corrected_text"] = prompts.map_partitions(ocr_correction)
prompts.to_parquet("corrected_text.parquet")
