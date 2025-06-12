import bodo
import bodo.pandas as pd
import os
from transformers import AutoTokenizer

batch_size = 350
model_name = "LLMDH/pleias_350m_ocr"
model_len = 2048
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

prompts = pd.read_parquet("hf://datasets/LLMDH/English-PD-bad-OCR/*.parquet")

def tokenize(row):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_start = [2,65522]
    text_end = [65523]
    ocr_correction_start = [65528]
    extra_tokens = len(text_start) + len(text_end) + len(ocr_correction_start)
    row_encoded = tokenizer.encode(row.text)
    num_row_tokens = (model_len // 2) - extra_tokens
    split_row_encoded = [row_encoded[i:i+num_row_tokens] for i in range(0, len(row_encoded), num_row_tokens)]
    split_row_encoded = [text_start + split_row_encoded_chunk[1:] + text_end + ocr_correction_start for split_row_encoded_chunk in split_row_encoded]
    return split_row_encoded

def ocr_correction(prompts):
    from vllm import LLM, SamplingParams, TokensPrompt
    gpu_ranks = bodo.libs.distributed_api.get_gpu_ranks()
    assert bodo.get_rank() in gpu_ranks, "Only use 1 bodo worker per GPU in the cluster, set workers with BODO_NUM_WORKERS environment variable"
    llm = LLM(
        model=model_name,
        dtype="auto",
        # Bodo workers have issues with async runtime
        disable_async_output_proc=True,
        max_model_len=model_len//2,
    )
    sampling_params = SamplingParams(
        repetition_penalty=1,
        stop_token_ids=[2],  # 2 is <|endoftext|>
        max_tokens=model_len,
    )
    batches = [[]]
    prompt_idx = 0
    batch_idx_to_prompt_idx = []
    while prompt_idx < len(prompts):

        split_encoded_prompts = prompts["split_encoded_prompts"][prompt_idx]

        split_prompt_idx = 0
        while split_prompt_idx < len(split_encoded_prompts):
            encoded_prompt = split_encoded_prompts[split_prompt_idx]
            gen_prompt = TokensPrompt(prompt_token_ids=encoded_prompt)
            if len(batches[-1]) < batch_size:
                batches[-1].append(gen_prompt)
            else:
                batches.append([gen_prompt])
            split_prompt_idx  += 1
        prompt_idx += 1
    
    for batch, i in enumerate(batches):
        batch_result = llm.generate(prompts=batch, sampling_params=sampling_params)
        print(f"Finished batch {i} of {len(batches)}")
    return None

prompts["split_encoded_prompts"] = prompts.apply(tokenize, axis=1)
prompts["corrected_text"] = prompts.map_partitions(ocr_correction)
