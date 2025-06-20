
from pandas import ArrowDtype
import pyarrow as pa
import bodo
import bodo.pandas as pd
import os

batch_size = 350
model_name = "LLMDH/pleias_350m_ocr"
model_len = 2048
split_encoded_path = None  # Path to the tokenized input, must be a location all nodes can access
corrected_path = None  # Path to save the corrected output, must be a location all nodes can access
assert split_encoded_path is not None, "Please set the split_encoded_path variable to a valid path for the tokenized input."
assert corrected_path is not None, "Please set the corrected_path variable to a valid path for saving the corrected output."

def ocr_correction(prompts):
    # Get the data from ranks not assigned to gpus on the gpu ranks
    gpu_ranks = bodo.libs.distributed_api.get_gpu_ranks()
    received_prompts = bodo.rebalance(prompts, dests=gpu_ranks, parallel=True)

    if received_prompts is None or len(received_prompts) == 0:
        # Just return an empty series from non-gpu ranks
        return pd.Series([], dtype=ArrowDtype(pa.large_string()))

    from vllm import LLM, SamplingParams, TokensPrompt
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
        split_encoded_prompts = prompts["split_encoded_prompts"].iloc[prompt_idx]

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

    # TODO: Rebalance batches instead of prompts
    text_results = pd.Series([""] * len(prompts), dtype=ArrowDtype(pa.large_string()))
    for i, batch in enumerate(batches):
        batch_result = llm.generate(prompts=batch, sampling_params=sampling_params)
        for j, result in enumerate(batch_result):
            text = "".join([output.text for output in result.outputs])
            # Add to the corresponding original prompt index (supports multiple fragments per row)
            text_results[batch_to_prompt_idx[i][j]] += text
        print(f"Finished batch {i + 1} of {len(batches)}")
    received_prompts["corrected_text"] = text_results
    received_prompts.to_parquet(corrected_path + f"{bodo.get_rank}.pq")
    return text_results

if __name__ == "__main__":
    prompts = pd.read_parquet(split_encoded_path)
    prompts["corrected_text"] = prompts.map_partitions(ocr_correction)
