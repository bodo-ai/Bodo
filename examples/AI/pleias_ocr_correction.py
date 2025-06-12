import bodo.pandas as pd
import os
from vllm import LLM, SamplingParams, TextPrompt

model_name = "LLMDH/pleias_350m_ocr"
model_len = 2048
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

prompts = pd.read_parquet("hf://datasets/LLMDH/English-PD-bad-OCR/**/*.parquet")
prompts = prompts[:10]

text = f"<|text_start|>{row.text[0:1024]}<|text_end|><|ocr_correction_start|>"
def init_llm():
    return LLM(
        model=model_name,
        dtype="auto",
        # Bodo workers have issues with async runtime
        disable_async_output_proc=True,
        # Max length of input text in input file
        max_model_len=model_len,
    )

def tokenize(row):
    global llm
    if llm is None:
        llm  = init_llm()
    tokenizer = llm.get_tokenizer()
    text_start = tokenizer.encode("<|text_start|>")
    text_end = tokenizer.encode("<|text_end|>")
    ocr_correction_start = tokenizer.encode("<|ocr_correction_start|>")
    extra_tokens = len(text_start) + len(text_end) + len(ocr_correction_start)
    row_encoded = tokenizer.encode(row.text)
    num_row_tokens = model_len - extra_tokens
    split_row_encoded = [row_encoded[i:i+num_row_tokens] for i in range(len(row_encoded), num_row_tokens)]
    split_row_encoded = [text_start + split_row_encoded_chunk + text_end + ocr_correction_start for split_row_encoded_chunk in split_row_encoded]
    return split_row_encoded

split_encoded_prompts = prompts.apply(tokenize, axis=1)


#sampling_params = SamplingParams(
#    max_tokens=model_len,
#    repetition_penalty=1,
#    stop_token_ids=[2],  # Assuming 2 is <|endoftext|>
#)
#result = llm.generate(prompts=TextPrompt(prompt=text), sampling_params=sampling_params)
