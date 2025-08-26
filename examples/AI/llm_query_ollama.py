"""This example demonstrates preprocessing data and querying an LLM using Bodo"""

import pandas as pd
import bodo
import ollama
import time

MODEL = "llama3.2"


@bodo.wrap_python(bodo.types.string_type)
def query_model(prompt):
    """
    Sends a prompt to the Ollama model and returns the response.
    """
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


@bodo.jit
def query_model_all(df):
    """Clean up prompts and query the model for all prompts in the dataframe."""
    t0 = time.time()
    cleaned_prompts = df["prompt"].str.strip().str.lower()
    df["response"] = cleaned_prompts.map(query_model)
    print("Processing time:", time.time() - t0)
    return df


if __name__ == "__main__":
    raw_prompts = [
        " What is the capital of France? ",
        " What is the capital of Germany? ",
        " What is the capital of Spain? ",
    ]

    # Repeat prompts 10 times for testing purposes
    df = pd.DataFrame({"prompt": raw_prompts * 10})
    out_df = query_model_all(df)
    print(out_df)
