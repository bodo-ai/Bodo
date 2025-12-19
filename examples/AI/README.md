# AI Examples

This folder contains various examples demonstrating the use of AI with Bodo. 

## Examples

### 1.  More efficient preprocessing of large text datasets for LLM training.

- **File**: `preprocess_thepile_bodo.py` 

- **Description**: This script preprocesses a large text dataset (The Pile) to train large language models (LLMs). It cleans, deduplicates, tokenizes the text, and saves the processed data in a format suitable for machine learning.

- **Bodo Benefits**: Efficient preprocessing is crucial for training LLMs. This script uses Bodo for parallel processing, drastically speeding up the preparation of massive datasets, which would otherwise be a major bottleneck.

### 2. More efficient preprocessing data for local model communication. 

- **File**: `llm_query_ollama.py`

- **Description:** This example script uses the Python Ollama library to communicate with local models and shows an example of how-to Bodo-ify the LLM query process.

- **Bodo Benefits:** This illustrates how to preprocess data and utilize an LLM using Bodo, and showcases the syntax for doing so, as well as a starting point for integrating Bodo into LLM workflows.

### 3.  Accelerating inference speeds driving  lower latency for LLM queries.

- **File**: `llm_inference_speedtest.ipynb`

- **Description:** This Jupyter Notebook contains two versions of a simple LLM querying script: a standard Python version and a Bodo-optimized version. The notebook compares the performance of querying an LLM using native Python (with the llm package and Gemini model) against Bodo's accelerated process.

- **Bodo Benefits:** This demonstrates the potential speed improvements Bodo can bring to LLM inference.  Faster inference means lower latency and potentially lower costs for real-world LLM. applications.