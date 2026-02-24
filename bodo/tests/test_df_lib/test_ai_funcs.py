from __future__ import annotations

import json
import pickle
import random
import re
import time

import boto3
import pandas as pd
import pyarrow as pa
import pytest
import requests

import bodo.ai.backend
import bodo.ai.train
import bodo.pandas as bd
from bodo.spawn.spawner import spawn_process_on_nodes
from bodo.tests.utils import _test_equal


def test_write_s3_vectors(datapath):
    """Test writing to S3 Vectors using Bodo DataFrame API."""
    texts = [
        "Star Wars: A farm boy joins rebels to fight an evil empire in space",
        "Jurassic Park: Scientists create dinosaurs in a theme park that goes wrong",
        "Finding Nemo: A father fish searches the ocean to find his lost son",
    ]
    embeddings_path = datapath("embeddings.pkl")

    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)

    df = pd.DataFrame(
        {
            "key": ["Star Wars", "Jurassic Park", "Finding Nemo"],
            "data": embeddings,
            "texts": texts,
        }
    )
    df["metadata"] = [
        {"source_text": texts[0], "genre": "scifi"},
        {"source_text": texts[1], "genre": "scifi"},
        {"source_text": texts[2], "genre": "family"},
    ]
    bdf = bd.from_pandas(df)
    region = "us-east-2"
    bucket_name = "ehsan-test-vector"
    s3vectors = boto3.client("s3vectors", region_name=region)

    index_name = f"test-index{random.randint(100000, 999999)}"
    s3vectors.create_index(
        vectorBucketName=bucket_name,
        indexName=index_name,
        dataType="float32",
        dimension=1024,
        distanceMetric="cosine",
    )

    try:
        bdf.to_s3_vectors(
            vector_bucket_name=bucket_name, index_name=index_name, region=region
        )
        # Verify write
        response = s3vectors.get_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            keys=["Star Wars", "Jurassic Park", "Finding Nemo"],
        )
        assert len(response["vectors"]) == 3, "Expected 3 vectors to be written"
    finally:
        # Clean up Index after the test
        s3vectors.delete_index(vectorBucketName=bucket_name, indexName=index_name)

    # Error checking
    with pytest.raises(ValueError, match="DataFrame must have columns"):
        bdf_invalid = bd.from_pandas(df.drop("key", axis=1))
        bdf_invalid.to_s3_vectors(
            vector_bucket_name=bucket_name, index_name=index_name, region=region
        )

    with pytest.raises(ValueError, match="column must be strings to write"):
        bdf_invalid = bd.from_pandas(df.assign(key=[1, 2, 3]))
        bdf_invalid.to_s3_vectors(
            vector_bucket_name=bucket_name, index_name=index_name, region=region
        )

    with pytest.raises(ValueError, match="column must be a list of floats"):
        bdf_invalid = bd.from_pandas(df.assign(data=[1, 2, 3]))
        bdf_invalid.to_s3_vectors(
            vector_bucket_name=bucket_name, index_name=index_name, region=region
        )

    with pytest.raises(ValueError, match="column must be a struct type"):
        bdf_invalid = bd.from_pandas(df.assign(metadata=[1, 2, 3]))
        bdf_invalid.to_s3_vectors(
            vector_bucket_name=bucket_name, index_name=index_name, region=region
        )


@pytest.mark.parametrize("return_distance", [True, False])
@pytest.mark.parametrize("return_metadata", [True, False])
def test_query_s3_vectors(datapath, return_distance, return_metadata):
    """Test querying S3 Vectors using Bodo Series AI API."""

    embedding_path = datapath("query_embedding.pkl")

    with open(embedding_path, "rb") as f:
        embedding = pickle.load(f)

    df = pd.DataFrame({"data": [embedding] * 10})
    bdf = bd.from_pandas(df)

    out_df = bdf.data.ai.query_s3_vectors(
        vector_bucket_name="ehsan-test-vector",
        index_name="test-ind",
        region="us-east-2",
        topk=3,
        filter={"genre": "scifi"},
        return_distance=return_distance,
        return_metadata=return_metadata,
    )

    assert isinstance(out_df, bd.BodoDataFrame), "Output should be a BodoDataFrame"
    expected_columns = ["keys"]
    if return_distance:
        expected_columns.append("distances")
    if return_metadata:
        expected_columns.append("metadata")

    pd.testing.assert_index_equal(out_df.columns, pd.Index(expected_columns))


def test_query_s3_vectors_error_checking():
    # Check data type errors
    df = pd.DataFrame({"data": [["ABC", "123"]] * 10})
    bdf = bd.from_pandas(df)

    with pytest.raises(
        TypeError, match=re.escape("expected list[float32] or list[float64]")
    ):
        bdf.data.ai.query_s3_vectors(
            vector_bucket_name="ehsan-test-vector",
            index_name="test-ind",
            region="us-east-2",
            topk=3,
            filter={"genre": "scifi"},
        )


@pytest.mark.parametrize("init_func", [True, False])
def test_tokenize(init_func):
    from transformers import AutoTokenizer

    a = pd.Series(
        [
            "bodo.ai will improve your workflows.",
            "This is a professional sentence.",
            "I am the third entry in this series.",
            "May the fourth be with you.",
        ]
    )
    ba = bd.Series(a)
    if init_func:
        tokenizer = lambda: AutoTokenizer.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    pd_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    b = a.map(lambda x: pd_tokenizer.encode(x, add_special_tokens=True))
    bb = ba.ai.tokenize(tokenizer)

    _test_equal(
        bb,
        b,
        check_pandas_types=False,
        reset_index=False,
        check_names=False,
    )


def get_ollama_models(url):
    for i in range(50):
        try:
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                return response
            else:
                time.sleep(3)
        except requests.exceptions.RequestException:
            time.sleep(3)
        if i == 49:
            raise AssertionError("Ollama server not available yet")


def wait_for_ollama(url):
    get_ollama_models(url)


def wait_for_ollama_model(url, model_name):
    for _ in range(20):
        models = get_ollama_models(url)
        if model_name in models.text:
            return
        else:
            time.sleep(3)
    raise AssertionError(
        f"Model {model_name} not found in Ollama server at {url} after waiting for 60 seconds"
    )


@pytest.mark.skip("TODO: Fix flakey test.")
@pytest.mark.jit_dependency
def test_llm_generate_ollama():
    prompts = bd.Series(
        [
            "bodo.ai will improve your workflows.",
            "This is a professional sentence.",
        ]
    )

    try:
        spawn_process_on_nodes(
            "docker run -v ollama:/root/.ollama -p 11434:11434 --name bodo_test_ollama ollama/ollama:latest".split(
                " "
            )
        )
        wait_for_ollama("http://localhost:11434")
        spawn_process_on_nodes(
            "docker exec bodo_test_ollama ollama run smollm:135m".split(" ")
        )
        wait_for_ollama_model("http://localhost:11434", "smollm:135m")
        res = prompts.ai.llm_generate(
            base_url="http://localhost:11434/v1",
            api_key="",
            model="smollm:135m",
            max_tokens=1,
            temperature=0.1,
        ).execute_plan()
        assert len(res) == 2
        assert all(isinstance(x, str) for x in res)

    finally:
        spawn_process_on_nodes("docker rm bodo_test_ollama -f".split(" "))


@pytest.mark.skip("TODO: Fix flakey test.")
@pytest.mark.jit_dependency
def test_embed_ollama():
    prompts = bd.Series(
        [
            "bodo.ai will improve your workflows.",
            "This is a professional sentence.",
        ]
    )

    try:
        spawn_process_on_nodes(
            "docker run -v ollama:/root/.ollama -p 11435:11434 --name bodo_test_ollama_embed ollama/ollama:latest".split(
                " "
            )
        )
        wait_for_ollama("http://localhost:11435")
        spawn_process_on_nodes(
            "docker exec bodo_test_ollama_embed ollama pull all-minilm:22m".split(" ")
        )
        wait_for_ollama_model("http://localhost:11435", "all-minilm:22m")
        res = prompts.ai.embed(
            base_url="http://localhost:11435/v1",
            api_key="",
            model="all-minilm:22m",
        ).execute_plan()
        assert len(res) == 2
        assert res.dtype.pyarrow_dtype.equals(pa.list_(pa.float64()))

    finally:
        spawn_process_on_nodes("docker rm bodo_test_ollama_embed -f".split(" "))


@pytest.mark.jit_dependency
def test_llm_generate_bedrock_custom_formatters():
    prompts = bd.Series(
        [
            "bodo.ai will improve your workflows.",
            "This is a professional sentence.",
            "I am the third entry in this series.",
            "May the fourth be with you.",
        ]
    )

    def request_formatter(row: str) -> str:
        return json.dumps(
            {
                "system": [
                    {
                        "text": "Act as a creative writing assistant. When the user provides you with a topic, write a short story about that topic."
                    }
                ],
                "inferenceConfig": {
                    "maxTokens": 500,
                    "topP": 0.9,
                    "topK": 20,
                    "temperature": 0.7,
                },
                "messages": [{"role": "user", "content": [{"text": row}]}],
            }
        )

    def response_formatter(response: str) -> str:
        return json.loads(response)["output"]["message"]["content"][0]["text"]

    res = prompts.ai.llm_generate(
        model="us.amazon.nova-micro-v1:0",
        request_formatter=request_formatter,
        response_formatter=response_formatter,
        region="us-east-2",
        backend=bodo.ai.backend.Backend.BEDROCK,
    ).execute_plan()

    assert len(res) == 4
    assert all(isinstance(x, str) for x in res)


@pytest.mark.jit_dependency
@pytest.mark.parametrize(
    "modelId",
    [
        "us.amazon.nova-lite-v1:0",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ],
)
def test_llm_generate_bedrock_default_formatter(modelId):
    prompts = bd.Series(
        [
            "bodo.ai will improve your workflows.",
            "This is a professional sentence.",
            "I am the third entry in this series.",
            "May the fourth be with you.",
        ]
    )

    res = prompts.ai.llm_generate(
        model=modelId,
        region="us-east-1",
        backend=bodo.ai.backend.Backend.BEDROCK,
    ).execute_plan()

    assert len(res) == 4
    assert all(isinstance(x, str) for x in res)


@pytest.mark.jit_dependency
def test_embed_bedrock_custom_formatters():
    prompts = bd.Series(
        [
            "bodo.ai will improve your workflows.",
            "This is a professional sentence.",
            "I am the third entry in this series.",
            "May the fourth be with you.",
        ]
    )

    def request_formatter(row: str) -> str:
        return json.dumps({"inputText": row})

    def response_formatter(response: str) -> list[float]:
        return json.loads(response)["embedding"]

    res = prompts.ai.embed(
        model="amazon.titan-embed-text-v2:0",
        request_formatter=request_formatter,
        response_formatter=response_formatter,
        region="us-east-2",
        backend=bodo.ai.backend.Backend.BEDROCK,
    ).execute_plan()

    assert len(res) == 4
    assert res.dtype.pyarrow_dtype.equals(pa.list_(pa.float64()))


@pytest.mark.jit_dependency
def test_embed_bedrock_default_formatter():
    prompts = bd.Series(
        [
            "bodo.ai will improve your workflows.",
            "This is a professional sentence.",
            "I am the third entry in this series.",
            "May the fourth be with you.",
        ]
    )

    res = prompts.ai.embed(
        model="amazon.titan-embed-text-v2:0",
        region="us-east-1",
        backend=bodo.ai.backend.Backend.BEDROCK,
    ).execute_plan()

    assert len(res) == 4
    assert res.dtype.pyarrow_dtype.equals(pa.list_(pa.float64()))


@pytest.mark.skip("TODO: Enable when PyTorch is available on Python 3.14.")
@pytest.mark.jit_dependency
def test_torch_train():
    import tempfile

    df = bd.DataFrame(
        {
            "feature1": pd.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype="float32"),
            "feature2": pd.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype="float32"),
            "label": pd.array([3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0], dtype="float32"),
        }
    )

    def train_loop(data, config):
        import torch
        import torch.distributed.checkpoint
        import torch.nn as nn

        # Simple linear regression model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(2, 32)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(32, 1)

            def forward(self, x):
                return self.linear2(self.relu(self.linear1(x)))

        model = SimpleModel()
        model = bodo.ai.train.prepare_model(model, parallel_strategy="ddp")
        dataloader = bodo.ai.train.prepare_dataset(
            data, batch_size=config.get("batch_size", 2)
        )
        if model is None:
            return

        device = next(model.parameters()).device
        # train on data
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for epoch in range(config.get("epochs", 5)):
            for batch in dataloader:
                batch = batch.to(device, non_blocking=True)
                inputs = batch[:, :2]
                labels = batch[:, 2].unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # Create checkpoint.
            base_model = (
                model.module
                if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model
            )
            torch.distributed.checkpoint.save(
                {"model_state_dict": base_model.state_dict()},
                checkpoint_id=config["checkpoint_dir"],
            )

    bodo.ai.train.torch_train(
        train_loop,
        df,
        {"batch_size": 2, "checkpoint_dir": tempfile.mkdtemp("checkpoint_dir")},
    )
