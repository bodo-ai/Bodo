from __future__ import annotations

import pickle
import random
import re
import time

import boto3
import pandas as pd
import pyarrow as pa
import pytest
import requests

import bodo.pandas as bd
from bodo.spawn.spawner import spawn_process_on_workers
from bodo.tests.utils import _test_equal
from bodo.utils.typing import BodoError


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
    with pytest.raises(BodoError, match="DataFrame must have columns"):
        bdf_invalid = bd.from_pandas(df.drop("key", axis=1))
        bdf_invalid.to_s3_vectors(
            vector_bucket_name=bucket_name, index_name=index_name, region=region
        )

    with pytest.raises(BodoError, match="column must be strings to write"):
        bdf_invalid = bd.from_pandas(df.assign(key=[1, 2, 3]))
        bdf_invalid.to_s3_vectors(
            vector_bucket_name=bucket_name, index_name=index_name, region=region
        )

    with pytest.raises(BodoError, match="column must be a list of floats"):
        bdf_invalid = bd.from_pandas(df.assign(data=[1, 2, 3]))
        bdf_invalid.to_s3_vectors(
            vector_bucket_name=bucket_name, index_name=index_name, region=region
        )

    with pytest.raises(BodoError, match="column must be a struct type"):
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


def test_tokenize():
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

    def ret_tokenizer():
        # Load a pretrained tokenizer (e.g., BERT)
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    pd_tokenizer = ret_tokenizer()
    b = a.map(lambda x: pd_tokenizer.encode(x, add_special_tokens=True))
    bb = ba.ai.tokenize(ret_tokenizer)

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


def test_llm_generate():
    prompts = bd.Series(
        [
            "bodo.ai will improve your workflows.",
            "This is a professional sentence.",
        ]
    )

    try:
        spawn_process_on_workers(
            "docker run -v ollama:/root/.ollama -p 11434:11434 --name bodo_test_ollama ollama/ollama:latest".split(
                " "
            )
        )
        wait_for_ollama("http://localhost:11434")
        spawn_process_on_workers(
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
        spawn_process_on_workers("docker rm bodo_test_ollama -f".split(" "))


def test_embed():
    prompts = bd.Series(
        [
            "bodo.ai will improve your workflows.",
            "This is a professional sentence.",
        ]
    )

    try:
        spawn_process_on_workers(
            "docker run -v ollama:/root/.ollama -p 11435:11434 --name bodo_test_ollama_embed ollama/ollama:latest".split(
                " "
            )
        )
        wait_for_ollama("http://localhost:11435")
        spawn_process_on_workers(
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
        spawn_process_on_workers("docker rm bodo_test_ollama_embed -f".split(" "))
