from __future__ import annotations

import pickle
import random

import boto3
import pandas as pd

import bodo.pandas as bd


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
        bdf.to_s3_vectors(vector_bucket_name=bucket_name, index_name=index_name)
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
