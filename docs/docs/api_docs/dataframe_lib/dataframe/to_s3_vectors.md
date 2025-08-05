# bodo.pandas.BodoDataFrame.to\_s3\_vectors
``` py
BodoDataFrame.to_s3_vectors(vector_bucket_name, index_name)
```
Write a DataFrame to an S3 Vectors index.
The DataFrame should have "key", "data" and "metadata" columns.
"key" column data should be strings, and "data" column should be float32
embeddings with the same length as expected by the vector index in each row.
"metadata" should be key-value pairs.
See [S3 documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-getting-started.html) for more details.

<p class="api-header">Parameters</p>

: __vector_bucket_name: *str*:__ S3 Vectors bucket name to use.

: __index_name : *str*:__ S3 Vectors index name to use.
: __region : *str, optional*:__ Region of S3 Vector bucket.

<p class="api-header">Example</p>

``` py
import pandas as pd
import bodo.pandas as bd
import boto3
import json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-2")
texts = [
    "Star Wars: A farm boy joins rebels to fight an evil empire in space", 
    "Jurassic Park: Scientists create dinosaurs in a theme park that goes wrong",
    "Finding Nemo: A father fish searches the ocean to find his lost son"
]

embeddings = []
for text in texts:
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text})
    )
    response_body = json.loads(response["body"].read())
    embeddings.append(response_body["embedding"])

df = pd.DataFrame({"key": ["Star Wars", "Jurassic Park", "Finding Nemo"],
                   "data": embeddings,
                   "texts": texts})
df["metadata"] = [
    {"source_text": texts[0], "genre": "scifi"},
    {"source_text": texts[1], "genre": "scifi"},
    {"source_text": texts[2], "genre": "family"}
]


bdf = bd.from_pandas(df)
bdf.to_s3_vectors(
    vector_bucket_name="my-test-vector",
    index_name="my-test-ind",
    region="us-east-2",
)

```

---
