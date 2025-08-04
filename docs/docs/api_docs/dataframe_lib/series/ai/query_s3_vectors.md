# bodo.pandas.BodoSeries.ai.query_s3_vectors
``` py
BodoSeries.query_s3_vectors(
        self,
        vector_bucket_name: str,
        index_name: str,
        topk: int,
        region: str = None,
        filter: dict = None,
        return_distance: bool = False,
        return_metadata: bool = False,
    ) -> BodoDataFrame
```
Query S3 vector index and return data of matching vectors in vector index.
See [S3 documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-getting-started.html) for more details.

<p class="api-header">Parameters</p>

: __vector_bucket_name: *str*:__ S3 Vectors bucket name to use.
: __index_name : *str*:__ S3 Vectors index name to use.
: __topk : *int*:__ Number of results to return.
: __region : *str, optional*:__ Region of S3 Vector bucket.
: __filter : *dict, optional*:__ Metadata filter to apply during the query.
: __return_distance : *bool, optional*:__ Whether to include the computed distance in the response.
: __return_metadata : *bool, optional*:__ Whether to include the metadata in the response.

<p class="api-header">Returns</p>

: __BodoDataFrame__

<p class="api-header">Example</p>

``` py
import pandas as pd
import bodo.pandas as bd
import boto3 
import json 

bedrock = boto3.client("bedrock-runtime", region_name="us-east-2")

input_text = "adventures in space"

response = bedrock.invoke_model(
    modelId="amazon.titan-embed-text-v2:0",
    body=json.dumps({"inputText": input_text})
)

model_response = json.loads(response["body"].read())
embedding = model_response["embedding"]

df = pd.DataFrame({"data": [embedding]*10})
bdf = bd.from_pandas(df)

out = bdf.data.ai.query_s3_vectors(
    vector_bucket_name="my-test-vector",
    index_name="my-test-ind",
    region="us-east-2",
    topk=3,
    filter={"genre": "scifi"},
    return_distance=True,
    return_metadata=True,
)
print(out)
```

Output:
```
                            keys              distances                                           metadata
0  ['Star Wars' 'Jurassic Park']  [0.7918925 0.8599859]  ["{'source_text': 'Star Wars: A farm boy joins...
1  ['Star Wars' 'Jurassic Park']  [0.7918925 0.8599859]  ["{'genre': 'scifi', 'source_text': 'Star Wars...
2  ['Star Wars' 'Jurassic Park']  [0.7918925 0.8599859]  ["{'genre': 'scifi', 'source_text': 'Star Wars...
3  ['Star Wars' 'Jurassic Park']  [0.7918925 0.8599859]  ["{'source_text': 'Star Wars: A farm boy joins...
4  ['Star Wars' 'Jurassic Park']  [0.7918925 0.8599859]  ["{'source_text': 'Star Wars: A farm boy joins...
5  ['Star Wars' 'Jurassic Park']  [0.7918925 0.8599859]  ["{'genre': 'scifi', 'source_text': 'Star Wars...
6  ['Star Wars' 'Jurassic Park']  [0.7918925 0.8599859]  ["{'genre': 'scifi', 'source_text': 'Star Wars...
7  ['Star Wars' 'Jurassic Park']  [0.7918925 0.8599859]  ["{'source_text': 'Star Wars: A farm boy joins...
8  ['Star Wars' 'Jurassic Park']  [0.7918925 0.8599859]  ["{'genre': 'scifi', 'source_text': 'Star Wars...
9  ['Star Wars' 'Jurassic Park']  [0.7918925 0.8599859]  ["{'genre': 'scifi', 'source_text': 'Star Wars...
```
---
