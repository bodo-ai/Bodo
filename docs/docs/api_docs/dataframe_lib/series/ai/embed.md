# bodo.pandas.BodoSeries.ai.embed

```py
BodoSeries.ai.embed(
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        **embed_kwargs) -> BodoSeries
```
Embed a series of strings using an LLM endpoint.
<p class="api-header">Parameters</p>
- __api_key: *str*__: The API key for authentication with the LLM endpoint
- __model: *str | None*__: The model to use for embedding. If None is provided, the default model from the endpoint will be used.
- __base_url: *str*__: The URL of the OpenAI compatible LLM endpoint.
- __**embed_kwargs: *dict*__: Additional keyword arguments for the LLM embedding API.
<p class="api-header">Returns</p>
- __BodoSeries__: A series containing the embedded vectors as lists of doubles.
<p class="api-header">Example</p>

```py
import bodo.pandas as pd

# Example series
a = pd.Series(["bodo.ai will improve your workflows.", "This is a professional sentence."])
# Define the LLM base_url and API key
base_url = "https://api.example.com/v1"
api_key = "your_api_key_here"
# Embed the series using the LLM
b = a.ai.embed(api_key=api_key, model="text-embedding-3-small", base_url=base_url)
print(b)
```

Output:
```
0    [0.123, 0.456, 0.789, ...]
1    [0.234, 0.567, 0.890, ...]
dtype: list<item: float64>[pyarrow]
```

---

# bodo.pandas.BodoSeries.ai.embed_bedrock

```py
BodoSeries.ai.embed_bedrock(
        modelId: str,
        request_formatter: Callable[[str], str] | None = None,
        response_formatter: Callable[[str], str] | None = None,
        region: str = None,
        **embed_kwargs) -> BodoSeries
```
Embed a series of strings using the Amazon Bedrock API.
<p class="api-header">Parameters</p>
- __modelId: *str*__: The ID of the Amazon Bedrock model to use
    for embedding.
- __request_formatter: *Callable[[str], str] | None*__: A function to
    format the input text into the model's expected format before sending it to the model. If a  Titan embedding model is used, this can be None and a default request formatter will be used. Otherwise, this must be provided.
- __response_formatter: *Callable[[str], str] | None*__: A function to
    format the model's response into a string. If a Titan embedding model is used, this can be None and a default response formatter will be used. Otherwise, this must be provided.
- __region: *str*__: The AWS region where the Bedrock model is hosted. If None, the configured default region will be used.
<p class="api-header">Returns</p>
- __BodoSeries__: A series containing the embedded vectors as lists of doubles.
<p class="api-header">Example</p>

```py
import bodo.pandas as pd
from bodo.pandas.ai import embed_bedrock
# Example series
a = pd.Series(["bodo.ai will improve your workflows.", "This is a professional sentence."])
# Define the Bedrock model ID and formatters
modelId = "amazon.titan-embed-text-v2:0"
region = "us-west-2"
# Generate embeddings using the Bedrock model
b = a.ai.embed_bedrock(
    modelId,
)
print(b)
```

---
