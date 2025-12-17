# bodo.pandas.BodoSeries.ai.embed

```py
BodoSeries.ai.embed(
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        request_formatter: Callable[[str], str] | None = None,
        response_formatter: Callable[[str], list[float]] | None = None,
        region: str | None = None,
        backend: Backend = Backend.OPENAI,
        **embedding_kwargs) -> BodoSeries
```

Embed a series of strings using the specified embedding backend.

Supports OpenAI-compatible endpoints and Amazon Bedrock via the backend parameter.

<p class="api-header">Parameters</p>

: __api_key: *str | None*__: The API key for authentication. Required for OpenAI backend. Must not be passed for Bedrock backend.

: __model: *str | None:*__
    The model to use for generation. If None, the backend's default model will be used. If the backend is Bedrock, this should be the model ID (e.g., "us.amazon.nova-lite-v1:0") and may not be None. For OpenAI, this should be the model name (e.g., "gpt-3.5-turbo").

: __base_url: *str | None:*__
    The URL of an OpenAI-compatible LLM endpoint (only applies to OpenAI-style backends).

: __request_formatter: *Callable[[str], str] | None:*__
    Optional function to format the input text before sending to the model. This is only used for the Bedrock backend and must not be passed otherwise.

    If None, a default formatter will be used for supported backends (e.g., Nova, Titan, Claude, OpenAI).

    For unsupported/custom models, this must be provided.

: __response_formatter: *Callable[[str], str] | None*:__
    Optional function to format the model's raw response into a string. This is only used for the Bedrock backend and must not be passed otherwise.

    If None, a default formatter will be used for supported backends.

    For unsupported/custom models, this must be provided.

: __region: *str | None:*__
    The AWS region where the Bedrock model is hosted (only applies to Bedrock backend).
    If None, the default configured region will be used.

: __backend: *bodo.ai.backend.Backend:*__
    The backend to use for generation. Currently supports:

        bodo.ai.backend.Backend.OPENAI – for OpenAI-compatible endpoints

        bodo.ai.backend.Backend.BEDROCK – for Amazon Bedrock models
: __**embedding_kwargs: *dict*__: Additional keyword arguments for the embedding API.
<p class="api-header">Returns</p>
: __BodoSeries__: A series containing the embedded vectors as lists of doubles. 

<p class="api-header">Example — OpenAI-compatible backend</p>

```py
import bodo.pandas as pd
from bodo.ai.backend import Backend

# Example series
a = pd.Series(["bodo.ai will improve your workflows.", "This is a professional sentence."])
# Define the LLM base_url and API key
base_url = "https://api.example.com/v1"
api_key = "your_api_key_here"
# Embed the series using the model
b = a.ai.embed(
    api_key=api_key,
    model="text-embedding-3-small",
    base_url=base_url,
    backend=Backend.OPENAI
)
print(b)
```


```
Output:

0    [0.123, 0.456, 0.789, ...]
1    [0.234, 0.567, 0.890, ...]
dtype: list<item: float64>[pyarrow]
```

<p class="api-header">Example — Amazon Bedrock backend</p>

```py
import bodo.pandas as pd
from bodo.ai.backend import Backend

# Example series
a = pd.Series(["bodo.ai will improve your workflows.", "This is a professional sentence."])
# Generate embeddings using the Bedrock model
b = a.ai.embed(
    model="amazon.titan-embed-text-v2:0",
    backend=Backend.BEDROCK,
    region="us-west-2"
)
print(b)
```


<p class="api-header">Example — Amazon Bedrock backend with custom formatters</p>

```py
import bodo.pandas as pd
from bodo.ai.backend import Backend

def request_formatter(row: str) -> str:
    return json.dumps({"inputText": row})

def response_formatter(response: str) -> list[float]:
    return json.loads(response)["embedding"]

a = pd.Series([
    "What is the capital of France?",
    "Who wrote 'To Kill a Mockingbird'?",
    "What is the largest mammal?",
])

b = a.ai.embed(
    model="custom_embedding_model_id",
    backend=Backend.BEDROCK,
    region="us-east-1"
    request_formatter=request_formatter,
    response_formatter=response_formatter,
)

print(b)
```

