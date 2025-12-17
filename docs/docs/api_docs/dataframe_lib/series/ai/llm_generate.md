# bodo.pandas.BodoSeries.ai.llm_generate

```py
BodoSeries.ai.llm_generate(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    request_formatter: Callable[[str], str] | None = None,
    response_formatter: Callable[[str], str] | None = None,
    region: str | None = None,
    backend: Backend = Backend.OPENAI,
    **generation_kwargs
) -> BodoSeries
```

Each element in the series is passed to the specified Large Language Model (LLM) backend for text generation.
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

: __**generation_kwargs: *dict*
    Additional keyword arguments to pass to the backend generation API (e.g., max_tokens, temperature).

<p class="api-header">Returns</p>

: __BodoSeries__: A series containing the generated text from the selected backend.

<p class="api-header">Example — OpenAI-compatible backend</p>

```py
import bodo.pandas as pd
from bodo.ai.backend import Backend

a = pd.Series([
    "What is the capital of France?",
    "Who wrote 'To Kill a Mockingbird'?",
    "What is the largest mammal?",
])

b = a.ai.llm_generate(
    api_key="your_api_key_here",
    model="gpt-3.5-turbo",
    base_url="https://api.example.com/v1",
    backend=Backend.OPENAI,
    max_tokens=50,
)
print(b)
```

```
Output:

0    The capital of France is Paris.
1    'To Kill a Mockingbird' was written by Harper Lee.
2    The largest mammal is the blue whale.
dtype: string[pyarrow]
```

<p class="api-header">Example — Amazon Bedrock backend</p>

```py
import bodo.pandas as pd
from bodo.ai.backend import Backend

a = pd.Series([
    "What is the capital of France?",
    "Who wrote 'To Kill a Mockingbird'?",
    "What is the largest mammal?",
])

b = a.ai.llm_generate(
    model="amazon.nova-micro-v1:0",
    backend=Backend.BEDROCK,
    region="us-east-1"
)

print(b)
```


<p class="api-header">Example — Amazon Bedrock backend with custom formatters</p>

```py
import bodo.pandas as pd
from bodo.ai.backend import Backend

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

a = pd.Series([
    "What is the capital of France?",
    "Who wrote 'To Kill a Mockingbird'?",
    "What is the largest mammal?",
])

b = a.ai.llm_generate(
    model="custom_model_id",
    backend=Backend.BEDROCK,
    region="us-east-1"
    request_formatter=request_formatter,
    response_formatter=response_formatter,
)

print(b)
```

