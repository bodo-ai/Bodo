# bodo.pandas.BodoSeries.ai.llm_generate

```py
BodoSeries.ai.llm_generate(
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
        **generation_kwargs) -> BodoSeries
```

Each element in the series is passed to the LLM endpoint for generation.

<p class="api-header">Parameters</p>
- __api_key: *str*__: The API key for authentication with the LLM endpoint
- __model: *str | None*__: The model to use for embedding. If None is provided, the default model from the endpoint will be used.
- __base_url: *str*__: The URL of the OpenAI compatible LLM endpoint.
- __**generation_kwargs: *dict*__: Additional keyword arguments for the LLM generation API.
<p class="api-header">Returns</p>
- __BodoSeries__: A series containing the generated text from the LLM.
<p class="api-header">Example</p>

```py
import bodo.pandas as pd

# Example series
a = pd.Series(["What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?", "What is the largest mammal?"])
# Define the LLM base_url and API key
base_url = "https://api.example.com/v1"
api_key = "your_api_key_here"
# Generate responses using the LLM
b = a.ai.llm_generate(api_key=api_key, model="gpt-3.5-turbo", base_url=base_url, max_tokens=50)
print(b)
```

Output:
```
0    The capital of France is Paris.
1    'To Kill a Mockingbird' was written by Harper Lee.
2    The largest mammal is the blue whale.
dtype: string[pyarrow]
```

---

# bodo.pandas.BodoSeries.ai.llm_generate_bedrock

```py
BodoSeries.ai.llm_generate_bedrock(
        modelId: str,
        request_formatter: Callable[[str], str] | None = None,
        response_formatter: Callable[[str], str] | None = None,
        region: str = None,
        **generation_kwargs) -> BodoSeries
```

Each element in the series is passed to the Amazon Bedrock API for generation.

<p class="api-header">Parameters</p>
- __modelId: *str*__: The ID of the Amazon Bedrock model to use for generation.
- __request_formatter: *Callable[[str], str] | None*__: A function to
    format the input text into the model's expected format before sending it to the model. If a Nova, Titan, Claude, or OpenAI model is used, this can be None and a default request formatter will be used. Otherwise, this must be provided.
- __response_formatter: *Callable[[str], str] | None*__: A function to
    format the model's response into a string. If a Nova, Titan, Claude, or OpenAI model is used, this can be None and a default response formatter will be used. Otherwise
    this must be provided.
- __region: *str*__: The AWS region where the Bedrock model is hosted. If not provided, the configured default region will be used.
- __**generation_kwargs: *dict*__: Additional keyword arguments for the Bedrock generation API.
<p class="api-header">Returns</p>
- __BodoSeries__: A series containing the generated text from the Bedrock model.

<p class="api-header">Example</p>

```py
import bodo.pandas as pd
from bodo.pandas.ai import llm_generate_bedrock

# Example series
a = pd.Series(
    [
        "What is the capital of France?",
        "Who wrote 'To Kill a Mockingbird'?",
        "What is the largest mammal?",
    ]
)
# Define the Bedrock model ID and region
modelId = "amazon.titan-text-lite-v1"
# Generate responses using the Bedrock model
b = a.ai.llm_generate_bedrock(
    modelId,
)
print(b)
```

---
