# bodo.pandas.BodoSeries.ai.llm_generate

```py
BodoSeries.ai.llm_generate(
        endpoint: str,
        api_token: str,
        model: str | None = None,
        **generation_kwargs) -> BodoSeries
```

Each element in the series is passed to the LLM endpoint for generation.

<p class="api-header">Parameters</p>
- __endpoint: *str*__: The URL of the OpenAI compatible LLM endpoint.
- __api_token: *str*__: The API token for authentication with the LLM endpoint.
- __model: *str | None*__: The model to use for generation. If None
    is provided, the default model from the endpoint will be used.
- __**generation_kwargs: *dict*__: Additional keyword arguments for the LLM generation API.
<p class="api-header">Returns</p>
- __BodoSeries__: A series containing the generated text from the LLM.
<p class="api-header">Example</p>

```py
import bodo.pandas as pd
from bodo.pandas.ai import llm_generate
# Example series
a = pd.Series(["What is the capital of France?", "Who wrote 'To Kill a Mockingbird'?", "What is the largest mammal?"])
# Define the LLM endpoint and API token
endpoint = "https://api.example.com/v1"
api_token = "your_api_token_here"
# Generate responses using the LLM
b = a.ai.llm_generate(endpoint, api_token, model="gpt-3.5-turbo", max_tokens=50)
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
