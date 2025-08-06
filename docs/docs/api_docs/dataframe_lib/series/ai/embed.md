# bodo.pandas.BodoSeries.ai.embed

```py
BodoSeries.ai.embed(
        endpoint: str,
        api_token: str,
        model: str | None = None,
        **embed_kwargs) -> BodoSeries
```
Embed a series of strings using an LLM endpoint.
<p class="api-header">Parameters</p>
- __endpoint: *str*__: The URL of the OpenAI compatible LLM endpoint.
- __api_token: *str*__: The API token for authentication with the LLM endpoint
- __model: *str | None*__: The model to use for embedding. If None is provided, the default model from the endpoint will be used.
- __**embed_kwargs: *dict*__: Additional keyword arguments for the LLM embedding API.
<p class="api-header">Returns</p>
- __BodoSeries__: A series containing the embedded vectors as lists of doubles.
<p class="api-header">Example</p>

```py
import bodo.pandas as pd
from bodo.pandas.ai import embed
# Example series
a = pd.Series(["bodo.ai will improve your workflows.", "This is a professional sentence."])
# Define the LLM endpoint and API token
endpoint = "https://api.example.com/v1"
api_token = "your_api_token_here"
# Embed the series using the LLM
b = a.ai.embed(endpoint, api_token, model="gpt-3.5-turbo")
print(b)
```

Output:
```
0    [0.123, 0.456, 0.789, ...]
1    [0.234, 0.567, 0.890, ...]
dtype: list<item: float64>[pyarrow]
```

---

