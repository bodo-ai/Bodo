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
