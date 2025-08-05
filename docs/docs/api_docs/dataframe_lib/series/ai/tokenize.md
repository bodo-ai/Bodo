# bodo.pandas.BodoSeries.ai.tokenize
```
BodoSeries.ai.tokenize(tokenizer) -> BodoSeries

```
Tokenize a series of string dtype into a series of lists of int64.

<p class="api-header">Parameters</p>

: __tokenizer: *function*:__ A function returning a Transformers.PreTrainedTokenizer.

<p class="api-header">Returns</p>

: __BodoSeries__

<p class="api-header">Example</p>

``` py
import bodo.pandas as pd
from transformers import AutoTokenizer

a = pd.Series(["bodo.ai will improve your workflows.", "This is a professional sentence.", "I am the third entry in this series.", "May the fourth be with you."])
def ret_tokenizer():
    # Load a pretrained tokenizer (e.g., BERT)
    return AutoTokenizer.from_pretrained("bert-base-uncased")
b = a.ai.tokenize(ret_tokenizer)
print(b)
```

Output:
```
0    [  101 28137  1012  9932  2097  5335  2115  21...
1            [ 101 2023 2003 1037 2658 6251 1012  102]
2    [ 101 1045 2572 1996 2353 4443 1999 2023 2186 ...
3       [ 101 2089 1996 2959 2022 2007 2017 1012  102]
dtype: list<item: int64>[pyarrow]
```

---
