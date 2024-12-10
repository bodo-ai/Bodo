# `pd.Index.name`

`pandasIndex.name`

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.name

>>> I = pd.Index([1,2,3], name = "hello world")
>>> f(I)
"hello world"
```
