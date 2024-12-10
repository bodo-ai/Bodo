# `pd.Index.to_frame`

`pandasIndex.to_frame(index=True, name=None)`

### Supported Arguments:

- `index`: can be a True or False
- `name`: can be a string or int

***Unsupported Index Types***

- IntervalIndex
- PeriodIndex

### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.to_frame(index=False)

>>> I = pd.Index(["A", "E", "I", "O", "U", "Y"], name="vowels")
>>> f(I)
  vowels
0      A
1      E
2      I
3      O
4      U
5      Y
```
