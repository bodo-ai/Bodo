# `pd.Index.names`


`pandasIndex.names`


!!! info "Important"
    Bodo returns a tuple instead of a FrozenList.


### Example Usage

```py
>>> @bodo.jit
... def f(I):
...   return I.names

>>> I = pd.MultiIndex.from_product([[1, 2], ["A", "B"]], names=["C1", "C2"])
>>> f(I)
('C1', 'C2')
```


