# `pd.DataFrame.cov`


`pandas.DataFrame.cov(min_periods=None, ddof=1)`


### Supported Arguments

- `min_periods`: Integer

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [0.695, 0.478, 0.628], "B": [-0.695, -0.478, -0.628], "C": [0.07, -0.68, 0.193]})
...   return df.cov()
>>> f()
          A         B         C
A  0.012346 -0.012346  0.047577
B -0.012346  0.012346 -0.047577
C  0.047577 -0.047577  0.223293
```

