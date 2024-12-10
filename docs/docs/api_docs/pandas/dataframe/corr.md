# `pd.DataFrame.corr`


`pandas.DataFrame.corr(method='pearson', min_periods=1)`


### Supported Arguments

- `min_periods`: Integer

### Example Usage

```py

>>> @bodo.jit
... def f():
...   df = pd.DataFrame({"A": [.9, .8, .7, .4], "B": [-.8, -.9, -.8, -.4], "c": [.7, .7, .7, .4]})
...   return df.corr()
>>> f()
          A         B        c
A  1.000000 -0.904656  0.92582
B -0.904656  1.000000 -0.97714
c  0.925820 -0.977140  1.00000
```

