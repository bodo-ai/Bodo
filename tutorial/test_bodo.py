# coding: utf-8
import numpy as np
import pandas as pd
import bodo

@bodo.jit
def f(n):
    return pd.DataFrame({'A': np.random.ranf(n)}).head(3)

print(f)
print(f(10))
