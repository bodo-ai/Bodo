# pd.DataFrame

`pandas.DataFrame(data=None, index=None, columns=None, dtype=None, copy=None)`


### Supported Arguments


- `data`: constant key dictionary, 2D Numpy array
    - `columns` argument is required when using a 2D Numpy array
- `index`: List, Tuple, Pandas index types, Pandas array types, Pandas series types, Numpy array types
- `columns`: Constant list of String, Constant tuple of String
    - **Must be constant at Compile Time**
- `dtype`: All values supported with `dataframe.astype` (see below)
- `copy`: boolean
    - **Must be constant at Compile Time**

