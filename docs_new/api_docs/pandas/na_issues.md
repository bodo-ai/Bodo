# Integer NA issue in Pandas {#integer-na-issue-pandas}

DataFrame and Series objects with integer data need special care due to
[integer NA issues in
Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#nan-integer-na-values-and-na-type-promotions){target=blank}.
By default, Pandas dynamically converts integer columns to floating
point when missing values (NAs) are needed (which can result in loss of
precision). This is because Pandas uses the NaN floating point value as
NA, and Numpy does not support NaN values for integers. Bodo does not
perform this conversion unless enough information is available at
compilation time.

Pandas introduced a new [nullable integer data
type](https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html#integer-na){target=blank}
that can solve this issue, which is also supported by Bodo. For example,
this code reads column `A` into a nullable integer array
(the capital `"I"` denotes nullable integer type):

```py
@bodo.jit
def example(fname):
  dtype = {'A': 'Int64', 'B': 'float64'}
  df = pd.read_csv(fname,
      names=dtype.keys(),
      dtype=dtype,
  )
  ...
```
