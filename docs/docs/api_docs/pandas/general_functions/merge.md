
# `pd.merge`


`pandas.merge(left, right, how="inner", on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=("_x", "_y"), copy=True, indicator=False, validate=None, _bodo_na_equal=True)`


### Supported Arguments

| argument         | datatypes                                                                                  | other requirements                                                                                                                                                                                                     |
|------------------|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `left`           | DataFrame                                                                                  |                                                                                                                                                                                                                        |
| `right`          | DataFrame                                                                                  |                                                                                                                                                                                                                        |
| `how`            | String                                                                                     | <ul> <li> Must be one of `"inner"`, `"outer"`,  `"left"`, `"right"`</li> <li> **Must be constant at  Compile Time** </li> </ul>                                                                                        |
| `on`             | Column Name, List of Column Names, or General Merge Condition String (see [merge-notes][]) | <ul> <li> **Must be constant at Compile Time** </li> </ul>                                                                                                                                                             |
| `left_on`        | Column Name or List of Column Names                                                        | <ul><li>   **Must be constant at  Compile Time** </li></ul>                                                                                                                                                            |
| `right_on`       | Column Name or List of Column  Names                                                       | <ul><li>   **Must be constant at  Compile Time** </li></ul>                                                                                                                                                            |
| `left_index`     | Boolean                                                                                    | <ul><li>   **Must be constant at  Compile Time** </li></ul>                                                                                                                                                            |
| `right_index`    | Boolean                                                                                    | <ul><li>   **Must be constant at  Compile Time** </li></ul>                                                                                                                                                            |
| `suffixes`       | Tuple of Strings                                                                           | <ul><li>   **Must be constant at  Compile Time** </li></ul>                                                                                                                                                            |
| `indicator`      | Boolean                                                                                    | <ul><li>   **Must be constant at  Compile Time** </li></ul>                                                                                                                                                            |
| `_bodo_na_equal` | Boolean                                                                                    | <ul><li>   **Must be constant at  Compile Time** </li> <li> This argument is  unique to Bodo and not  available in Pandas. If False, Bodo won't  consider NA/nan keys  as equal, which differs from Pandas. </li></ul> |


!!! info "Important"
    The argument `_bodo_na_equal` is unique to Bodo and not available in Pandas. If it is `False`, Bodo won't consider NA/nan keys as equal, which differs from Pandas.


### Merge Notes


-   *Output Ordering*:

    The output dataframe is not sorted by default for better parallel performance
    (Pandas may preserve key order depending on `how`).
    One can use explicit sort if needed.

-   *General Merge Conditions*:

    Within Pandas, the merge criteria supported by `pd.merge` are limited to equality between 1
    or more pairs of keys. For some use cases, this is not sufficient and more generalized
    support is necessary. For example, with these limitations, a `left outer join` where
    `df1.A == df2.B & df2.C < df1.A` cannot be efficiently computed.

    Bodo supports these use cases by allowing users to pass general merge conditions to `pd.merge`.
    We plan to contribute this feature to Pandas to ensure full compatibility of Bodo and Pandas code.

    General merge conditions are performed by providing the condition as a string via the `on` argument. Columns in the left table
    are referred to by `left.{column name}` and columns in the right table are referred to by `right.{column name}`.

    Here's an example demonstrating the above:

    ```py

    >>> @bodo.jit
    ... def general_merge(df1, df2):
    ...   return df1.merge(df2, on="left.`A` == right.`B` & right.`C` < left.`A`", how="left")

    >>> df1 = pd.DataFrame({"col": [2, 3, 5, 1, 2, 8], "A": [4, 6, 3, 9, 9, -1]})
    >>> df2 = pd.DataFrame({"B": [1, 2, 9, 3, 2], "C": [1, 7, 2, 6, 5]})
    >>> general_merge(df1, df2)

       col  A     B     C
    0    2  4  <NA>  <NA>
    1    3  6  <NA>  <NA>
    2    5  3  <NA>  <NA>
    3    1  9     9     2
    4    2  9     9     2
    5    8 -1  <NA>  <NA>
    ```

    These calls have a few additional requirements:

    * The condition must be constant string.
    * The condition must be of the form `cond_1 & ... & cond_N` where at least one `cond_i`
      is a simple equality. This restriction will be removed in a future release.
    * The columns specified in these conditions are limited to certain column types.
      We currently support `boolean`, `integer`, `float`, `datetime64`, `timedelta64`, `datetime.date`,
      and `string` columns.

### Example Usage

```py

>>> @bodo.jit
... def f(df1, df2):
...   return pd.merge(df1, df2, how="inner", on="key")

>>> df1 = pd.DataFrame({"key": [2, 3, 5, 1, 2, 8], "A": np.array([4, 6, 3, 9, 9, -1], float)})
>>> df2 = pd.DataFrame({"key": [1, 2, 9, 3, 2], "B": np.array([1, 7, 2, 6, 5], float)})
>>> f(df1, df2)

key    A    B
0    2  4.0  7.0
1    2  4.0  5.0
2    3  6.0  6.0
3    1  9.0  1.0
4    2  9.0  7.0
5    2  9.0  5.0
```
