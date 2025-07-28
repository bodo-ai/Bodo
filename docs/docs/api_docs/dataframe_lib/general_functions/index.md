# General Functions

## Conversion from Pandas
- [`bodo.pandas.from_pandas`][bodofrompandas]

---

## Top-level dealing with datetimelike data
!!! note
    `to_datetime` currently supports only BodoSeries and BodoDataFrame inputs. Passing arguments of other types will trigger a fallback to Pandas.

- [`bodo.pandas.to_datetime`][bodotodatetime]

---

## Top-level missing data
!!! note
    `isna`, `isnull`, `notna`, and `notnull` currently support only BodoSeries and scalar inputs (e.g., integers, strings). Passing other types will trigger a fallback to Pandas.

- [`bodo.pandas.isna`][bodoisna]
- [`bodo.pandas.isnull`][bodoisnull]
- [`bodo.pandas.notna`][bodonotna]
- [`bodo.pandas.notnull`][bodonotnull]
 
[bodotodatetime]: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
[bodoisna]: https://pandas.pydata.org/docs/reference/api/pandas.isna.html
[bodoisnull]: https://pandas.pydata.org/docs/reference/api/pandas.isnull.html
[bodonotna]: https://pandas.pydata.org/docs/reference/api/pandas.notna.html
[bodonotnull]: https://pandas.pydata.org/docs/reference/api/pandas.notnull.html

[bodofrompandas]: ../general_functions/from_pandas.md