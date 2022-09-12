import hashlib
import inspect
import warnings

import pandas as pd

pandas_version = tuple(map(int, pd.__version__.split(".")[:2]))

# flag for checking whether the functions we are replacing have changed in a later Pandas
# release. Needs to be checked for every new Pandas release so we update our changes.
_check_pandas_change = False

if pandas_version < (1, 4):
    # c_parser_wrapper change
    # Bodo Change: Upgrade to Pandas 1.4 implementation which replaces
    # col_indices with a dictionary
    def _set_noconvert_columns(self):
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions.
        """
        assert self.orig_names is not None
        # error: Cannot determine type of 'names'

        # Bodo Change vs 1.3.4 Replace orig_names.index(x) with
        # dictionary. This is already merged into Pandas 1.4
        # much faster than using orig_names.index(x) xref GH#44106
        names_dict = {x: i for i, x in enumerate(self.orig_names)}
        col_indices = [names_dict[x] for x in self.names]  # type: ignore[has-type]
        # error: Cannot determine type of 'names'
        noconvert_columns = self._set_noconvert_dtype_columns(
            col_indices,
            self.names,  # type: ignore[has-type]
        )
        for col in noconvert_columns:
            self._reader.set_noconvert(col)

    if _check_pandas_change:
        # make sure run_frontend hasn't changed before replacing it
        lines = inspect.getsource(
            pd.io.parsers.c_parser_wrapper.CParserWrapper._set_noconvert_columns
        )
        if (
            hashlib.sha256(lines.encode()).hexdigest()
            != "afc2d738f194e3976cf05d61cb16dc4224b0139451f08a1cf49c578af6f975d3"
        ):  # pragma: no cover
            warnings.warn(
                "pd.io.parsers.c_parser_wrapper.CParserWrapper._set_noconvert_columns has changed"
            )

    pd.io.parsers.c_parser_wrapper.CParserWrapper._set_noconvert_columns = (
        _set_noconvert_columns
    )


# Bodo change: allow Arrow LargeStringArray (64-bit offsets) type created by Bodo
# also allow dict-encoded string arrays from Bodo
# Pandas code: https://github.com/pandas-dev/pandas/blob/ca60aab7340d9989d9428e11a51467658190bb6b/pandas/core/arrays/string_arrow.py#L141
def ArrowStringArray__init__(self, values):
    import pyarrow as pa
    from pandas.core.arrays.string_ import StringDtype

    self._dtype = StringDtype(storage="pyarrow")
    if isinstance(values, pa.Array):
        self._data = pa.chunked_array([values])
    elif isinstance(values, pa.ChunkedArray):
        self._data = values
    else:
        raise ValueError(f"Unsupported type '{type(values)}' for ArrowStringArray")

    # Bodo change: allow Arrow LargeStringArray (64-bit offsets) type created by Bodo
    # also allow dict-encoded string arrays from Bodo
    if not (
        pa.types.is_string(self._data.type)
        or pa.types.is_large_string(self._data.type)
        or (
            pa.types.is_dictionary(self._data.type)
            and pa.types.is_large_string(self._data.type.value_type)
            and pa.types.is_int32(self._data.type.index_type)
        )
    ):
        raise ValueError(
            "ArrowStringArray requires a PyArrow (chunked) array of string type"
        )


if _check_pandas_change:
    lines = inspect.getsource(pd.core.arrays.string_arrow.ArrowStringArray.__init__)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "cbb9683b2e91867ef12470d4fda28ca6243fbb7b4f78ac2472fca805607c0942"
    ):  # pragma: no cover
        warnings.warn(
            "pd.core.arrays.string_arrow.ArrowStringArray.__init__ has changed"
        )

pd.core.arrays.string_arrow.ArrowStringArray.__init__ = ArrowStringArray__init__


# Bodo change to allow dict-encoded arrays
def factorize(self, na_sentinel: int = -1):
    import numpy as np
    import pyarrow as pa

    # Bodo change:
    # Arrow's ChunkedArray.dictionary_encode() doesn't work for arrays that are
    # dict-encoded already so needs a check to be avoided.
    encoded = (
        self._data
        if pa.types.is_dictionary(self._data.type)
        else self._data.dictionary_encode()
    )
    indices = pa.chunked_array(
        [c.indices for c in encoded.chunks], type=encoded.type.index_type
    ).to_pandas()
    if indices.dtype.kind == "f":
        indices[np.isnan(indices)] = na_sentinel
    indices = indices.astype(np.int64, copy=False)

    if encoded.num_chunks:
        uniques = type(self)(encoded.chunk(0).dictionary)
    else:
        uniques = type(self)(pa.array([], type=encoded.type.value_type))

    return indices.values, uniques


if _check_pandas_change:
    lines = inspect.getsource(pd.core.arrays.string_arrow.ArrowStringArray.factorize)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "80669a609cfe11b362dacec6bba8e5bf41418b35d0d8b58246a548858320efd9"
    ):  # pragma: no cover
        warnings.warn(
            "pd.core.arrays.string_arrow.ArrowStringArray.factorize has changed"
        )


pd.core.arrays.string_arrow.ArrowStringArray.factorize = factorize
