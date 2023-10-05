import hashlib
import inspect
import warnings

import numpy as np
import pandas as pd
from pandas._libs import lib

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
    from pandas.core.arrays.string_arrow import ArrowStringArray

    super(ArrowStringArray, self).__init__(values)
    self._dtype = StringDtype(storage="pyarrow")

    # Bodo change: allow Arrow LargeStringArray (64-bit offsets) type created by Bodo
    # also allow dict-encoded string arrays from Bodo
    if not (
        pa.types.is_string(self._data.type)
        or pa.types.is_large_string(self._data.type)
        or (
            pa.types.is_dictionary(self._data.type)
            and (
                pa.types.is_string(self._data.type.value_type)
                or pa.types.is_large_string(self._data.type.value_type)
            )
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
        != "2b9106983d8d35cd024233abc1201cde4de1941584b902b0e2350bcbdfaa291c"
    ):  # pragma: no cover
        warnings.warn(
            "pd.core.arrays.string_arrow.ArrowStringArray.__init__ has changed"
        )

pd.core.arrays.string_arrow.ArrowStringArray.__init__ = ArrowStringArray__init__


# Bodo change to allow dict-encoded arrays
def factorize(
    self,
    na_sentinel: int | lib.NoDefault = lib.no_default,
    use_na_sentinel: bool | lib.NoDefault = lib.no_default,
):
    import pyarrow as pa
    from pandas.core.algorithms import resolve_na_sentinel

    resolved_na_sentinel = resolve_na_sentinel(na_sentinel, use_na_sentinel)
    # Bodo change:
    # Arrow's ChunkedArray.dictionary_encode() doesn't work for arrays that are
    # dict-encoded already so needs a check to be avoided.
    if pd.compat.pa_version_under4p0:
        encoded = (
            self._data
            if pa.types.is_dictionary(self._data.type)
            else self._data.dictionary_encode()
        )
    else:
        null_encoding = "mask" if resolved_na_sentinel is not None else "encode"
        encoded = (
            self._data
            if pa.types.is_dictionary(self._data.type)
            else self._data.dictionary_encode(null_encoding=null_encoding)
        )
    indices = pa.chunked_array(
        [c.indices for c in encoded.chunks], type=encoded.type.index_type
    ).to_pandas()
    if indices.dtype.kind == "f":
        indices[np.isnan(indices)] = (
            resolved_na_sentinel if resolved_na_sentinel is not None else -1
        )
    indices = indices.astype(np.int64, copy=False)

    if encoded.num_chunks:
        uniques = type(self)(encoded.chunk(0).dictionary)
        if resolved_na_sentinel is None and pd.compat.pa_version_under4p0:
            # TODO: share logic with BaseMaskedArray.factorize
            # Insert na with the proper code
            na_mask = indices.values == -1
            na_index = na_mask.argmax()
            if na_mask[na_index]:
                na_code = 0 if na_index == 0 else indices[:na_index].max() + 1
                uniques = uniques.insert(na_code, self.dtype.na_value)
                indices[indices >= na_code] += 1
                indices[indices == -1] = na_code
    else:
        uniques = type(self)(pa.array([], type=encoded.type.value_type))

    return indices.values, uniques


if _check_pandas_change:
    lines = inspect.getsource(pd.core.arrays.ArrowExtensionArray.factorize)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "e9408b1ac5c424efc2e6343d6a99be636eed7c99014348cda173a7c23abf276c"
    ):  # pragma: no cover
        warnings.warn(
            "pd.core.arrays.string_arrow.ArrowStringArray.factorize has changed"
        )


pd.core.arrays.ArrowExtensionArray.factorize = factorize


def to_numpy(
    self,
    dtype=None,
    copy: bool = False,
    na_value=lib.no_default,
) -> np.ndarray:
    """
    Convert to a NumPy ndarray.
    """
    # TODO: copy argument is ignored

    # Bodo change: work around bugs in Arrow for all null and empty array cases
    # see test_all_null_pa_bug
    data = self._data.combine_chunks() if len(self) != 0 else self._data

    result = np.array(data, dtype=dtype)
    if self._data.null_count > 0:
        if na_value is lib.no_default:
            if dtype and np.issubdtype(dtype, np.floating):
                return result
            na_value = self._dtype.na_value
        mask = self.isna()
        result[mask] = na_value
    return result


if _check_pandas_change:
    lines = inspect.getsource(pd.core.arrays.string_arrow.ArrowStringArray.to_numpy)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "2f49768c0cb51d06eb41882aaf214938f268497fffa07bf81964a5056d572ea3"
    ):  # pragma: no cover
        warnings.warn(
            "pd.core.arrays.string_arrow.ArrowStringArray.to_numpy has changed"
        )


pd.core.arrays.string_arrow.ArrowStringArray.to_numpy = to_numpy


# Add support for pow() in join conditions
pd.core.computation.ops.MATHOPS = pd.core.computation.ops.MATHOPS + ("pow",)


def FuncNode__init__(self, name: str) -> None:
    if name not in pd.core.computation.ops.MATHOPS:
        raise ValueError(f'"{name}" is not a supported function')
    self.name = name
    # Bodo change: handle pow() which is not in Numpy
    self.func = pow if name == "pow" else getattr(np, name)


if _check_pandas_change:  # pragma: no cover
    lines = inspect.getsource(pd.core.computation.ops.FuncNode.__init__)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "dec403a61ed8a58a2b29f3e2e8d49d6398adc7e27a226fe870d2e4b62d5c5475"
    ):
        warnings.warn("pd.core.computation.ops.FuncNode.__init__ has changed")


pd.core.computation.ops.FuncNode.__init__ = FuncNode__init__
