import hashlib
import inspect
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa

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


@classmethod
def _concat_same_type(cls, to_concat):
    """
    Concatenate multiple ArrowExtensionArrays.

    Parameters
    ----------
    to_concat : sequence of ArrowExtensionArrays

    Returns
    -------
    ArrowExtensionArray
    """
    chunks = [array for ea in to_concat for array in ea._data.iterchunks()]
    if to_concat[0].dtype == "string":
        # Bodo change: use Arrow type of underlying data since it could be different
        # (dict-encoded or large_string)
        pa_dtype = to_concat[0]._data.type
    else:
        pa_dtype = to_concat[0].dtype.pyarrow_dtype
    arr = pa.chunked_array(chunks, type=pa_dtype)
    return cls(arr)


if _check_pandas_change:
    lines = inspect.getsource(
        pd.core.arrays.arrow.array.ArrowExtensionArray._concat_same_type
    )
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "6a7397b59a7264de2167cc79344f01c411aeeecccd47d9ee4e37da4b700316ff"
    ):  # pragma: no cover
        warnings.warn(
            "pd.core.arrays.arrow.array.ArrowExtensionArray._concat_same_typehas changed"
        )


pd.core.arrays.arrow.array.ArrowExtensionArray._concat_same_type = _concat_same_type


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
