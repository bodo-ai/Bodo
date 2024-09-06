import hashlib
import inspect
import typing as pt
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
    self._dtype = StringDtype(storage=self._storage)

    # Bodo change: allow Arrow LargeStringArray (64-bit offsets) type created by Bodo
    # also allow dict-encoded string arrays from Bodo
    if not (
        pa.types.is_string(self._pa_array.type)
        or pa.types.is_large_string(self._pa_array.type)
        or (
            pa.types.is_dictionary(self._pa_array.type)
            and (
                pa.types.is_string(self._pa_array.type.value_type)
                or pa.types.is_large_string(self._pa_array.type.value_type)
            )
            and pa.types.is_int32(self._pa_array.type.index_type)
        )
    ):
        raise ValueError(
            "ArrowStringArray requires a PyArrow (chunked) array of string type"
        )


if _check_pandas_change:
    lines = inspect.getsource(pd.core.arrays.string_arrow.ArrowStringArray.__init__)
    if (
        hashlib.sha256(lines.encode()).hexdigest()
        != "5127b219e8856a16ef858b0f120881e32623d75422e50597f8a2fbb5281900c0"
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
    chunks = [array for ea in to_concat for array in ea._pa_array.iterchunks()]
    if to_concat[0].dtype == "string":
        # Bodo change: use Arrow type of underlying data since it could be different
        # (dict-encoded or large_string)
        pa_dtype = to_concat[0]._pa_array.type
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
        != "8f29eb56a84ce4000be3ba611f5a23cbf81b981fd8cfe5c7776e79f7800ba94e"
    ):  # pragma: no cover
        warnings.warn(
            "pd.core.arrays.arrow.array.ArrowExtensionArray._concat_same_type has changed"
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


# Pandas as of 2.1.4 doesn't have notna() for DatetimeArray for some reason
# See test_series_value_counts
if not hasattr(pd.arrays.DatetimeArray, "notna"):
    pd.arrays.DatetimeArray.notna = lambda self: ~self.isna()


# Implementation of precision_from_unit() which has been move to a cdef and
# is not accessible from Python. This is the python equivalent of the function.
# When possible we attempt to call into exposed Pandas APIs directly so we can
# benefit from native code.
def precision_from_unit_to_nanoseconds(in_reso: pt.Optional[str]):
    if in_reso is None:
        in_reso = "ns"
    if in_reso == "Y":
        # each 400 years we have 97 leap years, for an average of 97/400=.2425
        #  extra days each year. We get 31556952 by writing
        #  3600*24*365.2425=31556952
        multiplier = pd._libs.tslibs.dtypes.periods_per_second(
            pd._libs.dtypes.abbrev_to_npy_unit("ns")
        )
        m = multiplier * 31556952
    elif in_reso == "M":
        # 2629746 comes from dividing the "Y" case by 12.
        multiplier = pd._libs.tslibs.dtypes.periods_per_second(
            pd._libs.dtypes.abbrev_to_npy_unit("ns")
        )
        m = multiplier * 2629746
    else:
        # Careful: if get_conversion_factor raises, the exception does
        #  not propagate, instead we get a warning about an ignored exception.
        #  https://github.com/pandas-dev/pandas/pull/51483#discussion_r1115198951
        m = get_conversion_factor_to_ns(in_reso)

    p = np.floor(np.log10(m))  # number of digits in 'm' minus 1
    return m, p


def get_conversion_factor_to_ns(in_reso: str) -> int:
    """
    Get the conversion factor between two resolutions.

    Parameters
    ----------
    in_reso : str
        The input resolution.
    out_reso : str
        The output resolution.

    Returns
    -------
    int
        The conversion factor.
    """
    if in_reso == "ns":
        return 1

    if in_reso == "W":
        value = get_conversion_factor_to_ns("D")
        factor = 7
    elif in_reso == "D" or in_reso == "d":
        value = get_conversion_factor_to_ns("h")
        factor = 24
    elif in_reso == "h":
        value = get_conversion_factor_to_ns("m")
        factor = 60
    elif in_reso == "m":
        value = get_conversion_factor_to_ns("s")
        factor = 60
    elif in_reso == "s":
        value = get_conversion_factor_to_ns("ms")
        factor = 1000
    elif in_reso == "ms":
        value = get_conversion_factor_to_ns("us")
        factor = 1000
    elif in_reso == "us":
        value = get_conversion_factor_to_ns("ns")
        factor = 1000
    else:
        raise ValueError(f"Unsupported resolution {in_reso}")
    return factor * value
