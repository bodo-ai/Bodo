from __future__ import annotations

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

if pandas_version < (3, 0):
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

else:
    # Bodo change: allow dictionary-encoded string arrays
    def ArrowStringArray__init__(self, values, *, dtype=None) -> None:
        import pyarrow.compute as pc
        from pandas._libs import (
            missing as libmissing,
        )
        from pandas.core.arrays.string_ import (
            StringDtype,
        )
        from pandas.core.arrays.string_arrow import (
            _check_pyarrow_available,
            _is_string_view,
        )

        _check_pyarrow_available()
        if isinstance(values, (pa.Array, pa.ChunkedArray)) and (
            pa.types.is_string(values.type) or _is_string_view(values.type)
            # Bodo change: allow dictionary-encoded string arrays
            # or (
            #     pa.types.is_dictionary(values.type)
            #     and (
            #         pa.types.is_string(values.type.value_type)
            #         or pa.types.is_large_string(values.type.value_type)
            #         or _is_string_view(values.type.value_type)
            #     )
            # )
        ):
            values = pc.cast(values, pa.large_string())

        super(pd.core.arrays.string_arrow.ArrowStringArray, self).__init__(values)

        if dtype is None:
            dtype = StringDtype(storage="pyarrow", na_value=libmissing.NA)
        self._dtype = dtype

        # Bodo change: allow dictionary-encoded string arrays
        if not (
            pa.types.is_large_string(self._pa_array.type)
            or (
                pa.types.is_dictionary(self._pa_array.type)
                and (
                    pa.types.is_string(self._pa_array.type.value_type)
                    or pa.types.is_large_string(self._pa_array.type.value_type)
                    or _is_string_view(self._pa_array.type.value_type)
                )
            )
        ):
            raise ValueError(
                "ArrowStringArray requires a PyArrow (chunked) array of "
                "large_string type"
            )

    if _check_pandas_change:
        lines = inspect.getsource(pd.core.arrays.string_arrow.ArrowStringArray.__init__)
        if (
            hashlib.sha256(lines.encode()).hexdigest()
            != "5ca54168210292bd5d20a27b9fd40f28d54f7afb63a8468db2630ca33e84dc2c"
        ):  # pragma: no cover
            warnings.warn(
                "pd.core.arrays.string_arrow.ArrowStringArray.__init__ has changed"
            )

    pd.core.arrays.string_arrow.ArrowStringArray.__init__ = ArrowStringArray__init__

    def _maybe_convert_setitem_value(self, value):
        """Maybe convert value to be pyarrow compatible."""
        from pandas.core.dtypes.common import is_scalar
        from pandas.core.dtypes.missing import isna

        if is_scalar(value):
            if isna(value):
                value = None
            elif not isinstance(value, str):
                raise TypeError(
                    f"Invalid value '{value}' for dtype 'str'. Value should be a "
                    f"string or missing value, got '{type(value).__name__}' instead."
                )
        # Bodo change: avoid converting to object array to handle dict-encoded string
        # arrays from Bodo properly
        elif isinstance(value, pd.core.arrays.string_arrow.ArrowStringArray):
            return value._pa_array
        else:
            value = np.array(value, dtype=object, copy=True)
            value[isna(value)] = None
            for v in value:
                if not (v is None or isinstance(v, str)):
                    raise TypeError(
                        "Invalid value for dtype 'str'. Value should be a "
                        "string or missing value (or array of those)."
                    )
        return super(
            pd.core.arrays.string_arrow.ArrowStringArray, self
        )._maybe_convert_setitem_value(value)

    if _check_pandas_change:
        lines = inspect.getsource(
            pd.core.arrays.string_arrow.ArrowStringArray._maybe_convert_setitem_value
        )
        if (
            hashlib.sha256(lines.encode()).hexdigest()
            != "f682cc4462f98a59bc4081c972050f33eecea909444909945ba707f8b5f3dc05"
        ):  # pragma: no cover
            warnings.warn(
                "pd.core.arrays.string_arrow.ArrowStringArray._maybe_convert_setitem_value has changed"
            )

    pd.core.arrays.string_arrow.ArrowStringArray._maybe_convert_setitem_value = (
        _maybe_convert_setitem_value
    )


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
    if pandas_version < (3, 0):
        return cls(arr)
    else:
        return to_concat[0]._from_pyarrow_array(arr)


if _check_pandas_change:
    lines = inspect.getsource(
        pd.core.arrays.arrow.array.ArrowExtensionArray._concat_same_type
    )
    if hashlib.sha256(lines.encode()).hexdigest() not in (
        "8f29eb56a84ce4000be3ba611f5a23cbf81b981fd8cfe5c7776e79f7800ba94e",
        "b06e7a78317c289db40080d30c60ae03fa93afd073c85fb033ec43e9ad1dd9f0",
    ):  # pragma: no cover
        warnings.warn(
            "pd.core.arrays.arrow.array.ArrowExtensionArray._concat_same_type has changed"
        )


pd.core.arrays.arrow.array.ArrowExtensionArray._concat_same_type = _concat_same_type


pd_str_find = pd.core.arrays.arrow.array.ArrowExtensionArray._str_find


def _str_find(self, sub: str, start: int = 0, end: int | None = None):
    # Bodo change: add fallback to regular Series.str.find() if args not supported by
    # ArrowExtensionArray. See: test_df_lib/test_series_str.py::test_auto_find
    if (start != 0 and end is not None) or (start == 0 and end is None):
        return pd_str_find(self, sub, start, end)
    else:
        return pd.Series(self.to_numpy()).str.find(sub, start, end).array


if _check_pandas_change:
    lines = inspect.getsource(pd.core.arrays.arrow.array.ArrowExtensionArray._str_find)
    if hashlib.sha256(lines.encode()).hexdigest() not in (
        "179388243335db6b590d875b3ac1c249efffac4194b8bc56c9c54d956ab5f370",
        "951a1ecf005bf7e00146c61e878ff1615950c20a95b323007ee9d69271524d78",
    ):  # pragma: no cover
        warnings.warn(
            "pd.core.arrays.arrow.array.ArrowExtensionArray._str_find has changed"
        )

pd.core.arrays.arrow.array.ArrowExtensionArray._str_find = _str_find

if pandas_version < (3, 0):

    def _explode(self):
        """
        See Series.explode.__doc__.
        """
        # child class explode method supports only list types; return
        # default implementation for non list types.
        if not hasattr(self.dtype, "pyarrow_dtype") or (
            # Bodo change: check is_large_list as well
            not (
                pa.types.is_list(self.dtype.pyarrow_dtype)
                or pa.types.is_large_list(self.dtype.pyarrow_dtype)
            )
        ):
            return super()._explode()
        values = self
        counts = pa.compute.list_value_length(values._pa_array)
        counts = counts.fill_null(1).to_numpy()
        fill_value = pa.scalar([None], type=self._pa_array.type)
        mask = counts == 0
        if mask.any():
            values = values.copy()
            values[mask] = fill_value
            counts = counts.copy()
            counts[mask] = 1
        values = values.fillna(fill_value)
        values = type(self)(pa.compute.list_flatten(values._pa_array))
        return values, counts

    if _check_pandas_change:
        lines = inspect.getsource(
            pd.core.arrays.arrow.array.ArrowExtensionArray._explode
        )
        if (
            hashlib.sha256(lines.encode()).hexdigest()
            != "6c1b05ccc4da39ec3b7d7dfd79a9d9e47968db3b2eb4c615d21d490b21f9b421"
        ):  # pragma: no cover
            warnings.warn(
                "pd.core.arrays.arrow.array.ArrowExtensionArray._explode has changed"
            )

    pd.core.arrays.arrow.array.ArrowExtensionArray._explode = _explode


if pandas_version < (3, 0):
    # Fixes iloc Indexing for ArrowExtensionArray (see test_slice_with_series in Narwhals tests)
    # Pandas 3.0+ has a fix already: https://github.com/pandas-dev/pandas/pull/61924
    pd.core.arrays.arrow.array.ArrowExtensionArray.max = lambda self: self._reduce(
        "max"
    )
    pd.core.arrays.arrow.array.ArrowExtensionArray.min = lambda self: self._reduce(
        "min"
    )


# Bodo change: add missing str_map() for ArrowExtensionArray that is used in operations
# like zfill.
def arrow_arr_str_map(self, f, na_value=None, dtype=None, convert=True):
    return pd.Series(self.to_numpy()).array._str_map(f, na_value, dtype, convert)


pd.core.arrays.arrow.array.ArrowExtensionArray._str_map = arrow_arr_str_map


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
def precision_from_unit_to_nanoseconds(in_reso: str | None):
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


# Class responsible for executing UDFs using Bodo as the engine in
# newer version of Pandas. See:
# https://github.com/pandas-dev/pandas/pull/61032
bodo_pandas_udf_execution_engine = None


def _prepare_function_arguments(
    func: Callable, args: tuple, kwargs: dict, *, num_required_args: int = 1
) -> tuple[tuple, dict]:
    """
    Prepare arguments for jitted function by trying to move keyword arguments inside
    of args to eliminate kwargs.

    This simplifies typing as well as catches keyword-only arguments,
    which lead to unexpected behavior in Bodo. Copied from:
    https://github.com/pandas-dev/pandas/blob/5fef9793dd23867e7b227a1df7aa60a283f6204e/pandas/core/util/numba_.py#L97
    """
    _sentinel = object()

    if not kwargs:
        return args, kwargs

    # the udf should have this pattern: def udf(arg1, arg2, ..., *args, **kwargs):...
    signature = inspect.signature(func)
    arguments = signature.bind(*[_sentinel] * num_required_args, *args, **kwargs)
    arguments.apply_defaults()
    # Ref: https://peps.python.org/pep-0362/
    # Arguments which could be passed as part of either *args or **kwargs
    # will be included only in the BoundArguments.args attribute.
    args = arguments.args
    kwargs = arguments.kwargs

    if kwargs:
        # Bodo change: error message
        raise ValueError("Bodo does not support keyword only arguments.")

    args = args[num_required_args:]
    return args, kwargs


if pandas_version >= (3, 0):
    from collections.abc import Callable
    from typing import Any

    from pandas._typing import AggFuncType, Axis
    from pandas.core.apply import BaseExecutionEngine

    class BodoExecutionEngine(BaseExecutionEngine):
        @staticmethod
        def map(
            data: pd.Series | pd.DataFrame | np.ndarray,
            func: AggFuncType,
            args: tuple,
            kwargs: dict[str, Any],
            decorator: Callable | None,
            skip_na: bool,
        ):
            if not isinstance(data, pd.Series):
                raise ValueError(
                    f"BodoExecutionEngine: map() expected input data to be Series, got: {type(data)}"
                )

            if isinstance(func, Callable):
                args, _ = _prepare_function_arguments(
                    func, args, kwargs, num_required_args=1
                )

            na_action = "ignore" if skip_na else None

            def map_func(data, args):
                return data.map(func, na_action=na_action, args=args)

            map_func_jit = decorator(map_func)

            return map_func_jit(data, args)

        @staticmethod
        def apply(
            data: pd.Series | pd.DataFrame | np.ndarray,
            func: AggFuncType,
            args: tuple,
            kwargs: dict[str, Any],
            decorator: Callable,
            axis: Axis,
        ):
            # raw = True converts data to ndarray first
            if isinstance(data, np.ndarray):
                raise ValueError(
                    "BodoExecutionEngine: does not support the raw=True for DataFrame.apply."
                )

            if isinstance(func, Callable):
                args, _ = _prepare_function_arguments(
                    func, args, kwargs, num_required_args=1
                )

            def apply_func(data, axis, args):
                return data.apply(func, axis=axis, args=args)

            apply_func_jit = decorator(apply_func)

            return apply_func_jit(data, axis, args)

    bodo_pandas_udf_execution_engine = BodoExecutionEngine
