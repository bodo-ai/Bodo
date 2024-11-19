"""Support scikit-learn utils helpers."""

import numba
import sklearn.utils
from numba.core import types
from numba.extending import overload

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_index_ext import NumericIndexType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.utils.typing import (
    is_overload_false,
)

# -----------------------------------------------------------------------------
# ----------------------------------shuffle------------------------------------


@overload(sklearn.utils.shuffle, no_unliteral=True)
def sklearn_utils_shuffle_overload(
    data,
    random_state=None,
    n_samples=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Implement shuffle. If data is replicated, we simply call sklearn,
    else we use our native implementation.
    This simple implementation only supports one array for now.
    """
    if is_overload_false(_is_data_distributed):
        # If data is not distributed, then just call sklearn
        # Note: _is_data_distributed is set in the Distributed compiler pass
        #
        # Here, data is the underlying numba type of `data`. We need to set the
        # kwargs of objmode to be compile-time constants that represent the
        # output type of each PyObject defined under the bodo.objmode context.
        #
        # Following https://github.com/numba/numba/blob/main/numba/core/withcontexts.py#L182
        # and https://github.com/numba/numba/blob/main/numba/core/sigutils.py#L12,
        # bodo.objmode() will eval() the given type annotation string, with
        # the entries of numba.core.types as global variables, to determine the
        # type signature of each output.

        # Therefore, we need to define a unique entry for `data`'s type within
        # numba.core.types:
        data_type_name = f"utils_shuffle_type_{numba.core.ir_utils.next_label()}"
        if isinstance(data, (DataFrameType, SeriesType)):
            # Following train_test_split, make sure we use NumericIndexType
            # over other unsupported index types for pandas inputs
            data_typ = data.copy(index=NumericIndexType(types.int64))
            setattr(types, data_type_name, data_typ)
        else:
            setattr(types, data_type_name, data)
        func_text = "def _utils_shuffle_impl(\n"
        func_text += (
            "    data, random_state=None, n_samples=None, _is_data_distributed=False\n"
        )
        func_text += "):\n"
        func_text += f"    with bodo.objmode(out='{data_type_name}'):\n"
        func_text += "        out = sklearn.utils.shuffle(\n"
        func_text += (
            "            data, random_state=random_state, n_samples=n_samples\n"
        )
        func_text += "        )\n"
        func_text += "    return out\n"
        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        _utils_shuffle_impl = loc_vars["_utils_shuffle_impl"]

    else:
        # If distributed, directly call bodo random_shuffle
        def _utils_shuffle_impl(
            data, random_state=None, n_samples=None, _is_data_distributed=False
        ):  # pragma: no cover
            m = bodo.random_shuffle(
                data, seed=random_state, n_samples=n_samples, parallel=True
            )
            return m

    return _utils_shuffle_impl
