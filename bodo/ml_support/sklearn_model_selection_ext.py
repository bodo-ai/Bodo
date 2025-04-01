"""Support scikit-learn model selection methods."""

import sys
import types as pytypes
from itertools import combinations

import numba
import numpy as np
import pandas as pd
import sklearn.model_selection
from numba.core import types
from numba.extending import (
    overload,
    overload_method,
)
from scipy.special import comb

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_index_ext import NumericIndexType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.libs.distributed_api import Reduce_Type
from bodo.ml_support.sklearn_ext import check_sklearn_version
from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.typing import (
    check_unsupported_args,
    get_overload_const_int,
    is_overload_false,
    is_overload_none,
    is_overload_true,
)

this_module = sys.modules[__name__]

# ----------------------------------------------------------------------------------------
# ----------------------------------------LeavePOut---------------------------------------
# Support for sklearn.model_selection.LeavePOut.

# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoModelSelectionLeavePOutType, _ = install_py_obj_class(
    types_name="model_selection_leave_p_out_type",
    python_type=sklearn.model_selection.LeavePOut,
    module=this_module,
    class_name="BodoModelSelectionLeavePOutType",
    model_name="BodoModelSelectionLeavePOutModel",
)


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoModelSelectionLeavePOutGeneratorType, _ = install_py_obj_class(
    types_name="model_selection_leave_p_out_generator_type",
    module=this_module,
    class_name="BodoModelSelectionLeavePOutGeneratorType",
    model_name="BodoModelSelectionLeavePOutGeneratorModel",
)


@overload(sklearn.model_selection.LeavePOut, no_unliteral=True)
def sklearn_model_selection_leave_p_out_overload(
    p,
):
    """
    Provide implementation for __init__ function of LeavePOut.
    We simply call sklearn in objmode.
    """

    check_sklearn_version()

    def _sklearn_model_selection_leave_p_out_impl(
        p,
    ):  # pragma: no cover
        with bodo.objmode(m="model_selection_leave_p_out_type"):
            m = sklearn.model_selection.LeavePOut(
                p=p,
            )
        return m

    return _sklearn_model_selection_leave_p_out_impl


def sklearn_model_selection_leave_p_out_generator_dist_helper(m, X):
    """
    Distributed calculation of train/test split indices for LeavePOut.
    We use sklearn on all the indices, then filter out the indices assigned
    to each individual rank.
    """
    # Compute index offset of each rank
    my_rank = bodo.get_rank()
    nranks = bodo.get_size()
    rank_data_len = np.empty(nranks, np.int64)
    bodo.libs.distributed_api.allgather(rank_data_len, len(X))
    if my_rank > 0:
        rank_start = np.sum(rank_data_len[:my_rank])
    else:
        rank_start = 0
    rank_end = rank_start + rank_data_len[my_rank]

    # Compute total data size
    global_data_size = np.sum(rank_data_len)

    # Raise error if dataset is too small
    if global_data_size <= m.p:
        raise ValueError(
            f"p={m.p} must be strictly less than the number of samples={global_data_size}"
        )

    # For all possible test set combinations, compute train and test indices
    # that belong to the current rank.
    # Since `combinations` returns deterministic and fixed-ordered output,
    # in lexicographic ordering according to the order of the input iterable,
    # this is safe to do in parallel on all ranks at once
    local_indices = np.arange(rank_start, rank_end)
    for combination in combinations(range(global_data_size), m.p):
        test_index = np.array(combination)
        test_index = test_index[test_index >= rank_start]
        test_index = test_index[test_index < rank_end]

        test_mask = np.zeros(len(X), dtype=bool)
        test_mask[test_index - rank_start] = True

        train_index = local_indices[np.logical_not(test_mask)]
        yield train_index, test_index


@overload_method(BodoModelSelectionLeavePOutType, "split", no_unliteral=True)
def overload_model_selection_leave_p_out_generator(
    m,
    X,
    y=None,
    groups=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementations for the split function, which is a generator.
    In case input is replicated, we simply call sklearn,
    else we use our native implementation.
    """

    if is_overload_true(_is_data_distributed):
        # If distributed, then use native implementation
        def _model_selection_leave_p_out_generator_impl(
            m, X, y=None, groups=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(gen="model_selection_leave_p_out_generator_type"):
                gen = sklearn_model_selection_leave_p_out_generator_dist_helper(m, X)
            return gen

    else:
        # If replicated, then just call sklearn
        def _model_selection_leave_p_out_generator_impl(
            m, X, y=None, groups=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(gen="model_selection_leave_p_out_generator_type"):
                gen = m.split(X, y=y, groups=groups)
            return gen

    return _model_selection_leave_p_out_generator_impl


@overload_method(BodoModelSelectionLeavePOutType, "get_n_splits", no_unliteral=True)
def overload_model_selection_leave_p_out_get_n_splits(
    m,
    X,
    y=None,
    groups=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementations for the get_n_splits function.
    """

    if is_overload_true(_is_data_distributed):
        # If distributed, then use native implementation
        def _model_selection_leave_p_out_get_n_splits_impl(
            m, X, y=None, groups=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(out="int64"):
                global_data_size = bodo.libs.distributed_api.dist_reduce(
                    len(X), np.int32(Reduce_Type.Sum.value)
                )
                out = int(comb(global_data_size, m.p, exact=True))
            return out

    else:
        # If replicated, then just call sklearn
        def _model_selection_leave_p_out_get_n_splits_impl(
            m, X, y=None, groups=None, _is_data_distributed=False
        ):  # pragma: no cover
            with bodo.objmode(out="int64"):
                out = m.get_n_splits(X)

            return out

    return _model_selection_leave_p_out_get_n_splits_impl


# ----------------------------------------------------------------------------------------
# ---------------------------------------- KFold -----------------------------------------
# Support for sklearn.model_selection.KFold.
# Both get_n_splits and split functions are supported.
# For split, if data is distributed and shuffle=False, use sklearn individually
# on each rank then add a rank offset. If data is distributed and shuffle=True,
# use sklearn on each rank individually, add a rank offset, then permute the output.
#
# Our implementation differs from sklearn's to ensure both train and test data are
# evenly distributed across ranks if possible. For example, if X=range(8), nprocs=2,
# and n_splits=4, then our first fold is test = [0,4] and train = [1,2,3,5,6,7],
# while sklearn's first fold is test = [0,1] and train = [2,3,4,5,6,7].
# ----------------------------------------------------------------------------------------


# We don't technically need to get class from the method,
# but it's useful to avoid IDE not found errors.
BodoModelSelectionKFoldType, _ = install_py_obj_class(
    types_name="model_selection_kfold_type",
    python_type=sklearn.model_selection.KFold,
    module=this_module,
    class_name="BodoModelSelectionKFoldType",
    model_name="BodoModelSelectionKFoldModel",
)


@overload(sklearn.model_selection.KFold, no_unliteral=True)
def sklearn_model_selection_kfold_overload(
    n_splits=5,
    shuffle=False,
    random_state=None,
):
    """
    Provide implementation for __init__ function of KFold.
    We simply call sklearn in objmode.
    """

    check_sklearn_version()

    def _sklearn_model_selection_kfold_impl(
        n_splits=5,
        shuffle=False,
        random_state=None,
    ):  # pragma: no cover
        with bodo.objmode(m="model_selection_kfold_type"):
            m = sklearn.model_selection.KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state,
            )
        return m

    return _sklearn_model_selection_kfold_impl


def sklearn_model_selection_kfold_generator_dist_helper(m, X, y=None, groups=None):
    """
    Distributed calculation of train/test split indices for KFold.
    We use sklearn on the indices assigned to each individual rank,
    then add the rank offset afterwards.
    """
    # Compute index offset of each rank
    my_rank = bodo.get_rank()
    nranks = bodo.get_size()
    rank_data_len = np.empty(nranks, np.int64)
    bodo.libs.distributed_api.allgather(rank_data_len, len(X))
    if my_rank > 0:
        rank_start = np.sum(rank_data_len[:my_rank])
    else:
        rank_start = 0
    rank_end = rank_start + len(X)

    # Compute total data size and global/local indices
    global_data_size = np.sum(rank_data_len)

    if global_data_size < m.n_splits:
        raise ValueError(
            f"number of splits n_splits={m.n_splits} greater than the number of samples {global_data_size}"
        )

    # Convert global_data_size from np.int64 to int to get platform int size (int32 on Windows, int64 on Linux) from np.arange
    global_indices = np.arange(int(global_data_size))
    if m.shuffle:
        if m.random_state is None:
            seed = bodo.libs.distributed_api.bcast_scalar(np.random.randint(0, 2**31))
            np.random.seed(seed)
        else:
            np.random.seed(m.random_state)
        np.random.shuffle(global_indices)

    local_indices = global_indices[rank_start:rank_end]

    # Compute local fold sizes so that global fold sizes match sklearn's.
    # Suppose that m.n_splits = 3, global_data_size = 22 and nranks = 4, so
    # len(X) = [6, 6, 5, 5] on each rank. We want our global fold sizes to
    # be [8, 7, 7] to match sklearn, which yields these local fold sizes:
    #
    #       fold0  fold1  fold2
    # rank0   2      2      2
    # rank1   2      2      2
    # rank2   2      2      1
    # rank3   2      1      2
    #
    # Assuming that data is evenly distributed, each local fold has exactly
    # `global_data_size // (nranks * m.n_splits)` elements or maybe one more.
    # First, we compute the number of extra elements per fold, and further
    # subdivide into [4, 3, 3] extra elements for folds 0, 1, and 2. We use
    # np.repeat() to expand this into [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]. Now,
    # slicing this array by `my_rank::n_ranks` tells us which local folds get
    # an extra element in the current rank. Example: In rank 0, arr[0::4] gives
    # [arr[0], arr[4], arr[8]] which is [0, 1, 2]; while in rank 2, arr[2::4]
    # gives [arr[2], arr[6]] which is [0, 1].

    local_fold_sizes = np.full(
        m.n_splits, global_data_size // (nranks * m.n_splits), dtype=np.int32
    )

    n_extras = global_data_size % (nranks * m.n_splits)
    extras_per_local_fold = np.full(m.n_splits, n_extras // m.n_splits, dtype=int)
    extras_per_local_fold[: n_extras % m.n_splits] += 1

    global_extra_locs = np.repeat(np.arange(m.n_splits), extras_per_local_fold)
    local_extra_locs = global_extra_locs[my_rank::nranks]
    local_fold_sizes[local_extra_locs] += 1

    start = 0
    for fold_size in local_fold_sizes:
        stop = start + fold_size
        test_index = local_indices[start:stop]
        train_index = np.concatenate(
            (local_indices[:start], local_indices[stop:]), axis=0
        )
        yield train_index, test_index
        start = stop


@overload_method(BodoModelSelectionKFoldType, "split", no_unliteral=True)
def overload_model_selection_kfold_generator(
    m,
    X,
    y=None,
    groups=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementations for the split function.
    In case input is replicated, we simply call sklearn,
    else we use our native implementation.

    Although the split function is a generator in sklearn, returning it as an
    opaque type to the user (as we do in LeavePOut) means we cannot iterate
    through folds in jitted code. As a quick hack to fix this, we return the
    result as a list of (train_idx, test_idx) tuples across all folds.
    This has O(nk) memory cost instead of O(n) for the generator case.

    Properly supporting split by returning an actual generator would require
    lowering the generator to numba and implementing `getiter` and `iternext`.
    """

    is_int32 = (
        sys.platform == "win32" and np.lib.NumpyVersion(np.__version__) < "2.0.0b1"
    )
    out_type = (
        "List(UniTuple(int32[:], 2))" if is_int32 else "List(UniTuple(int64[:], 2))"
    )

    if is_overload_true(_is_data_distributed):
        # If distributed, then use native implementation

        def _model_selection_kfold_generator_impl(
            m, X, y=None, groups=None, _is_data_distributed=False
        ):  # pragma: no cover
            # Since we do not support iterating through generators directly,
            # as an unperformant hack, we convert the output to a list
            with bodo.objmode(gen=out_type):
                gen = list(
                    sklearn_model_selection_kfold_generator_dist_helper(
                        m, X, y=None, groups=None
                    )
                )

            return gen

    else:
        # If replicated, then just call sklearn

        def _model_selection_kfold_generator_impl(
            m, X, y=None, groups=None, _is_data_distributed=False
        ):  # pragma: no cover
            # Since we do not support iterating through generators directly,
            # as an unperformant hack, we convert the output to a list
            with bodo.objmode(gen=out_type):
                gen = list(m.split(X, y=y, groups=groups))

            return gen

    return _model_selection_kfold_generator_impl


@overload_method(BodoModelSelectionKFoldType, "get_n_splits", no_unliteral=True)
def overload_model_selection_kfold_get_n_splits(
    m,
    X=None,
    y=None,
    groups=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Provide implementations for the get_n_splits function.
    We simply return the model's value of n_splits.
    """

    def _model_selection_kfold_get_n_splits_impl(
        m, X=None, y=None, groups=None, _is_data_distributed=False
    ):  # pragma: no cover
        with bodo.objmode(out="int64"):
            out = m.n_splits
        return out

    return _model_selection_kfold_get_n_splits_impl


# ---------------------------------------------------------------------------------------
# -----------------------------------train_test_split------------------------------------
def get_data_slice_parallel(data, labels, len_train):  # pragma: no cover
    """When shuffle=False, just split the data/labels using slicing.
    Run in bodo.jit to do it across ranks"""
    data_train = data[:len_train]
    data_test = data[len_train:]
    data_train = bodo.rebalance(data_train)
    data_test = bodo.rebalance(data_test)
    # TODO: labels maynot be present
    labels_train = labels[:len_train]
    labels_test = labels[len_train:]
    labels_train = bodo.rebalance(labels_train)
    labels_test = bodo.rebalance(labels_test)
    return data_train, data_test, labels_train, labels_test


@numba.njit
def get_train_test_size(train_size, test_size):  # pragma: no cover
    """Set train_size and test_size values"""
    if train_size is None:
        train_size = -1.0
    if test_size is None:
        test_size = -1.0
    if train_size == -1.0 and test_size == -1.0:
        return 0.75, 0.25
    elif test_size == -1.0:
        return train_size, 1.0 - train_size
    elif train_size == -1.0:
        return 1.0 - test_size, test_size
    elif train_size + test_size > 1:
        raise ValueError(
            "The sum of test_size and train_size, should be in the (0, 1) range. Reduce test_size and/or train_size."
        )
    else:
        return train_size, test_size


# TODO: labels can be 2D (We don't currently support multivariate in any ML algorithm.)


def set_labels_type(labels, label_type):  # pragma: no cover
    return labels


@overload(set_labels_type, no_unliteral=True)
def overload_set_labels_type(labels, label_type):
    """Change labels type to be same as data variable type if they are different"""
    if get_overload_const_int(label_type) == 1:

        def _set_labels(labels, label_type):  # pragma: no cover
            # Make it a series
            return pd.Series(labels)

        return _set_labels

    elif get_overload_const_int(label_type) == 2:

        def _set_labels(labels, label_type):  # pragma: no cover
            # Get array from labels series
            return labels.values

        return _set_labels
    else:

        def _set_labels(labels, label_type):  # pragma: no cover
            return labels

        return _set_labels


def reset_labels_type(labels, label_type):  # pragma: no cover
    return labels


@overload(reset_labels_type, no_unliteral=True)
def overload_reset_labels_type(labels, label_type):
    """Reset labels to its original type if changed"""
    if get_overload_const_int(label_type) == 1:

        def _reset_labels(labels, label_type):  # pragma: no cover
            # Change back to array
            return labels.values

        return _reset_labels
    elif get_overload_const_int(label_type) == 2:

        def _reset_labels(labels, label_type):  # pragma: no cover
            # Change back to Series
            return pd.Series(labels, index=np.arange(len(labels)))

        return _reset_labels
    else:

        def _reset_labels(labels, label_type):  # pragma: no cover
            return labels

        return _reset_labels


# Overload to use train_test_split inside Bodo functions
# directly via sklearn's API
@overload(sklearn.model_selection.train_test_split, no_unliteral=True)
def overload_train_test_split(
    data,
    labels=None,
    train_size=None,
    test_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
    _is_data_distributed=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    """
    Implement train_test_split. If data is replicated, run sklearn version.
    If data is distributed and shuffle=False, use slicing and then rebalance across ranks
    If data is distributed and shuffle=True, generate a global train/test mask, shuffle, and rebalance across ranks.
    """
    check_sklearn_version()
    # TODO: Check if labels is None and change output accordingly
    # no_labels = False
    # if is_overload_none(labels):
    #    no_labels = True
    args_dict = {
        "stratify": stratify,
    }

    args_default_dict = {
        "stratify": None,
    }
    check_unsupported_args("train_test_split", args_dict, args_default_dict, "ml")
    # If data is replicated, run scikit-learn directly

    if is_overload_false(_is_data_distributed):
        data_type_name = f"data_split_type_{numba.core.ir_utils.next_label()}"
        labels_type_name = f"labels_split_type_{numba.core.ir_utils.next_label()}"
        for d, d_type_name in ((data, data_type_name), (labels, labels_type_name)):
            if isinstance(d, (DataFrameType, SeriesType)):
                d_typ = d.copy(index=NumericIndexType(types.int64))
                setattr(types, d_type_name, d_typ)
            else:
                setattr(types, d_type_name, d)
        func_text = "def _train_test_split_impl(\n"
        func_text += "    data,\n"
        func_text += "    labels=None,\n"
        func_text += "    train_size=None,\n"
        func_text += "    test_size=None,\n"
        func_text += "    random_state=None,\n"
        func_text += "    shuffle=True,\n"
        func_text += "    stratify=None,\n"
        func_text += "    _is_data_distributed=False,\n"
        func_text += "):  # pragma: no cover\n"
        func_text += f"    with bodo.objmode(data_train='{data_type_name}', data_test='{data_type_name}', labels_train='{labels_type_name}', labels_test='{labels_type_name}'):\n"
        func_text += "        data_train, data_test, labels_train, labels_test = sklearn.model_selection.train_test_split(\n"
        func_text += "            data,\n"
        func_text += "            labels,\n"
        func_text += "            train_size=train_size,\n"
        func_text += "            test_size=test_size,\n"
        func_text += "            random_state=random_state,\n"
        func_text += "            shuffle=shuffle,\n"
        func_text += "            stratify=stratify,\n"
        func_text += "        )\n"
        func_text += "    return data_train, data_test, labels_train, labels_test\n"
        loc_vars = {}
        exec(func_text, globals(), loc_vars)
        _train_test_split_impl = loc_vars["_train_test_split_impl"]
        return _train_test_split_impl
    else:
        global get_data_slice_parallel
        if isinstance(get_data_slice_parallel, pytypes.FunctionType):
            get_data_slice_parallel = bodo.jit(
                get_data_slice_parallel,
                all_args_distributed_varlength=True,
                all_returns_distributed=True,
            )

        # Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.

        label_type = 0
        # 0: no change, 1: change to series, 2: change to array
        if isinstance(data, DataFrameType) and isinstance(labels, types.Array):
            label_type = 1
        elif isinstance(data, types.Array) and isinstance(labels, (SeriesType)):
            label_type = 2
        if is_overload_none(random_state):
            random_state = 42

        def _train_test_split_impl(
            data,
            labels=None,
            train_size=None,
            test_size=None,
            random_state=None,
            shuffle=True,
            stratify=None,
            _is_data_distributed=False,
        ):  # pragma: no cover
            if data.shape[0] != labels.shape[0]:
                raise ValueError(
                    "Found input variables with inconsistent number of samples\n"
                )
            train_size, test_size = get_train_test_size(train_size, test_size)
            # Get total size of data on each rank
            global_data_size = bodo.libs.distributed_api.dist_reduce(
                len(data), np.int32(Reduce_Type.Sum.value)
            )
            len_train = int(global_data_size * train_size)
            len_test = global_data_size - len_train

            if shuffle:
                # Check type. This is needed for shuffle behavior.
                labels = set_labels_type(labels, label_type)

                my_rank = bodo.get_rank()
                nranks = bodo.get_size()
                rank_data_len = np.empty(nranks, np.int64)
                bodo.libs.distributed_api.allgather(rank_data_len, len(data))
                rank_offset = np.cumsum(rank_data_len[0 : my_rank + 1])
                # Create mask where True is for training and False for testing
                global_mask = np.full(global_data_size, True)
                global_mask[:len_test] = False
                np.random.seed(42)
                np.random.permutation(global_mask)
                # Let each rank find its train/test dataset
                if my_rank:
                    start = rank_offset[my_rank - 1]
                else:
                    start = 0
                end = rank_offset[my_rank]
                local_mask = global_mask[start:end]

                data_train = data[local_mask]
                data_test = data[~local_mask]
                labels_train = labels[local_mask]
                labels_test = labels[~local_mask]

                data_train = bodo.random_shuffle(
                    data_train, seed=random_state, parallel=True
                )
                data_test = bodo.random_shuffle(
                    data_test, seed=random_state, parallel=True
                )
                labels_train = bodo.random_shuffle(
                    labels_train, seed=random_state, parallel=True
                )
                labels_test = bodo.random_shuffle(
                    labels_test, seed=random_state, parallel=True
                )

                # Restore type
                labels_train = reset_labels_type(labels_train, label_type)
                labels_test = reset_labels_type(labels_test, label_type)
            else:
                (
                    data_train,
                    data_test,
                    labels_train,
                    labels_test,
                ) = get_data_slice_parallel(data, labels, len_train)

            return data_train, data_test, labels_train, labels_test

        return _train_test_split_impl
