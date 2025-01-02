"""LazyArrayManager and LazySingleArrayManager classes for lazily loading data from workers in BodoSeries/DataFrames."""

import typing as pt
from collections.abc import Callable

import numpy as np
import pandas as pd
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.arrow.array import ArrowExtensionArray

try:
    from pandas.core.internals.array_manager import ArrayManager, SingleArrayManager
except ModuleNotFoundError:
    # Pandas > 2.2 does not have an array_manager module (uses BlockManager/SinglBlockManager).
    class ArrayManager:
        pass

    class SingleArrayManager:
        pass


import bodo.user_logging
from bodo.pandas.lazy_metadata import LazyMetadataMixin
from bodo.spawn.utils import debug_msg


class LazyArrayManager(ArrayManager, LazyMetadataMixin[ArrayManager]):
    """
    ArrayManager to lazily load data from workers in BodoDataFrames. It must also function as a normal ArrayManager
    since some pandas functions call the passed in ArrayManager's constructor.
    """

    # Use __slots__ to avoid creating __dict__ and __weakref__ for each instance, store it like a C struct
    __slots__ = [
        "_md_nrows",
        "_md_head",
        "_md_result_id",
        "_collect_func",
        "_del_func",
        "logger",
    ]

    def __init__(
        self,
        # Normal ArrayManager arguments
        arrays: list[np.ndarray | ExtensionArray],
        axes: list[pd.Index],
        verify_integrity: bool = False,
        *,
        # LazyArrayManager specific arguments
        result_id: str | None = None,
        nrows: int | None = None,
        head: ArrayManager | None = None,
        collect_func: Callable[[str], pt.Any] | None = None,
        del_func: Callable[[str], None] | None = None,
        # Can be used for lazy index data
        index_data: ArrowExtensionArray
        | tuple[ArrowExtensionArray, ArrowExtensionArray]
        | pd.DataFrame
        | None = None,
    ):
        self._axes = axes
        self.arrays = arrays
        _arrays = arrays
        self._md_result_id = result_id
        self._md_nrows = nrows
        self._md_head = head
        self.logger = bodo.user_logging.get_current_bodo_verbose_logger()
        self._collect_func = collect_func
        self._del_func = del_func

        if result_id is not None:
            # This is the lazy case, we don't have the full data yet
            assert nrows is not None
            assert head is not None
            assert collect_func is not None
            assert del_func is not None

            head_axis0 = head._axes[0]  # Per row
            head_axis1 = head._axes[1]  # Per column

            new_axis0 = None
            # BSE-4099: Support other types of indexes
            match type(head_axis0):
                case pd.RangeIndex:
                    new_axis0 = pd.RangeIndex(
                        head_axis0.start,
                        head_axis0.start + (head_axis0.step * nrows),
                        head_axis0.step,
                        name=head_axis0.name,
                    )
                case pd.Index:
                    new_axis0 = pd.Index(index_data, name=head_axis0.name)
                case pd.MultiIndex:
                    new_axis0 = pd.MultiIndex.from_frame(
                        index_data,
                        sortorder=head_axis0.sortorder,
                        names=head_axis0.names,
                    )
                case pd.IntervalIndex:
                    assert index_data is not None
                    new_axis0 = pd.IntervalIndex.from_arrays(
                        index_data[0],
                        index_data[1],
                        head_axis0.closed,
                        head_axis0.name,
                        dtype=head_axis0.dtype,
                    )
                case pd.CategoricalIndex:
                    assert index_data is not None
                    new_axis0 = pd.CategoricalIndex(
                        index_data,
                        categories=head_axis0.categories,
                        ordered=head_axis0.ordered,
                        name=head_axis0.name,
                    )
                case pd.DatetimeIndex:
                    assert index_data is not None
                    new_axis0 = pd.DatetimeIndex(
                        index_data,
                        name=head_axis0.name,
                        tz=head_axis0.tz,
                        freq=head_axis0.freq,
                    )
                case pd.PeriodIndex:
                    assert index_data is not None
                    new_axis0 = index_data
                case pd.TimedeltaIndex:
                    assert index_data is not None
                    new_axis0 = pd.TimedeltaIndex(
                        index_data, name=head_axis0.name, unit=head_axis0.unit
                    )
                case _:
                    raise ValueError(
                        "{type(head_axis0)} is not supported in LazyArrayManager"
                    )

            self._axes = [
                new_axis0,
                head_axis1,
            ]
            self.arrays = None  # type: ignore This can't be None when accessed because we overload __getattribute__
            _arrays = None
        else:
            # This is the base ArrayManager case
            assert nrows is None
            assert head is None
        super().__init__(
            _arrays,
            self._axes,
            verify_integrity=(verify_integrity if (result_id is None) else False),
        )

    @property
    def is_single_block(self) -> bool:
        if self._md_head is not None:
            # Just check the head if we don't have the data yet
            return len(self._md_head.arrays) == 1
        else:
            # Same as the base ArrayManager
            assert self.arrays is not None
            return len(self.arrays) == 1

    def get_dtypes(self) -> np.typing.NDArray[np.object_]:
        """
        Get dtypes of the arrays in the manager.
        Uses head if we don't have the data yet, otherwise uses the base ArrayManager's get_dtypes.
        """
        if self._md_head is not None:
            return self._md_head.get_dtypes()
        return super().get_dtypes()

    def __repr__(self) -> str:
        """
        Print the representation of the ArrayManager.
        Uses head if we don't have the data yet, otherwise uses the full arrays.
        """
        output = type(self).__name__
        output += f"\nIndex: {self._axes[0]}"
        if self.ndim == 2:
            output += f"\nColumns: {self._axes[1]}"
        if self._md_head is not None:
            output += f"\n{len(self._md_head.arrays)} arrays:"
            for arr in self._md_head.arrays:
                output += f"\n{arr.dtype}"
        else:
            output += f"\n{len(self.arrays)} arrays:"
            for arr in self.arrays:
                output += f"\n{arr.dtype}"
        return output

    # This is useful for cases like df.head()
    def get_slice(self, slobj: slice, axis: int = 0) -> ArrayManager:
        """
        Returns a new ArrayManager with the data sliced along the given axis.
        If we don't have the data yet, and the slice is within the head, we slice the head,
        otherwise we collect and slice the full data. A slice along axis 1 will always lead to a full collection.
        """
        axis = self._normalize_axis(axis)

        # Normalize negative and None start/stop/step values
        start, stop, step = slobj.indices(len(self))

        # TODO Check if this condition is correct.
        if (
            self._md_head is not None
            and start <= self._md_head.shape[1]
            and stop <= self._md_head.shape[1]
            and axis == 0
        ):
            slobj = slice(start, stop, step)
            tmp_arrs = self._md_head.arrays
            arrays = [arr[slobj] for arr in tmp_arrs]
            new_axes = list(self._axes)
            new_axes[axis] = new_axes[axis]._getitem_slice(slobj)
            return ArrayManager(arrays, new_axes, verify_integrity=False)

        if axis == 0:
            arrays = [arr[slobj] for arr in self.arrays]
        elif axis == 1:
            arrays = self.arrays[slobj]
        else:
            raise IndexError("Requested axis not found in manager")

        new_axes = list(self._axes)
        new_axes[axis] = new_axes[axis]._getitem_slice(slobj)

        return type(self)(arrays, new_axes, verify_integrity=False)

    def _collect(self):
        """
        Collect the data from the workers if we don't have it and clear metadata.
        """
        if self._md_result_id is not None:
            assert self._md_head is not None
            assert self._md_nrows is not None
            assert self._collect_func is not None
            debug_msg(self.logger, "[LazyArrayManager] Collecting data...")
            data = self._collect_func(self._md_result_id)
            self.arrays = data._mgr.arrays

            self._md_result_id = None
            self._md_head = None
            self._md_nrows = None
            # Collect should only be done once
            self._collect_func = None

    def __getattribute__(self, name: str) -> pt.Any:
        """
        Overload __getattribute__ to handle lazy loading of data.
        """
        # Overriding LazyArrayManager attributes so we can use ArrayManager's __getattribute__
        if name in {
            "_collect",
            "_md_nrows",
            "_md_head",
            "_md_result_id",
            "logger",
            "_collect_func",
            "_del_func",
        }:
            return object.__getattribute__(self, name)
        # If the attribute is 'arrays', we ensure we have the data.
        if name == "arrays":
            self._collect()
        return ArrayManager.__getattribute__(self, name)

    def __del__(self):
        """
        Handles cleanup of the result on deletion. If we have a result ID, we ask the workers to delete the result,
        otherwise we do nothing because the data is already collected/deleted.
        """
        if (r_id := self._md_result_id) is not None:
            assert self._del_func is not None
            self._del_func(r_id)
            self._del_func = None

    def __len__(self) -> int:
        """
        Get length of the arrays in the manager.
        Uses nrows if we don't have the data yet.
        Otherwise, verify that data shape is not corrupted and return nrows.
        Note that we cannot simply call the super implementation due to layout axis being swapped.
        """
        if self._md_head is not None:
            return self._md_nrows
        self._verify_integrity()
        return self.shape_proper[0]


class LazySingleArrayManager(SingleArrayManager, LazyMetadataMixin[SingleArrayManager]):
    """
    ArrayManager to lazily load data from workers in BodoSeries. It must also function as a normal SingleArrayManager
    since some pandas functions call the passed in ArrayManager's constructor. Very similar to LazyArrayManager, but only for a single array.
    """

    # Use __slots__ to avoid creating __dict__ and __weakref__ for each instance, store it like a C struct
    __slots__ = [
        "_md_nrows",
        "_md_head",
        "_md_result_id",
        "_collect_func",
        "_del_func",
        "logger",
    ]

    def __init__(
        self,
        # Normal SingleArrayManager arguments
        arrays: list[np.ndarray | ExtensionArray],
        axes: list[pd.Index],
        verify_integrity: bool = True,
        # LazyArrayManager specific arguments
        result_id: str | None = None,
        nrows: int | None = None,
        head: SingleArrayManager | None = None,
        collect_func: Callable[[str], pt.Any] | None = None,
        del_func: Callable[[str], None] | None = None,
        # Can be used for lazy index data
        index_data: ArrowExtensionArray
        | tuple[ArrowExtensionArray, ArrowExtensionArray]
        | pd.DataFrame
        | None = None,
    ):
        self._axes = axes
        self.arrays = arrays

        _arrays = arrays
        self._md_result_id = result_id
        self._md_nrows = nrows
        self._md_head = head
        self.logger = bodo.user_logging.get_current_bodo_verbose_logger()
        self._collect_func = collect_func
        self._del_func = del_func

        if result_id is not None:
            # This is the lazy case, we don't have the full data yet
            assert nrows is not None
            assert head is not None
            assert collect_func is not None
            assert del_func is not None

            head_axis = head._axes[0]
            new_axis = None
            # BSE-4099: Support other types of indexes
            match type(head_axis):
                case pd.RangeIndex:
                    new_axis = pd.RangeIndex(
                        head_axis.start,
                        head_axis.start + (head_axis.step * nrows),
                        head_axis.step,
                        name=head_axis.name,
                    )
                case pd.Index:
                    new_axis = pd.Index(index_data, name=head_axis.name)
                case pd.MultiIndex:
                    new_axis = pd.MultiIndex.from_frame(
                        index_data, sortorder=head_axis.sortorder, names=head_axis.names
                    )
                case pd.IntervalIndex:
                    assert index_data is not None
                    new_axis = pd.IntervalIndex.from_arrays(
                        index_data[0],
                        index_data[1],
                        head_axis.closed,
                        head_axis.name,
                        dtype=head_axis.dtype,
                    )
                case pd.CategoricalIndex:
                    assert index_data is not None
                    new_axis = pd.CategoricalIndex(
                        index_data,
                        categories=head_axis.categories,
                        ordered=head_axis.ordered,
                        name=head_axis.name,
                    )
                case pd.DatetimeIndex:
                    assert index_data is not None
                    new_axis = pd.DatetimeIndex(
                        index_data,
                        name=head_axis.name,
                        tz=head_axis.tz,
                        freq=head_axis.freq,
                    )
                case pd.PeriodIndex:
                    assert index_data is not None
                    new_axis = index_data
                case pd.TimedeltaIndex:
                    assert index_data is not None
                    new_axis = pd.TimedeltaIndex(
                        index_data, name=head_axis.name, unit=head_axis.unit
                    )

                case _:
                    raise ValueError(
                        "{type(head_axis)} is not supported in LazySingleArrayManager"
                    )
            self._axes = [new_axis]
            self.arrays = None  # type: ignore This is can't be None when accessed because we overload __getattribute__
            _arrays = None
        else:
            # This is the base ArrayManager case
            assert nrows is None
            assert head is None

        super().__init__(
            _arrays,
            self._axes,
            verify_integrity=(verify_integrity if (result_id is None) else False),
        )

    @property
    def dtype(self):
        """
        Get the dtype of the array in the manager. Uses head if we don't have the data yet, otherwise uses the base SingleArrayManager's dtype.
        """
        if self._md_head is not None:
            return self._md_head.dtype
        return super().dtype

    def _collect(self):
        """
        Collect the data from the workers if we don't have it and clear metadata.
        """
        if self._md_result_id is not None:
            debug_msg(self.logger, "[LazySingleArrayManager] Collecting data...")
            assert self._md_head is not None
            assert self._md_nrows is not None
            assert self._collect_func is not None
            data = self._collect_func(self._md_result_id)

            self.arrays = data._mgr.arrays
            self._md_result_id = None
            self._md_nrows = None
            self._md_head = None
            self._collect_func = None

    def get_slice(self, slobj: slice, axis: int = 0) -> SingleArrayManager:
        """
        Returns a new SingleArrayManager with the data sliced along the given axis.
        If we don't have the data yet, and the slice is within the head, we slice the head,
        otherwise we collect and slice the full data. A slice along axis 1 will always lead to a full collection.
        """
        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        # Normalize negative and None start/stop/step values
        start, stop, step = slobj.indices(len(self))

        if (
            (self._md_head is not None)
            and start <= len(self._md_head)
            and stop <= len(self._md_head)
            and axis == 0
        ):
            slobj = slice(start, stop, step)
            tmp_arrs = self._md_head.arrays
            arrays = [arr[slobj] for arr in tmp_arrs]
            new_axes = list(self._axes)
            new_axes[axis] = new_axes[axis]._getitem_slice(slobj)
            return SingleArrayManager(arrays, new_axes, verify_integrity=False)

        new_array = self.array[slobj]
        new_index = self.index._getitem_slice(slobj)
        return type(self)([new_array], [new_index], verify_integrity=False)

    def __repr__(self) -> str:
        """
        Print the representation of the SingleArrayManager.
        Uses head if we don't have the data yet, otherwise uses the full arrays.
        """
        output = type(self).__name__
        output += f"\nIndex: {self._axes[0]}"
        if self.ndim == 2:
            output += f"\nColumns: {self._axes[1]}"
        output += "\n1 arrays:"
        if self._md_head is not None:
            arr = self._md_head.array
        else:
            arr = self.array
        output += f"\n{arr.dtype}"
        return output

    def __getattribute__(self, name: str) -> pt.Any:
        """
        Overload __getattribute__ to handle lazy loading of data.
        """
        # Overriding LazyArrayManager attributes so we can use SingleArrayManager's __getattribute__
        if name in {
            "_collect",
            "_md_nrows",
            "_md_result_id",
            "_md_head",
            "logger",
            "_collect_func",
            "_del_func",
        }:
            return object.__getattribute__(self, name)
        # If the attribute is 'arrays', we ensure we have the data.
        if name == "arrays":
            self._collect()
        return SingleArrayManager.__getattribute__(self, name)

    def __del__(self):
        """
        Handles cleanup of the result on deletion. If we have a result ID, we ask the workers to delete the result,
        otherwise we do nothing because the data is already collected/deleted.
        """
        if (r_id := self._md_result_id) is not None:
            assert self._del_func is not None
            self._del_func(r_id)
            self._del_func = None

    def __len__(self) -> int:
        """
        Get length of the arrays in the manager.
        Uses nrows if we don't have the data yet, otherwise uses the super implementation.
        """
        if self._md_head is not None:
            return self._md_nrows
        return super().__len__()
