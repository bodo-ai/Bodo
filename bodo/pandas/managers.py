import typing as pt
from collections.abc import Callable

import numpy as np
import pandas as pd
from pandas._libs.internals import (
    BlockPlacement,
)
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.internals.blocks import (
    Block,
)
from pandas.core.internals.managers import (
    BlockManager,
    SingleBlockManager,
)

import bodo.user_logging
from bodo.pandas.lazy_metadata import LazyMetadataMixin
from bodo.spawn.utils import debug_msg


class LazyBlockManager(BlockManager, LazyMetadataMixin[BlockManager]):
    """
    A BlockManager that supports lazy evaluation of data, for use in BodoDataFrames. Data will be fetched from the workers when needed.
    It must also function as a BlockManager since pandas creates new BlockManagers from existing ones.
    """

    logger = bodo.user_logging.get_current_bodo_verbose_logger()

    @classmethod
    # BlockManager is implemented in Cython so we can't override __init__ directly
    def __new__(cls, *args, **kwargs):
        if result_id := kwargs.get("result_id"):
            # This is the lazy case
            head = kwargs["head"]
            nrows = kwargs["nrows"]
            collect_func = kwargs["collect_func"]
            del_func = kwargs["del_func"]
            index_data = kwargs.get("index_data", None)
            dummy_blocks = head.blocks
            # XXX Copy?
            col_index = [head.axes[0]]
            row_indexes = []
            for ss_axis in head.axes[1:]:
                # BSE-4099: Support other types of indexes
                match type(ss_axis):
                    case pd.RangeIndex:
                        row_indexes.append(
                            pd.RangeIndex(
                                ss_axis.start,
                                ss_axis.start + (ss_axis.step * nrows),
                                ss_axis.step,
                                name=ss_axis.name,
                            )
                        )
                    case pd.Index:
                        assert index_data is not None
                        row_indexes.append(pd.Index(index_data, name=ss_axis.name))
                    case pd.MultiIndex:
                        assert index_data is not None
                        row_indexes.append(
                            pd.MultiIndex.from_frame(
                                index_data,
                                sortorder=ss_axis.sortorder,
                                names=ss_axis.names,
                            )
                        )
                    case pd.IntervalIndex:
                        assert index_data is not None
                        row_indexes.append(
                            pd.IntervalIndex.from_arrays(
                                index_data[0],
                                index_data[1],
                                ss_axis.closed,
                                ss_axis.name,
                                dtype=ss_axis.dtype,
                            )
                        )
                    case pd.CategoricalIndex:
                        assert index_data is not None
                        row_indexes.append(
                            pd.CategoricalIndex(
                                index_data,
                                categories=ss_axis.categories,
                                ordered=ss_axis.ordered,
                                name=ss_axis.name,
                            )
                        )
                    case pd.DatetimeIndex:
                        assert index_data is not None
                        row_indexes.append(
                            pd.DatetimeIndex(
                                index_data,
                                name=ss_axis.name,
                                tz=ss_axis.tz,
                                freq=ss_axis.freq,
                            )
                        )
                    case pd.PeriodIndex:
                        assert index_data is not None
                        row_indexes.append(index_data)
                    case pd.TimedeltaIndex:
                        assert index_data is not None
                        row_indexes.append(
                            pd.TimedeltaIndex(
                                index_data, name=ss_axis.name, unit=ss_axis.unit
                            )
                        )

                    case _:
                        raise ValueError(
                            f"Index type {type(ss_axis)} not supported in LazyBlockManager"
                        )

            obj = super().__new__(
                cls,
                tuple(dummy_blocks),
                col_index + row_indexes,
                verify_integrity=False,
            )
            obj._md_nrows = nrows
            obj._md_head = head
            obj._md_result_id = result_id
            obj._collect_func = collect_func
            obj._del_func = del_func
            return obj
        else:
            # This is the normal BlockManager case
            obj = super().__new__(*args, **kwargs)
            obj._md_nrows = None
            obj._md_head = None
            obj._md_result_id = None
            obj._collect_func = None
            obj._del_func = None
            return obj

    def __init__(
        self,
        blocks: pt.Sequence[Block],
        axes: pt.Sequence[pd.Index],
        verify_integrity: bool = True,
        *,
        head=None,
        nrows=None,
        result_id=None,
        collect_func: Callable[[str], pt.Any] | None = None,
        del_func: Callable[[str], None] | None = None,
        # Can be used for lazy index data
        index_data: ArrowExtensionArray
        | tuple[ArrowExtensionArray, ArrowExtensionArray]
        | None = None,
    ):
        super().__init__(
            blocks,
            axes,
            verify_integrity=verify_integrity if (result_id is None) else False,
        )
        if result_id is not None:
            # Set pandas internal metadata
            self._rebuild_blknos_and_blklocs_lazy()

    def get_dtypes(self) -> np.typing.NDArray[np.object_]:
        """
        Get the dtypes of the blocks in this BlockManager.
        Uses the head if available, otherwise falls back to the default implementation.
        """
        if self._md_head is not None:
            return self._md_head.get_dtypes()
        else:
            return super().get_dtypes()

    def __repr__(self) -> str:
        """
        Return a string representation of this BlockManager.
        Uses the head if available, otherwise falls back to the default implementation.
        """
        if self._md_head is not None:
            output = type(self).__name__
            for i, ax in enumerate(self.axes):
                if i == 0:
                    output += f"\nItems: {ax}"
                else:
                    output += f"\nAxis {i}: {ax}"
            for block in self._md_head.blocks:
                shape = f"{block.shape[0]} x {self._md_nrows}"
                output += f"\n{type(block).__name__}: {block.mgr_locs.indexer}, {shape}, dtype: {block.dtype}"
            return output
        else:
            return super().__repr__()

    def _collect(self):
        """
        Collect data from workers if needed.
        """
        if self._md_result_id is not None:
            debug_msg(self.logger, "[LazyBlockManager] Collecting data from workers...")
            assert self._md_nrows is not None
            assert self._md_head is not None
            assert self._collect_func is not None
            data = self._collect_func(self._md_result_id)
            self._collect_func = None

            self.blocks = data._mgr.blocks
            self._md_result_id = None
            self._md_nrows = None
            self._md_head = None
            BlockManager._rebuild_blknos_and_blklocs(self)

    def __getattribute__(self, name: str) -> pt.Any:
        """
        Intercept attribute access to collect data from workers if needed.
        """
        # These attributes should be accessed directly but aren't part of the superclass
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
        # Most of the time _rebuild_blknos_and_blklocs is called by pandas internals
        # and should require collecting data, but in __init__ we need to call it
        # without it triggering a collect
        if name == "_rebuild_blknos_and_blklocs_lazy":
            return object.__getattribute__(self, "_rebuild_blknos_and_blklocs")
        # These attributes require data collection
        if name in {
            "blocks",
            "get_slice",
            "_rebuild_blknos_and_blklocs",
            "__reduce__",
            "__setstate__",
            "_slice_mgr_rows",
        }:
            self._collect()
        return super().__getattribute__(name)

    def __del__(self):
        """
        Delete the result from the workers if it hasn't been collected yet.
        """
        if (r_id := self._md_result_id) is not None:
            assert self._del_func is not None
            self._del_func(r_id)
            self._del_func = None


class LazySingleBlockManager(SingleBlockManager, LazyMetadataMixin[SingleBlockManager]):
    """
    A SingleBlockManager that supports lazy evaluation of data, for use in BodoSeries. Data will be fetched from the workers when needed.
    It must also function as a SingleBlockManager since pandas creates new SingleBlockManagers from existing ones.
    """

    logger = bodo.user_logging.get_current_bodo_verbose_logger()

    def __init__(
        self,
        block: Block,
        axis: pd.Index,
        verify_integrity: bool = True,
        *,
        nrows=None,
        result_id=None,
        head=None,
        collect_func: Callable[[str], pt.Any] | None = None,
        del_func: Callable[[str], None] | None = None,
        # Can be used for lazy index data
        index_data: ArrowExtensionArray
        | tuple[ArrowExtensionArray, ArrowExtensionArray]
        | None = None,
    ):
        block_ = block
        axis_ = axis
        self._md_nrows = nrows
        self._md_result_id = result_id
        self._md_head = head
        self._collect_func = collect_func
        self._del_func = del_func
        if result_id is not None:
            assert nrows is not None
            assert result_id is not None
            assert head is not None
            assert collect_func is not None
            assert del_func is not None

            # Replace with a dummy block for now.
            block_ = head.blocks[0]
            # Create axis based on head
            head_axis = head.axes[0]
            # BSE-4099: Support other types of indexes
            match type(head_axis):
                case pd.RangeIndex:
                    axis_ = pd.RangeIndex(
                        head_axis.start,
                        head_axis.start + (head_axis.step * nrows),
                        head_axis.step,
                        name=head_axis.name,
                    )
                case pd.Index:
                    assert index_data is not None
                    axis_ = pd.Index(index_data, name=head_axis.name)
                case pd.MultiIndex:
                    axis_ = pd.MultiIndex.from_frame(
                        index_data, sortorder=head_axis.sortorder, names=head_axis.names
                    )
                case pd.IntervalIndex:
                    assert index_data is not None
                    axis_ = pd.IntervalIndex.from_arrays(
                        index_data[0],
                        index_data[1],
                        head_axis.closed,
                        head_axis.name,
                        dtype=head_axis.dtype,
                    )
                case pd.CategoricalIndex:
                    assert index_data is not None
                    axis_ = pd.CategoricalIndex(
                        index_data,
                        categories=head_axis.categories,
                        ordered=head_axis.ordered,
                        name=head_axis.name,
                    )
                case pd.DatetimeIndex:
                    assert index_data is not None
                    axis_ = pd.DatetimeIndex(
                        index_data,
                        name=head_axis.name,
                        tz=head_axis.tz,
                        freq=head_axis.freq,
                    )
                case pd.PeriodIndex:
                    assert index_data is not None
                    axis_ = index_data
                case pd.TimedeltaIndex:
                    assert index_data is not None
                    axis_ = pd.TimedeltaIndex(
                        index_data, name=head_axis.name, unit=head_axis.unit
                    )
                case _:
                    raise ValueError(
                        "Index type {type(head_axis)} not supported in LazySingleBlockManager"
                    )

        super().__init__(
            block_,
            axis_,
            verify_integrity=verify_integrity if (result_id is None) else False,
        )

    def get_dtypes(self) -> np.typing.NDArray[np.object_]:
        """
        Get the dtypes of the blocks in this BlockManager.
        Uses the head if available, otherwise falls back to the default implementation.
        """
        if self._md_head is not None:
            return self._md_head.get_dtypes()
        else:
            return super().get_dtypes()

    @property
    def dtype(self):
        """
        Get the dtype of the block in this SingleBlockManager.
        The dtype is determined by the head if available, otherwise falls back to the default implementation.
        """
        if self._md_head is not None:
            return self._md_head._block.dtype
        return self._block.dtype

    def __repr__(self) -> str:
        """
        Return a string representation of this BlockManager.
        Uses the head if available, otherwise falls back to the default implementation.
        """
        if self._md_head is not None:
            output = type(self).__name__
            for i, ax in enumerate(self.axes):
                if i == 0:
                    output += f"\nItems: {ax}"
                else:
                    output += f"\nAxis {i}: {ax}"
            head_block = self._md_head._block
            shape = f"1 x {self._md_nrows}"
            output += f"\n{type(head_block).__name__}: {head_block.mgr_locs.indexer}, {shape}, dtype: {head_block.dtype}"
            return output
        else:
            return super().__repr__()

    def _collect(self):
        """
        Collect data from workers if needed.
        """
        if self._md_result_id is not None:
            assert self._md_nrows is not None
            assert self._md_head is not None
            assert self._collect_func is not None
            debug_msg(
                self.logger, "[LazySingleBlockManager] Collecting data from workers..."
            )
            data = self._collect_func(self._md_result_id)
            self.blocks = data._mgr.blocks

            self._md_result_id = None
            self._md_nrows = None
            self._md_head = None
            self._collect_func = None

    def __getattribute__(self, name: str) -> pt.Any:
        """
        Intercept attribute access to collect data from workers if needed.
        """
        # These attributes should be accessed directly but aren't part of the superclass
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
        if name == "blocks":
            self._collect()
        return super().__getattribute__(name)

    def __del__(self):
        """
        Delete the result from the workers if it hasn't been collected yet.
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

    def get_slice(self, slobj: slice, axis: int = 0) -> SingleBlockManager:
        """
        Returns a new SingleBlockManager with the data sliced along the given axis.
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
            tmp_block = self._md_head._block
            array = tmp_block.values[slobj]
            bp = BlockPlacement(slice(0, len(array)))
            block = type(tmp_block)(array, placement=bp, ndim=1, refs=tmp_block.refs)
            new_index = self.index._getitem_slice(slobj)
            return type(self)(block, new_index)

        blk = self._block
        array = blk.values[slobj]
        bp = BlockPlacement(slice(0, len(array)))
        block = type(blk)(array, placement=bp, ndim=1, refs=blk.refs)
        new_index = self.index._getitem_slice(slobj)
        return type(self)(block, new_index)
