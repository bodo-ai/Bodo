import typing as pt

import numpy as np
import pandas as pd
from pandas.core.arrays import ExtensionArray
from pandas.core.internals.blocks import (
    Block,
    new_block,
)
from pandas.core.internals.managers import (
    BlockManager,
    SingleBlockManager,
)

import bodo.user_logging
from bodo.pandas.lazy_metadata import LazyMetadataMixin
from bodo.submit.spawner import debug_msg


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
            dummy_blocks = head.blocks
            # XXX Copy?
            col_index = [head.axes[0]]
            row_indexes = []
            for ss_axis in head.axes[1:]:
                # BSE-4099: Support other types of indexes
                if isinstance(ss_axis, pd.RangeIndex):
                    row_indexes.append(
                        pd.RangeIndex(
                            ss_axis.start,
                            ss_axis.start + (ss_axis.step * nrows),
                            ss_axis.step,
                        )
                    )
                else:
                    raise ValueError("Only RangeIndex is supported!")

            obj = super().__new__(
                cls,
                tuple(dummy_blocks),
                col_index + row_indexes,
                verify_integrity=False,
            )
            obj._md_nrows = nrows
            obj._md_head = head
            obj._md_result_id = result_id
            return obj
        else:
            # This is the normal BlockManager case
            obj = super().__new__(*args, **kwargs)
            obj._md_nrows = None
            obj._md_head = None
            obj._md_result_id = None
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
            new_blocks = []
            assert self._md_nrows is not None
            assert self._md_head is not None
            for block in self._md_head.blocks:
                arr = block.values
                # TODO:: Get data from workers BSE-4095
                # Just duplicate the head to get the full array for testing
                if isinstance(arr, ExtensionArray):
                    b_ncols, b_nrows = block.shape
                    repl_ct = (self._md_nrows // b_nrows) + 1
                    new_blocks.append(
                        new_block(
                            values=type(arr)._concat_same_type([arr] * repl_ct)[
                                : self._md_nrows
                            ],
                            placement=block._mgr_locs,
                            ndim=self.ndim,
                        )
                    )
                elif isinstance(arr, np.ndarray):
                    b_ncols, b_nrows = block.shape
                    repl_ct = (self._md_nrows // b_nrows) + 1
                    new_blocks.append(
                        new_block(
                            values=np.hstack([arr] * repl_ct)[:, : self._md_nrows],
                            placement=block._mgr_locs,
                            ndim=self.ndim,
                        ),
                    )
                else:
                    raise ValueError(f"Unsupported array type: {type(arr)}")

            self.blocks = tuple(new_blocks)
            self._md_result_id = None
            self._md_nrows = None
            self._md_head = None
            BlockManager._rebuild_blknos_and_blklocs(self)

    def __getattribute__(self, name: str) -> pt.Any:
        """
        Intercept attribute access to collect data from workers if needed.
        """
        # These attributes should be accessed directly but aren't part of the superclass
        if name in {"_collect", "_md_nrows", "_md_head", "_md_result_id", "logger"}:
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
        if self._md_result_id is not None and (
            (r_id := self._md_result_id) is not None
        ):
            # TODO: Delete data BSE-4096
            debug_msg(
                self.logger,
                f"[LazyBlockManager] Asking workers to delete result '{r_id}'",
            )


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
    ):
        block_ = block
        axis_ = axis
        self._md_nrows = nrows
        self._md_result_id = result_id
        self._md_head = head
        if result_id is not None:
            assert nrows is not None
            assert result_id is not None
            assert head is not None
            # Replace with a dummy block for now.
            block_ = head.blocks[0]
            # Create axis based on head
            head_axis = head.axes[0]
            # BSE-4099: Support other types of indexes
            if isinstance(head_axis, pd.RangeIndex):
                axis_ = pd.RangeIndex(
                    head_axis.start,
                    head_axis.start + (head_axis.step * nrows),
                    head_axis.step,
                )
            else:
                raise ValueError("Only RangeIndex is supported!")

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
            debug_msg(
                self.logger, "[LazySingleBlockManager] Collecting data from workers..."
            )
            head_block = self._md_head._block
            arr = head_block.values
            # TODO:: Get data from workers BSE-4095
            # Just duplicate the head to get the full array for testing
            if isinstance(arr, ExtensionArray):
                # TODO Verify this logic
                repl_ct = (self._md_nrows // len(arr)) + 1
                self.blocks = (
                    new_block(
                        values=type(arr)._concat_same_type([arr] * repl_ct)[
                            : self._md_nrows
                        ],
                        placement=head_block._mgr_locs,
                        ndim=self.ndim,
                    ),
                )
            elif isinstance(arr, np.ndarray):
                repl_ct = (self._md_nrows // len(arr)) + 1
                self.blocks = (
                    new_block(
                        values=np.hstack([arr] * repl_ct)[: self._md_nrows],
                        placement=head_block._mgr_locs,
                        ndim=self.ndim,
                    ),
                )
            else:
                raise ValueError(f"Unsupported array type: {type(arr)}")

            self._md_result_id = None
            self._md_nrows = None
            self._md_head = None

    # BSE-4097
    # TODO Override get_slice for s.head() support BSE-4097
    # TODO Override __len__

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
        }:
            return object.__getattribute__(self, name)
        if name == "blocks":
            self._collect()
        return super().__getattribute__(name)

    def __del__(self):
        """
        Delete the result from the workers if it hasn't been collected yet.
        """
        if self._md_result_id is not None and (
            (r_id := self._md_result_id) is not None
        ):
            # TODO: Delete data BSE-4096
            debug_msg(
                self.logger,
                f"[LazySingleBlockManager] Asking workers to delete result '{r_id}'",
            )
