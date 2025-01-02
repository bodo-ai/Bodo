import typing as pt
from collections.abc import Callable

import pyarrow as pa
from pandas.core.arrays.arrow.array import ArrowExtensionArray

import bodo.user_logging
from bodo.pandas.lazy_metadata import LazyMetadata, LazyMetadataMixin
from bodo.pandas.lazy_wrapper import BodoLazyWrapper
from bodo.spawn.utils import debug_msg


class LazyArrowExtensionArray(
    ArrowExtensionArray, LazyMetadataMixin[ArrowExtensionArray], BodoLazyWrapper
):
    """
    A lazy ArrowExtensionArray that will collect data from workers when needed. Also functions as a normal ArrowExtensionArray.
    """

    def __init__(
        self,
        values: pa.Array | pa.ChunkedArray,
        *,
        nrows: int | None = None,
        result_id: str | None = None,
        head: ArrowExtensionArray | None = None,
        collect_func: Callable[[str], pt.Any] | None = None,
        del_func: Callable[[str], None] | None = None,
    ):
        self._md_nrows = nrows
        self._md_result_id = result_id
        self._md_head = head
        self.logger = bodo.user_logging.get_current_bodo_verbose_logger()
        self._collect_func = collect_func
        self._del_func = del_func
        if self._md_result_id is not None:
            assert nrows is not None
            assert head is not None
            assert collect_func is not None
            assert del_func is not None
            self._pa_array = None
            self._dtype = head._dtype
        else:
            super().__init__(values)

    @classmethod
    def from_lazy_metadata(
        cls,
        lazy_metadata: LazyMetadata,
        collect_func: Callable[[str], pt.Any] | None = None,
        del_func: Callable[[str], None] | None = None,
    ) -> "LazyArrowExtensionArray":
        """
        Create a LazyArrowExtensionArray from a lazy metadata object.
        """
        assert isinstance(lazy_metadata.head, ArrowExtensionArray)
        return cls(
            None,
            nrows=lazy_metadata.nrows,
            result_id=lazy_metadata.result_id,
            head=lazy_metadata.head,
            collect_func=collect_func,
            del_func=del_func,
        )

    def update_from_lazy_metadata(self, lazy_metadata: LazyMetadata):
        """
        Update this array from a lazy metadata object.
        """
        assert self._lazy
        assert isinstance(lazy_metadata.head, ArrowExtensionArray)
        # Call del since we are updating the array and the old result won't have a reference anymore
        self._del_func(self._md_result_id)
        self._md_nrows = lazy_metadata.nrows
        self._md_result_id = lazy_metadata.result_id
        self._md_head = lazy_metadata.head

    def __len__(self) -> int:
        """
        Length of this array.
        Metadata is used if present otherwise fallsback to superclass implementation

        Returns
        -------
        length : int
        """
        if self._md_nrows is not None:
            return self._md_nrows
        return super().__len__()

    def _collect(self):
        """
        Collects data from workers if it has not been collected yet.
        """
        if self._md_result_id is not None:
            # Just duplicate the head to get the full array for testing
            assert self._md_head is not None
            assert self._md_nrows is not None
            assert self._collect_func is not None
            debug_msg(
                self.logger, "[LazyArrowExtensionArray] Collecting data from workers..."
            )
            collected = self._collect_func(self._md_result_id)
            # Collected could be bodo array types too so we need to convert to pyarrow
            # array. The _pa_array attribute should be a ChunkedArray object. See:
            # https://github.com/pandas-dev/pandas/blob/d9cdd2ee5a58015ef6f4d15c7226110c9aab8140/pandas/core/arrays/arrow/array.py#L294
            self._pa_array = pa.chunked_array(
                [pa.array(collected, type=self._md_head._pa_array.type)]
            )
            self._md_result_id = None
            self._md_nrows = None
            self._md_head = None
            self._collect_func = None

    def __getattribute__(self, name: str) -> pt.Any:
        """
        Overridden to collect data from workers if needed before accessing any attribute.
        """
        # We want to access these but can't use the super __getattribute__ because they don't exist in ArrowExtensionArray
        if name in {"_collect", "_md_nrows", "_md_result_id", "_md_head", "logger"}:
            return object.__getattribute__(self, name)
        if name == "_pa_array":
            self._collect()
        return ArrowExtensionArray.__getattribute__(self, name)

    def __del__(self):
        """
        Delete the result from workers if it exists.
        """
        if (r_id := self._md_result_id) is not None:
            assert self._del_func is not None
            self._del_func(r_id)
            self._del_func = None

    def _get_result_id(self) -> str | None:
        return self._md_result_id
