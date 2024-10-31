import typing as pt

import pyarrow as pa
from pandas.core.arrays.arrow.array import ArrowExtensionArray

import bodo.user_logging
from bodo.pandas.lazy_metadata import LazyMetadataMixin


class LazyArrowExtensionArray(
    ArrowExtensionArray, LazyMetadataMixin[ArrowExtensionArray]
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
    ):
        self._md_nrows = nrows
        self._md_result_id = result_id
        self._md_head = head
        self.logger = bodo.user_logging.get_current_bodo_verbose_logger()
        if self._md_result_id is not None:
            assert nrows is not None
            assert head is not None
            self._pa_array = None
            self._dtype = head._dtype
        else:
            super().__init__(values)

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
        # TODO:: Get data from workers BSE-4095
        if self._md_result_id is not None:
            # Just duplicate the head to get the full array for testing
            assert self._md_head is not None
            assert self._md_nrows is not None
            self.logger.debug(
                "[LazyArrowExtensionArray] Collecting data from workers..."
            )
            repl_ct = (self._md_nrows // len(self._md_head)) + 1
            new_array = type(self._md_head)._concat_same_type(
                [self._md_head] * repl_ct
            )[: self._md_nrows]
            self._pa_array = new_array._pa_array
            self._md_result_id = None
            self._md_nrows = None
            self._md_head = None

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
            # TODO: Delete data BSE-4096
            self.logger.debug(
                f"[LazyArrowExtensionArray] Asking workers to delete result '{r_id}'"
            )
