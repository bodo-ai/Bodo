import typing as pt
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from pandas.core.arrays import ArrowExtensionArray

if pt.TYPE_CHECKING:
    from bodo.pandas.array_manager import LazyArrayManager, LazySingleArrayManager
    from bodo.pandas.arrow.array import LazyArrowExtensionArray
    from bodo.pandas.managers import LazyBlockManager, LazySingleBlockManager

T = pt.TypeVar(
    "T",
    bound=pt.Union[
        "LazySingleBlockManager",
        "LazyBlockManager",
        "LazySingleArrayManager",
        "LazyArrayManager",
        "LazyArrowExtensionArray",
    ],
)


class LazyMetadataMixin(pt.Generic[T]):
    """
    Superclass for lazy data structures with common metadata fields
    """

    __slots__ = ()
    # Number of rows in the result, this isn't part of the head so we need to store it separately
    _md_nrows: int | None
    # head of the result, which is used to determine the properties of the result e.g. columns/dtype
    _md_head: T | None
    # The result ID, used to fetch the result from the workers
    _md_result_id: str | None
    # Function to load the data this object represents.
    # Will be called with _md_result_id
    # Should also do any cleanup (after this is called _del_func won't be called)
    # Only callable once, will be set to None after called
    _collect_func: Callable[[str], pt.Any] | None
    # Function to clean up the data this object represents.
    # Will be called with _md_result_id
    # Should only be called if _collect_func hasn't been called
    # Only callable once, will be set to None after called
    _del_func: Callable[[str], None] | None


@dataclass
class LazyMetadata:
    result_id: str
    head: pd.DataFrame | pd.Series | ArrowExtensionArray
    nrows: int
    index_data: "LazyMetadata | None"
