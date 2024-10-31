import typing as pt

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
