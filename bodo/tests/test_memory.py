import math
import mmap
from pathlib import Path

import pyarrow as pa
import pyarrow.lib
import pytest

import bodo
from bodo.libs.memory import (
    BufferPool,
    BufferPoolAllocation,
    BufferPoolOptions,
    SizeClass,
    StorageOptions,
    get_arrow_memory_pool_wrapper_for_buffer_pool,
    set_default_buffer_pool_as_arrow_memory_pool,
)
from bodo.tests.utils import temp_env_override

# Python doesn't always raise exceptions, particularly
# when raised inside of `del` statements and sometimes
# when not handled correctly in Cython before v3.
# PyTest can capture them as warnings, but to be extra
# safe, we can treat those warnings as exceptions
pytestmark = pytest.mark.filterwarnings(
    "error::pytest.PytestUnraisableExceptionWarning"
)


def test_default_buffer_pool_options():
    """
    Test that the default BufferPoolOptions instance
    has expected attributes.
    """

    # Unset all env vars to test the case where no
    # env vars are set.
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size > 0
        assert options.min_size_class == 64
        assert options.max_num_size_classes == 21
        assert not options.ignore_max_limit_during_allocation

    # Check that specifying the memory explicitly works as
    # expected.
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": "128",
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size == 128
        assert options.min_size_class == 64
        assert options.max_num_size_classes == 21

    # Check that specifying min_size_class and
    # max_num_size_classes through env vars works as
    # expected
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": "32",
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": "5",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size > 0
        assert options.min_size_class == 32
        assert options.max_num_size_classes == 5

    # Check that specifying ignore_max_limit_during_allocation
    # through env vars works as expected
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_IGNORE_MAX_ALLOCATION_LIMIT": "1",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size > 0
        assert options.ignore_max_limit_during_allocation

    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_IGNORE_MAX_ALLOCATION_LIMIT": "0",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size > 0
        assert not options.ignore_max_limit_during_allocation

    ## Test that BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT works as expected.

    # We will first get the full memory amount by setting it to 100.
    total_mem = 0
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": "100",
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
        }
    ):
        options = BufferPoolOptions.defaults()
        total_mem = options.memory_size

    # Now set it to 65 and verify that memory_size is 65% of total_mem.
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": "65",
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size == int(0.65 * total_mem)

    # Now check that the default is 95%
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
        }
    ):
        options = BufferPoolOptions.defaults()
        # This has been temporarily changed to 95% to 500% to
        # unblock issues in the latest release
        assert options.memory_size == int(5.0 * total_mem)


def test_default_storage_options(tmp_path: Path):
    """
    Test that the default StorageOptions instance
    has expected attributes.
    """

    # Check if no location is provided, nothing is returned
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": None,
        }
    ):
        assert StorageOptions.defaults(1) is None

    # Check that just specifying location is not enough as well
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": None,
        }
    ):
        assert StorageOptions.defaults(1) is None

    # Check that specifying the memory explicitly works as
    # expected.
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "4",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": "50",
        }
    ):
        options: StorageOptions = StorageOptions.defaults(1)
        assert options.location == bytes(tmp_path)
        assert options.usable_size == math.floor(
            (2 * 1024 * 1024 * 1024) / bodo.get_size()
        )

    # Check that default percentage is 0.9
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "4",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": None,
        }
    ):
        options: StorageOptions = StorageOptions.defaults(1)
        assert options.location == bytes(tmp_path)
        assert math.isclose(
            options.usable_size, (0.9 * (4 * 1024 * 1024 * 1024) / bodo.get_size())
        )

    # Check that distributing ranks across drives works
    # Note that this test only works when under 4 ranks
    if bodo.get_size() < 4:
        loc_a = tmp_path / "a"
        loc_b = tmp_path / "b"
        loc_c: Path = tmp_path / "c"
        locs = [str(loc_a), str(loc_b), str(loc_c)]
        with temp_env_override(
            {
                "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": ",".join(locs),
                "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "4",
                "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": "100",
            }
        ):
            options: StorageOptions = StorageOptions.defaults(1)
            assert options.location == bytes(locs[bodo.get_rank()], "utf-8")
            assert options.usable_size == (4 * 1024 * 1024 * 1024)

    # Check that not specifying a storage env vars doesn't
    # create a option in BufferPoolOptions
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": None,
        }
    ):
        options = BufferPoolOptions.defaults()
        assert len(options.storage_options) == 0

    # Check that not specifying a storage env vars doesn't
    # create a option in BufferPoolOptions
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "4",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_2_DRIVES": str(tmp_path / "inner"),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_2_SPACE_PER_DRIVE_GiB": "4",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_3_DRIVES": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_3_SPACE_PER_DRIVE_GiB": "4",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_4_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_4_SPACE_PER_DRIVE_GiB": "4",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert len(options.storage_options) == 2
        assert options.storage_options[0].location == bytes(tmp_path)
        assert options.storage_options[1].location == bytes(tmp_path / "inner")

    # Check that not specifying a storage env vars doesn't
    # create a option in BufferPoolOptions
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "4",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_2_DRIVES": str(tmp_path / "inner"),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_2_SPACE_PER_DRIVE_GiB": "4",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_3_DRIVES": str(tmp_path / "inner2"),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_3_SPACE_PER_DRIVE_GiB": "4",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_4_DRIVES": str(tmp_path / "inner3"),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_4_SPACE_PER_DRIVE_GiB": "4",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_5_DRIVES": str(tmp_path / "inner4"),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_5_SPACE_PER_DRIVE_GiB": "4",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert len(options.storage_options) == 4
        assert options.storage_options[0].location == bytes(tmp_path)
        assert options.storage_options[1].location == bytes(tmp_path / "inner")
        assert options.storage_options[2].location == bytes(tmp_path / "inner2")
        assert options.storage_options[3].location == bytes(tmp_path / "inner3")

    # Check that the disable spilling flag works and enables
    # ignore_max_limit_during_allocation
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "4",
            "BODO_BUFFER_POOL_DISABLE_SPILLING": "1",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert len(options.storage_options) == 0
        assert options.ignore_max_limit_during_allocation

    # Check that the disable spilling flag works and can override
    # ignore_max_limit_during_allocation
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "4",
            "BODO_BUFFER_POOL_DISABLE_SPILLING": "1",
            "BODO_BUFFER_POOL_IGNORE_MAX_ALLOCATION_LIMIT": "0",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert len(options.storage_options) == 0
        assert not options.ignore_max_limit_during_allocation


def test_malloc_allocation():
    """
    Test that small allocations that go through
    malloc work as expected.
    """
    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 5KiB which is <(3/4)*8KiB and therefore should
    # go through malloc
    allocation: BufferPoolAllocation = pool.allocate(5 * 1024)

    # Verify that default allocation is 64B
    assert allocation.alignment == 64
    assert allocation.get_ptr_as_int() % 64 == 0

    # Verify stats after allocation
    assert pool.bytes_allocated() == 5 * 1024
    assert pool.max_memory() == 5 * 1024

    # Verify that none of the bits are set in either bitmap
    # of any of the SizeClass-es and that the allocation
    # is not in the range.
    for size_class_idx in range(pool.num_size_classes()):
        size_class: SizeClass = pool.get_size_class(size_class_idx)
        assert not size_class.is_in_range(allocation)
        num_blocks = size_class.get_num_blocks()
        for frame_idx in range(num_blocks):
            assert not size_class.is_frame_mapped(
                frame_idx
            ), f"Frame at index {frame_idx} is mapped even though it shouldn't be!"
            assert not size_class.is_frame_pinned(
                frame_idx
            ), f"Frame at index {frame_idx} is pinned even though it shouldn't be!"

    # Free allocation
    pool.free(allocation)

    # Verify that no memory is allocated anymore
    assert pool.bytes_allocated() == 0
    # Max memory should still be the same
    assert pool.max_memory() == 5 * 1024

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_mmap_smallest_size_class_allocation():
    """
    Test that allocations through the smallest SizeClass
    work as expected.
    """
    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)

    size_class: SizeClass = pool.get_size_class(0)

    # Allocate 6KiB+1 (minimum amount to allocate through mmap)
    alloc1: BufferPoolAllocation = pool.allocate((6 * 1024) + 1)

    # Verify that default allocation is 64B
    assert alloc1.alignment == 64
    assert alloc1.get_ptr_as_int() % 64 == 0

    # Verify stats
    assert pool.bytes_allocated() == 8 * 1024
    assert pool.max_memory() == 8 * 1024

    # Verify that the correct frame is allocated.
    assert size_class.is_in_range(alloc1)
    assert size_class.get_frame_index(alloc1) == 0
    assert alloc1.has_same_ptr(size_class.build_alloc_for_frame(0))
    assert size_class.get_swip_at_frame(0) == alloc1.get_swip_as_int()

    # Verify that the correct bits are set
    assert size_class.is_frame_mapped(0)
    assert size_class.is_frame_pinned(0)
    for frame_idx in range(1, size_class.get_num_blocks()):
        assert not size_class.is_frame_mapped(frame_idx)
        assert not size_class.is_frame_pinned(frame_idx)

    # Allocate 8KiB
    alloc2: BufferPoolAllocation = pool.allocate(8 * 1024)

    # Verify stats
    assert pool.bytes_allocated() == 8 * 1024 * 2
    assert pool.max_memory() == 8 * 1024 * 2

    # Verify that the correct frame is allocated.
    assert size_class.is_in_range(alloc2)
    assert size_class.get_frame_index(alloc2) == 1
    assert alloc2.has_same_ptr(size_class.build_alloc_for_frame(1))
    assert size_class.get_swip_at_frame(1) == alloc2.get_swip_as_int()

    # Verify that the correct bits are set
    assert size_class.is_frame_mapped(0)
    assert size_class.is_frame_pinned(0)
    assert size_class.is_frame_mapped(1)
    assert size_class.is_frame_pinned(1)

    for frame_idx in range(2, size_class.get_num_blocks()):
        assert not size_class.is_frame_mapped(frame_idx)
        assert not size_class.is_frame_pinned(frame_idx)

    pool.free(alloc1)
    pool.free(alloc2)

    # No more bytes should be allocated
    assert pool.bytes_allocated() == 0

    # Max memory should still be the same
    assert pool.max_memory() == 8 * 1024 * 2

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_mmap_medium_size_classes_allocation():
    """
    Test that allocations through medium size SizeClass-es
    works as expected.
    """
    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 12KiB (in the 16KiB SizeClass)
    alloc1: BufferPoolAllocation = pool.allocate(12 * 1024)

    # Verify that default allocation is 64B
    assert alloc1.alignment == 64
    assert alloc1.get_ptr_as_int() % 64 == 0

    # Verify stats
    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024

    # Verify that the correct bits are set
    size_class_16: SizeClass = pool.get_size_class(1)
    assert size_class_16.is_frame_mapped(0)
    assert size_class_16.is_frame_pinned(0)

    # Verify that the correct frame is allocated.
    assert size_class_16.is_in_range(alloc1)
    assert size_class_16.get_frame_index(alloc1) == 0
    assert alloc1.has_same_ptr(size_class_16.build_alloc_for_frame(0))
    assert size_class_16.get_swip_at_frame(0) == alloc1.get_swip_as_int()

    # Allocate 16KiB
    alloc2: BufferPoolAllocation = pool.allocate(16 * 1024)

    # Verify stats
    assert pool.bytes_allocated() == 16 * 1024 * 2
    assert pool.max_memory() == 16 * 1024 * 2

    # Verify that the correct bits are set
    assert size_class_16.is_frame_mapped(0)
    assert size_class_16.is_frame_pinned(0)
    assert size_class_16.is_frame_mapped(1)
    assert size_class_16.is_frame_pinned(1)

    # Verify that the correct frame is allocated.
    assert size_class_16.is_in_range(alloc2)
    assert size_class_16.get_frame_index(alloc2) == 1
    assert alloc2.has_same_ptr(size_class_16.build_alloc_for_frame(1))
    assert size_class_16.get_swip_at_frame(1) == alloc2.get_swip_as_int()

    # Allocated 31KiB (in the 32KiB Size Class)
    alloc3: BufferPoolAllocation = pool.allocate(31 * 1024)

    # Verify stats
    assert pool.bytes_allocated() == ((16 * 1024 * 2) + (32 * 1024))
    assert pool.max_memory() == ((16 * 1024 * 2) + (32 * 1024))

    # Verify that the correct bits are set
    size_class_32: SizeClass = pool.get_size_class(2)
    assert size_class_32.is_frame_mapped(0)
    assert size_class_32.is_frame_pinned(0)

    # Verify that the correct frame is allocated.
    assert size_class_32.is_in_range(alloc3)
    assert size_class_32.get_frame_index(alloc3) == 0
    alloc3_frame = size_class_32.build_alloc_for_frame(0)
    assert alloc3.has_same_ptr(alloc3_frame)

    pool.free(alloc1)
    pool.free(alloc2)
    pool.free(alloc3)

    # No bytes should be allocated anymore
    assert pool.bytes_allocated() == 0

    # Max memory should still be the same
    assert pool.max_memory() == ((16 * 1024 * 2) + (32 * 1024))

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_mmap_largest_size_class_allocation():
    """
    Test that allocations through the largest SizeClass
    works as expected.
    """
    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)

    # The largest size-class (index 9) will have
    # capacity: 2 and block_size: 4194304 (4MiB)
    size_class: SizeClass = pool.get_size_class(9)

    # Allocate 3MiB (in the 4MiB SizeClass)
    alloc1: BufferPoolAllocation = pool.allocate(3 * 1024 * 1024)

    # Verify that default allocation is 64B
    assert alloc1.alignment == 64
    assert alloc1.get_ptr_as_int() % 64 == 0

    # Verify stats
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024

    # Verify that the correct frame is allocated.
    assert size_class.is_in_range(alloc1)
    assert size_class.get_frame_index(alloc1) == 0
    assert alloc1.has_same_ptr(size_class.build_alloc_for_frame(0))
    assert size_class.get_swip_at_frame(0) == alloc1.get_swip_as_int()

    # Verify that the correct bits are set
    assert size_class.is_frame_mapped(0)
    assert size_class.is_frame_pinned(0)

    pool.free(alloc1)

    # Max memory should still be the same.
    # Bytes allocated should now be 0
    assert pool.max_memory() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 0

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_larger_than_pool_allocation():
    """
    Test that trying to allocate more memory than
    the pool size (or larger than biggest SizeClass)
    raises the expected error.
    """
    # Allocate a very small pool for testing
    options = BufferPoolOptions(memory_size=4, min_size_class=4)
    pool: BufferPool = BufferPool.from_options(options)

    with pytest.raises(
        ValueError,
        match="Request allocation size is larger than the largest block-size available!",
    ):
        _: BufferPoolAllocation = pool.allocate((4 * 1024 * 1024) + 1)

    with pytest.raises(
        ValueError,
        match="Request allocation size is larger than the largest block-size available!",
    ):
        _: BufferPoolAllocation = pool.allocate(5 * 1024 * 1024)

    # Verify stats after allocation
    assert pool.bytes_allocated() == 0
    assert pool.max_memory() == 0

    # Verify that none of the bits are set in either bitmap
    # of any of the SizeClass-es
    for size_class_idx in range(pool.num_size_classes()):
        size_class: SizeClass = pool.get_size_class(size_class_idx)
        num_blocks = size_class.get_num_blocks()
        for frame_idx in range(num_blocks):
            assert not size_class.is_frame_mapped(
                frame_idx
            ), f"Frame at index {frame_idx} is mapped even though it shouldn't be!"
            assert not size_class.is_frame_pinned(
                frame_idx
            ), f"Frame at index {frame_idx} is pinned even though it shouldn't be!"

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_alignment():
    """
    Test that setting the alignment during allocations
    works as expected.
    """

    def size_align(size, alignment):
        remainder = size % alignment
        return size if (remainder == 0) else (size + alignment - remainder)

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)
    malloc_threshold = int(0.75 * 8 * 1024)

    # Test with malloc/mmap and small/medium/large alignments.
    for size, alignment in [
        (5 * 1024, 16),  # Should go through malloc
        (5 * 1024, 32),  # Should go through malloc
        (5 * 1024, 64),  # Should go through malloc
        (5 * 1024, 128),  # Should go through malloc
        (14 * 1024, 16),  # Should go through mmap
        (14 * 1024, 32),  # Should go through mmap
        (14 * 1024, 64),  # Should go through mmap
        (14 * 1024, 128),  # Should go through mmap
        (
            1024,
            4096,
        ),  # Should go through malloc
        (
            5 * 1024,
            4096,
        ),  # Would usually be malloc, but large alignment should force it to go through mmap
    ]:
        # Aligned size is what the BufferPool should actually try to allocate.
        aligned_size = size_align(size, alignment)
        # Collect state before allocation
        bytes_allocated_before = pool.bytes_allocated()
        allocation: BufferPoolAllocation = pool.allocate(size, alignment=alignment)
        bytes_allocated_after = pool.bytes_allocated()
        # Check that alignment matches up as expected
        assert allocation.alignment == alignment
        assert allocation.get_ptr_as_int() % alignment == 0
        # Check that actual allocation size is >= aligned_size.
        # In case of malloc, it should be exactly aligned_size.
        # In case of mmap frame, it might be higher.
        if aligned_size <= malloc_threshold:
            assert (bytes_allocated_after - bytes_allocated_before) == aligned_size
        else:
            assert (bytes_allocated_after - bytes_allocated_before) >= aligned_size
        pool.free(allocation)

    # Trying to allocate with alignment larger than page-size should error
    with pytest.raises(
        ValueError, match="Requested alignment higher than max supported alignment."
    ):
        page_size = mmap.PAGESIZE
        _: BufferPoolAllocation = pool.allocate(20 * 1024, page_size * 2)

    # Trying to allocate with non power of 2 alignment should error out
    with pytest.raises(
        ValueError, match="Alignment must be a positive number and a power of 2."
    ):
        # Malloc
        _: BufferPoolAllocation = pool.allocate(10, 48)

    with pytest.raises(
        ValueError, match="Alignment must be a positive number and a power of 2."
    ):
        # Mmap
        _: BufferPoolAllocation = pool.allocate(20 * 1024, 48)


def test_multiple_allocations():
    """
    Try to make multiple allocations in the same SizeClass
    and use all frames. We will then free a frame in the middle
    and try to allocate again and make sure that the BufferPool
    state is as expected at every stage.
    This is basically an E2E test.
    """
    # Allocate a very small pool for testing
    options = BufferPoolOptions(memory_size=7, min_size_class=4)
    pool: BufferPool = BufferPool.from_options(options)

    # Size Class at index 7 will have capacity: 14 and block_size: 524288 (0.5MiB)
    size_class: SizeClass = pool.get_size_class(7)
    assert size_class.get_block_size() == 524288
    assert size_class.get_num_blocks() == 14

    # Allocate all frames
    allocations = []
    for _ in range(14):
        allocation: BufferPoolAllocation = pool.allocate(524200)
        allocations.append(allocation)

    for frame_idx in range(14):
        assert size_class.is_frame_mapped(frame_idx)
        assert size_class.is_frame_pinned(frame_idx)

    # Free a random frame in the middle
    pool.free(allocations[9])

    assert not size_class.is_frame_mapped(9)
    assert not size_class.is_frame_pinned(9)

    for frame_idx in range(14):
        if frame_idx != 9:
            assert size_class.is_frame_mapped(frame_idx)
            assert size_class.is_frame_pinned(frame_idx)

    pool.free(allocations[10])

    # Allocate another frame and make sure the correct one
    # is selected.
    allocations[9] = pool.allocate(524200)

    # Verify bitmaps
    assert size_class.is_frame_mapped(9)
    assert size_class.is_frame_pinned(9)
    assert not size_class.is_frame_mapped(10)
    assert not size_class.is_frame_pinned(10)

    for frame_idx in range(14):
        if frame_idx != 10:
            assert size_class.is_frame_mapped(frame_idx)
            assert size_class.is_frame_pinned(frame_idx)
            pool.free(allocations[frame_idx])

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_verify_pool_attributes():
    """
    Verify that the pool attributes like number of size-classes,
    backend-name, etc. look correct.
    We also verify that the 'release_unused' function
    doesn't raise any exception (it should be a NOP).
    """

    # Verify that max_num_size_classes works as expected
    options = BufferPoolOptions(memory_size=7, min_size_class=4, max_num_size_classes=5)
    pool: BufferPool = BufferPool.from_options(options)
    assert pool.num_size_classes() == 5
    del pool

    options = BufferPoolOptions(
        memory_size=7, min_size_class=4, max_num_size_classes=15
    )
    pool: BufferPool = BufferPool.from_options(options)
    # We can only have a maximum of 11 classes given the memory-size
    # and minimum SizeClass.
    assert pool.num_size_classes() == 11

    # Check that the backend_name property works as expected
    assert pool.backend_name == "bodo"

    # Just run release_unused to make sure it doesn't raise any exceptions
    pool.release_unused()

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_reallocate_same_size_malloc():
    """
    Test the reallocate works as expected when trying
    to reallocate the same amount of memory
    through malloc (i.e. small allocation).
    Currently, we don't store information
    about malloc allocations such as the size at the
    BufferPool level, so we will always allocate new memory
    and do a memcpy.
    NOTE: The behavior may change in the future
    and this test may need to be modified.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 5KiB which is <(3/4)*8KiB and therefore should
    # go through malloc
    allocation: BufferPoolAllocation = pool.allocate(5 * 1024)
    orig_allocation_copy = BufferPoolAllocation.copy(allocation)

    assert pool.bytes_allocated() == 5 * 1024
    assert pool.max_memory() == 5 * 1024

    # Since we don't know the original
    # size at the BufferPool level, there will be a memcpy
    # and therefore max-memory will be twice.
    pool.reallocate(5 * 1024, allocation)

    assert pool.bytes_allocated() == 5 * 1024
    assert pool.max_memory() == 2 * 5 * 1024
    assert allocation != orig_allocation_copy

    pool.free(allocation)

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_reallocate_same_size_mmap():
    """
    Test the reallocate works as expected when trying
    to reallocate the same amount of memory
    through mmap.
    This should be a NOP essentially.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 10KiB
    allocation: BufferPoolAllocation = pool.allocate(10 * 1024)
    orig_allocation_copy = BufferPoolAllocation.copy(allocation)

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024

    pool.reallocate(10 * 1024, allocation)
    size_class: SizeClass = pool.get_size_class(1)

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024
    assert size_class.get_swip_at_frame(0) == allocation.get_swip_as_int()
    assert size_class.get_swip_at_frame(0) != orig_allocation_copy.get_swip_as_int()
    assert allocation == orig_allocation_copy

    pool.free(allocation)

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_reallocate_malloc_to_mmap():
    """
    Test the trying to reallocate from a small size (that would
    go through malloc) to a large size (that would go through a
    SizeClass) works as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 5KiB which is <(3/4)*8KiB and therefore should
    # go through malloc
    allocation: BufferPoolAllocation = pool.allocate(5 * 1024)
    orig_allocation_copy = BufferPoolAllocation.copy(allocation)

    assert pool.bytes_allocated() == 5 * 1024
    assert pool.max_memory() == 5 * 1024

    # Re-allocate to large enough to move to mmap
    # (will require a memcpy)
    pool.reallocate(10 * 1024, allocation)

    size_class: SizeClass = pool.get_size_class(1)
    assert size_class.is_frame_mapped(0)
    assert size_class.is_frame_pinned(0)
    assert size_class.get_swip_at_frame(0) == allocation.get_swip_as_int()
    assert size_class.get_swip_at_frame(0) != orig_allocation_copy.get_swip_as_int()

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == (5 * 1024) + (16 * 1024)

    assert allocation != orig_allocation_copy

    pool.free(allocation)

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_reallocate_mmap_to_malloc():
    """
    Test the trying to reallocate from a large size (that would
    go through a SizeClass) to a small size (that would go through
    malloc) works as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 10KiB through mmap
    allocation: BufferPoolAllocation = pool.allocate(10 * 1024)
    orig_allocation_copy = BufferPoolAllocation.copy(allocation)

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024

    # Re-allocate to small enough for malloc
    pool.reallocate(5 * 1024, allocation)

    size_class: SizeClass = pool.get_size_class(1)
    assert not size_class.is_frame_mapped(0)
    assert not size_class.is_frame_pinned(0)

    assert pool.bytes_allocated() == 5 * 1024
    assert pool.max_memory() == (5 * 1024) + (16 * 1024)

    assert allocation != orig_allocation_copy

    pool.free(allocation)

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_reallocate_larger_mem_same_size_class():
    """
    Test that trying to reallocate a larger amount of memory,
    but a size that would still be assigned to the same
    SizeClass works as expected. This should be a NOP.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)
    size_class: SizeClass = pool.get_size_class(1)

    # Allocate 10KiB through mmap
    allocation: BufferPoolAllocation = pool.allocate(10 * 1024)
    orig_allocation_copy = BufferPoolAllocation.copy(allocation)

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024

    assert size_class.is_frame_mapped(0)
    assert size_class.is_frame_pinned(0)

    for frame_idx in range(1, size_class.get_num_blocks()):
        assert not size_class.is_frame_mapped(frame_idx)
        assert not size_class.is_frame_pinned(frame_idx)

    # Re-allocate to 12KiB (same SizeClass)
    pool.reallocate(12 * 1024, allocation)

    assert size_class.is_frame_mapped(0)
    assert size_class.is_frame_pinned(0)

    for frame_idx in range(1, size_class.get_num_blocks()):
        assert not size_class.is_frame_mapped(frame_idx)
        assert not size_class.is_frame_pinned(frame_idx)

    assert size_class.get_swip_at_frame(0) == allocation.get_swip_as_int()
    assert size_class.get_swip_at_frame(0) != orig_allocation_copy.get_swip_as_int()

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024

    # BufferPoolAllocation objects won't match exactly
    # since the size has changed (from a user perspective),
    # so we verify that the memory pointer is the same.
    assert allocation != orig_allocation_copy
    assert allocation.has_same_ptr(orig_allocation_copy)

    pool.free(allocation)

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_reallocate_smaller_mem_same_size_class():
    """
    Test that trying to reallocate a smaller amount of memory,
    but a size that would still be assigned to the same
    SizeClass works as expected.
    Currently, this would allocate a separate frame
    and do a memcpy.
    NOTE: This behavior may change in the future.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)
    size_class: SizeClass = pool.get_size_class(1)

    # Allocate 12KiB through mmap
    allocation: BufferPoolAllocation = pool.allocate(12 * 1024)
    orig_allocation_copy = BufferPoolAllocation.copy(allocation)

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024

    assert size_class.is_frame_mapped(0)
    assert size_class.is_frame_pinned(0)

    for frame_idx in range(1, size_class.get_num_blocks()):
        assert not size_class.is_frame_mapped(frame_idx)
        assert not size_class.is_frame_pinned(frame_idx)

    # Re-allocate to 10KiB (same SizeClass)
    pool.reallocate(10 * 1024, allocation)

    assert not size_class.is_frame_mapped(0)
    assert not size_class.is_frame_pinned(0)
    assert size_class.is_frame_mapped(1)
    assert size_class.is_frame_pinned(1)

    for frame_idx in range(2, size_class.get_num_blocks()):
        assert not size_class.is_frame_mapped(frame_idx)
        assert not size_class.is_frame_pinned(frame_idx)

    assert size_class.get_swip_at_frame(1) == allocation.get_swip_as_int()
    assert size_class.get_swip_at_frame(1) != orig_allocation_copy.get_swip_as_int()

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 2 * 16 * 1024

    assert allocation != orig_allocation_copy
    assert not allocation.has_same_ptr(orig_allocation_copy)

    pool.free(allocation)

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_reallocate_0_to_mmap():
    """
    Test that trying to reallocate from 0B to
    a size that would go through mmap works
    as expected. This is important to test
    since we treat 0B allocations in a special
    way.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 0B
    allocation: BufferPoolAllocation = pool.allocate(0)
    orig_allocation_copy = BufferPoolAllocation.copy(allocation)

    # Verify that none of the bits are set in either bitmap
    # of any of the SizeClass-es
    for size_class_idx in range(pool.num_size_classes()):
        size_class: SizeClass = pool.get_size_class(size_class_idx)
        num_blocks = size_class.get_num_blocks()
        for frame_idx in range(num_blocks):
            assert not size_class.is_frame_mapped(
                frame_idx
            ), f"Frame at index {frame_idx} is mapped even though it shouldn't be!"
            assert not size_class.is_frame_pinned(
                frame_idx
            ), f"Frame at index {frame_idx} is pinned even though it shouldn't be!"

    assert pool.bytes_allocated() == 0
    assert pool.max_memory() == 0

    # Re-allocate to 10KiB (will go through mmap)
    pool.reallocate(10 * 1024, allocation)
    size_class: SizeClass = pool.get_size_class(1)

    assert size_class.is_frame_mapped(0)
    assert size_class.is_frame_pinned(0)

    for frame_idx in range(1, size_class.get_num_blocks()):
        assert not size_class.is_frame_mapped(frame_idx)
        assert not size_class.is_frame_pinned(frame_idx)

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024

    assert allocation != orig_allocation_copy
    assert not allocation.has_same_ptr(orig_allocation_copy)

    pool.free(allocation)

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_reallocate_0_to_malloc():
    """
    Test that trying to reallocate from 0B to
    a size that would go through malloc works
    as expected. This is important to test
    since we treat 0B allocations in a special
    way.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 0B
    allocation: BufferPoolAllocation = pool.allocate(0)
    orig_allocation_copy = BufferPoolAllocation.copy(allocation)

    # Verify that none of the bits are set in either bitmap
    # of any of the SizeClass-es
    for size_class_idx in range(pool.num_size_classes()):
        size_class: SizeClass = pool.get_size_class(size_class_idx)
        num_blocks = size_class.get_num_blocks()
        for frame_idx in range(num_blocks):
            assert not size_class.is_frame_mapped(
                frame_idx
            ), f"Frame at index {frame_idx} is mapped even though it shouldn't be!"
            assert not size_class.is_frame_pinned(
                frame_idx
            ), f"Frame at index {frame_idx} is pinned even though it shouldn't be!"

    assert pool.bytes_allocated() == 0
    assert pool.max_memory() == 0

    # Re-allocate to 5KiB (will go through mmap)
    pool.reallocate(5 * 1024, allocation)

    for size_class_idx in range(pool.num_size_classes()):
        size_class: SizeClass = pool.get_size_class(size_class_idx)
        num_blocks = size_class.get_num_blocks()
        for frame_idx in range(num_blocks):
            assert not size_class.is_frame_mapped(
                frame_idx
            ), f"Frame at index {frame_idx} is mapped even though it shouldn't be!"
            assert not size_class.is_frame_pinned(
                frame_idx
            ), f"Frame at index {frame_idx} is pinned even though it shouldn't be!"

    assert pool.bytes_allocated() == 5 * 1024
    assert pool.max_memory() == 5 * 1024

    assert allocation != orig_allocation_copy
    assert not allocation.has_same_ptr(orig_allocation_copy)

    pool.free(allocation)

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_reallocate_malloc_to_0():
    """
    Test that trying to reallocate to 0B from
    a size that would go through malloc works
    as expected. This is important to test
    since we treat 0B allocations in a special
    way.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 5KiB (will go through mmap)
    allocation: BufferPoolAllocation = pool.allocate(5 * 1024)
    orig_allocation_copy = BufferPoolAllocation.copy(allocation)

    assert pool.bytes_allocated() == 5 * 1024
    assert pool.max_memory() == 5 * 1024

    # Re-allocate to 0B
    pool.reallocate(0, allocation)

    assert pool.bytes_allocated() == 0
    assert pool.max_memory() == 5 * 1024

    assert allocation != orig_allocation_copy
    assert not allocation.has_same_ptr(orig_allocation_copy)

    pool.free(allocation)

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_reallocate_mmap_to_0():
    """
    Test that trying to reallocate to 0B from
    a size that would go through mmap works
    as expected. This is important to test
    since we treat 0B allocations in a special
    way.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)
    size_class: SizeClass = pool.get_size_class(1)

    # Allocate 12KiB (will go through mmap)
    allocation: BufferPoolAllocation = pool.allocate(12 * 1024)
    orig_allocation_copy = BufferPoolAllocation.copy(allocation)

    assert size_class.is_frame_mapped(0)
    assert size_class.is_frame_pinned(0)

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024

    # Re-allocate to 0B
    pool.reallocate(0, allocation)

    assert pool.bytes_allocated() == 0
    assert pool.max_memory() == 16 * 1024

    assert not size_class.is_frame_mapped(0)
    assert not size_class.is_frame_pinned(0)

    assert allocation != orig_allocation_copy
    assert not allocation.has_same_ptr(orig_allocation_copy)

    pool.free(allocation)

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_reallocate_smaller_mem_diff_size_class():
    """
    Test that trying to reallocate to a smaller
    SizeClass works as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)
    size_class_1MiB: SizeClass = pool.get_size_class(7)
    size_class_16KiB: SizeClass = pool.get_size_class(1)

    # Allocate 1MiB (will go through mmap)
    allocation: BufferPoolAllocation = pool.allocate(1024 * 1024)
    orig_allocation_copy = BufferPoolAllocation.copy(allocation)

    assert size_class_1MiB.is_frame_mapped(0)
    assert size_class_1MiB.is_frame_pinned(0)
    assert not size_class_16KiB.is_frame_mapped(0)
    assert not size_class_16KiB.is_frame_pinned(0)
    assert size_class_1MiB.get_swip_at_frame(0) == allocation.get_swip_as_int()

    assert pool.bytes_allocated() == 1024 * 1024
    assert pool.max_memory() == 1024 * 1024

    # Re-allocate to 12KiB
    pool.reallocate(12 * 1024, allocation)

    assert not size_class_1MiB.is_frame_mapped(0)
    assert not size_class_1MiB.is_frame_pinned(0)
    assert size_class_16KiB.is_frame_mapped(0)
    assert size_class_16KiB.is_frame_pinned(0)
    assert size_class_16KiB.get_swip_at_frame(0) == allocation.get_swip_as_int()
    assert (
        size_class_16KiB.get_swip_at_frame(0) != orig_allocation_copy.get_swip_as_int()
    )

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == (1024 * 1024) + (16 * 1024)

    assert allocation != orig_allocation_copy
    assert not allocation.has_same_ptr(orig_allocation_copy)

    pool.free(allocation)

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_reallocate_larger_mem_diff_size_class():
    """
    Test that trying to reallocate to a larger
    SizeClass works as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8, min_size_class=8, max_num_size_classes=10
    )
    pool: BufferPool = BufferPool.from_options(options)
    size_class_1MiB: SizeClass = pool.get_size_class(7)
    size_class_16KiB: SizeClass = pool.get_size_class(1)

    # Allocate 12KiB
    allocation: BufferPoolAllocation = pool.allocate(12 * 1024)
    orig_allocation_copy = BufferPoolAllocation.copy(allocation)

    assert not size_class_1MiB.is_frame_mapped(0)
    assert not size_class_1MiB.is_frame_pinned(0)
    assert size_class_16KiB.is_frame_mapped(0)
    assert size_class_16KiB.is_frame_pinned(0)

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024

    # Re-allocate to 1MiB
    pool.reallocate(1024 * 1024, allocation)

    assert size_class_1MiB.is_frame_mapped(0)
    assert size_class_1MiB.is_frame_pinned(0)
    assert size_class_1MiB.get_swip_at_frame(0) == allocation.get_swip_as_int()
    assert not size_class_16KiB.is_frame_mapped(0)
    assert not size_class_16KiB.is_frame_pinned(0)
    assert size_class_16KiB.get_swip_at_frame(0) is None

    assert pool.bytes_allocated() == 1024 * 1024
    assert pool.max_memory() == (1024 * 1024) + (16 * 1024)

    assert allocation != orig_allocation_copy
    assert not allocation.has_same_ptr(orig_allocation_copy)

    pool.free(allocation)

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_pyarrow_allocation():
    """
    Test that setting our BufferPool as the PyArrow
    MemoryPool works by verifying that subsequent
    allocations go through the BufferPool as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(memory_size=4, min_size_class=4)
    pool: BufferPool = BufferPool.from_options(options)
    arrow_mem_pool_wrapper = get_arrow_memory_pool_wrapper_for_buffer_pool(pool)
    pa.set_memory_pool(arrow_mem_pool_wrapper)

    try:
        # This will try to allocate 16*400 = 6400B, i.e. frame in 8KiB SizeClass.
        # It will also try to allocate a null bitmap (100B, allocates 128B for alignment),
        # but free it immediately.
        arr1 = pa.array(
            [4, 5] * (400),
            type=pa.int64(),
        )

        assert pool.bytes_allocated() == 8 * 1024
        assert pool.max_memory() == ((8 * 1024) + 128)

        size_class_8KiB: SizeClass = pool.get_size_class(1)
        assert size_class_8KiB.is_frame_mapped(0)
        assert size_class_8KiB.is_frame_pinned(0)

        # This should free the space
        del arr1

        assert pool.bytes_allocated() == 0
        assert pool.max_memory() == ((8 * 1024) + 128)
        assert not size_class_8KiB.is_frame_mapped(0)
        assert not size_class_8KiB.is_frame_pinned(0)

        # This will allocate 24*400 = 9600B, i.e. frame in 16KiB SizeClass.
        # It will also allocate a null-bitmap with 1200bits i.e. 150B (192B after alignment, will go through malloc).
        arr2 = pa.array([4, 5, None] * (400), type=pa.int64())

        assert pool.bytes_allocated() == (16 * 1024) + 192
        assert pool.max_memory() == (16 * 1024) + 192

        size_class_16KiB: SizeClass = pool.get_size_class(2)
        assert size_class_16KiB.is_frame_mapped(0)
        assert size_class_16KiB.is_frame_pinned(0)

        # This should free the space
        del arr2

        assert pool.bytes_allocated() == 0
        assert pool.max_memory() == (16 * 1024) + 192
        assert not size_class_16KiB.is_frame_mapped(0)
        assert not size_class_16KiB.is_frame_pinned(0)

        # Cleanup and delete pool (to be conservative)
        pool.cleanup()
        del pool

    finally:
        # Restore default buffer pool as arrow memory pool
        set_default_buffer_pool_as_arrow_memory_pool()


def test_pin_unpin():
    """
    Test that the Buffer Pool tracks pinned
    and unpinned frames correctly
    """
    # Allocate a very small pool for testing
    options = BufferPoolOptions(memory_size=4, min_size_class=1024)
    pool: BufferPool = BufferPool.from_options(options)
    size_class_1MiB: SizeClass = pool.get_size_class(0)

    # Create a single allocation
    allocation = pool.allocate(1000 * 1024)

    # Verify stats
    assert pool.bytes_pinned() == 1024 * 1024
    assert pool.bytes_allocated() == 1024 * 1024
    assert pool.max_memory() == 1024 * 1024
    assert size_class_1MiB.is_frame_pinned(0)

    # Unpin and check stats again
    pool.unpin(allocation)
    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 1024 * 1024
    assert pool.max_memory() == 1024 * 1024
    assert not size_class_1MiB.is_frame_pinned(0)

    # Pin and check stats again
    pool.pin(allocation)
    assert pool.bytes_pinned() == 1024 * 1024
    assert pool.bytes_allocated() == 1024 * 1024
    assert pool.max_memory() == 1024 * 1024
    assert size_class_1MiB.is_frame_pinned(0)

    # Cleanup and delete pool (to be conservative)
    pool.free(allocation)
    pool.cleanup()
    del pool


def test_oom_all_mem_pinned():
    """
    Test that trying to allocate more memory than
    space available raises an OOM error when all
    memory is pinned.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(memory_size=4, min_size_class=1024)
    pool: BufferPool = BufferPool.from_options(options)
    size_class_1MiB: SizeClass = pool.get_size_class(0)
    size_class_2MiB: SizeClass = pool.get_size_class(1)

    # Allocate 3.5 MiB
    allocation_1: BufferPoolAllocation = pool.allocate(1024 * 1024)
    allocation_2: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    allocation_3: BufferPoolAllocation = pool.allocate(512 * 1024)

    # Verify stats
    assert pool.bytes_allocated() == 3.5 * 1024 * 1024
    assert pool.max_memory() == 3.5 * 1024 * 1024
    assert size_class_1MiB.is_frame_pinned(0)
    assert size_class_2MiB.is_frame_pinned(0)

    # Trying to allocate 1MiB should fail
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match="Allocation failed. Not enough space in the buffer pool.",
    ):
        _: BufferPoolAllocation = pool.allocate(600 * 1024)

    # Verify stats after failed allocation attempt
    assert pool.bytes_allocated() == 3.5 * 1024 * 1024
    assert pool.max_memory() == 3.5 * 1024 * 1024

    pool.free(allocation_1)
    pool.free(allocation_2)
    pool.free(allocation_3)

    assert pool.bytes_allocated() == 0

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_oom_some_mem_unpinned():
    """
    Test that trying to allocate more memory than
    space available raises an OOM error, even when
    some memory is unpinned (but not enough space still
    and no storage managers provided).
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(memory_size=4, min_size_class=1024)
    pool: BufferPool = BufferPool.from_options(options)
    size_class_1MiB: SizeClass = pool.get_size_class(0)
    size_class_2MiB: SizeClass = pool.get_size_class(1)

    # Allocate 3.5 MiB
    allocation_1: BufferPoolAllocation = pool.allocate(1024 * 1024)
    allocation_2: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    allocation_3: BufferPoolAllocation = pool.allocate(512 * 1024)

    # Verify stats
    assert pool.bytes_pinned() == 3.5 * 1024 * 1024
    assert pool.bytes_allocated() == 3.5 * 1024 * 1024
    assert pool.max_memory() == 3.5 * 1024 * 1024
    assert size_class_1MiB.is_frame_pinned(0)
    assert size_class_2MiB.is_frame_pinned(0)

    # Unpinning 2MiB
    pool.unpin(allocation_1)

    # Trying to allocate 2MiB should fail
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match="Allocation failed. Not enough space in the buffer pool.",
    ):
        _: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)

    # Verify stats after failed allocation attempt
    assert pool.bytes_pinned() == 2.5 * 1024 * 1024
    assert pool.bytes_allocated() == 3.5 * 1024 * 1024
    assert pool.max_memory() == 3.5 * 1024 * 1024
    assert not size_class_1MiB.is_frame_pinned(0)
    assert size_class_2MiB.is_frame_pinned(0)

    pool.free(allocation_1)
    pool.free(allocation_2)
    pool.free(allocation_3)

    assert pool.bytes_allocated() == 0

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_oom_no_storage():
    """
    Test that trying to allocate more memory than
    space available raises an OOM error when no
    storage managers are provided.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(memory_size=4, min_size_class=1024)
    pool: BufferPool = BufferPool.from_options(options)
    size_class_2MiB: SizeClass = pool.get_size_class(1)

    # Allocate and Unpin 4 MiB
    allocation_1: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    allocation_2: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    pool.unpin(allocation_1)
    pool.unpin(allocation_2)

    # Verify stats
    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert not size_class_2MiB.is_frame_pinned(0)
    assert not size_class_2MiB.is_frame_pinned(1)

    # Trying to allocate 2MiB should fail
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match="Allocation failed. No storage locations provided to evict to.",
    ):
        _: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)

    # Verify stats after failed allocation attempt
    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert not size_class_2MiB.is_frame_pinned(0)
    assert not size_class_2MiB.is_frame_pinned(1)

    pool.free(allocation_1)
    pool.free(allocation_2)

    assert pool.bytes_allocated() == 0
    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_larger_than_available_space_allocation_limit_ignored():
    """
    Test that trying to allocate more memory than
    space available in the buffer pool doesn't raise
    an error when ignore_max_limit_during_allocation is True
    and there actually is enough space in physical memory (and
    a buffer frame of appropriate size is available in the pool).
    Also verifies that if all frames are taken up, the appropriate
    error is raised even if ignore_max_limit_during_allocation is
    True.
    """
    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=4, min_size_class=4, ignore_max_limit_during_allocation=True
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 3.5 MiB
    allocation_1: BufferPoolAllocation = pool.allocate(1024 * 1024)
    allocation_2: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    allocation_3: BufferPoolAllocation = pool.allocate(512 * 1024)

    # Verify stats
    assert pool.bytes_allocated() == 3.5 * 1024 * 1024
    assert pool.max_memory() == 3.5 * 1024 * 1024

    # This allocation should now go through since there should be
    # a free 1MiB frame available.
    allocation_4: BufferPoolAllocation = pool.allocate(1024 * 1024)

    # Verify stats after allocation attempt
    assert pool.bytes_allocated() == 4.5 * 1024 * 1024
    assert pool.max_memory() == 4.5 * 1024 * 1024

    # Try to fill up all 1MiB frames:
    allocation_5: BufferPoolAllocation = pool.allocate(1024 * 1024)
    allocation_6: BufferPoolAllocation = pool.allocate(1024 * 1024)

    # Try allocating again:
    # Trying to allocate 1MiB should fail
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match="Could not find an empty frame of required size!",
    ):
        _: BufferPoolAllocation = pool.allocate(1024 * 1024)

    # Verify stats after allocation attempt
    assert pool.bytes_allocated() == 6.5 * 1024 * 1024
    assert pool.max_memory() == 6.5 * 1024 * 1024

    # Unpinning and pinning should continue working
    pool.unpin(allocation_6)
    pool.pin(allocation_6)

    # Unpinning and a new allocation should be fine
    pool.unpin(allocation_6)
    allocation_7: BufferPoolAllocation = pool.allocate(512 * 1024)
    pool.pin(allocation_6)

    pool.free(allocation_1)
    pool.free(allocation_2)
    pool.free(allocation_3)
    pool.free(allocation_4)
    pool.free(allocation_5)
    pool.free(allocation_6)
    pool.free(allocation_7)

    # Delete pool (to be conservative)
    del pool


def test_allocate_spill_eq_block(tmp_path: Path):
    """
    Test that trying to allocate more memory than
    space available leads to spilling an equal-size frame
    """

    # Allocate a very small pool for testing
    local_opt = StorageOptions(usable_size=5 * 1024 * 1024, location=bytes(tmp_path))
    options = BufferPoolOptions(
        memory_size=4, min_size_class=1024, storage_options=[local_opt]
    )
    pool: BufferPool = BufferPool.from_options(options)
    size_class_1MiB: SizeClass = pool.get_size_class(0)
    size_class_2MiB: SizeClass = pool.get_size_class(1)

    # Allocate 3.5 MiB
    allocation_1: BufferPoolAllocation = pool.allocate(1024 * 1024)
    allocation_2: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    allocation_3: BufferPoolAllocation = pool.allocate(512 * 1024)

    # Verify stats
    assert pool.bytes_pinned() == 3.5 * 1024 * 1024
    assert pool.bytes_allocated() == 3.5 * 1024 * 1024
    assert pool.max_memory() == 3.5 * 1024 * 1024
    assert size_class_1MiB.is_frame_pinned(0)
    assert size_class_2MiB.is_frame_pinned(0)

    # Unpinning 2MiB
    pool.unpin(allocation_2)
    assert not size_class_2MiB.is_frame_pinned(0)
    assert not allocation_2.is_nullptr() and allocation_2.is_in_memory()

    # Trying to allocate 2MiB should cause allocation_2 to spill
    allocation_4: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)

    # Verify stats after allocation + spill
    assert pool.bytes_pinned() == 3.5 * 1024 * 1024
    assert pool.bytes_allocated() == 3.5 * 1024 * 1024
    assert pool.max_memory() == 3.5 * 1024 * 1024
    assert size_class_1MiB.is_frame_pinned(0)
    assert size_class_2MiB.is_frame_pinned(0)  # allocation_4
    assert not allocation_2.is_nullptr() and not allocation_2.is_in_memory()

    pool.free(allocation_1)
    pool.free(allocation_2)
    pool.free(allocation_3)
    pool.free(allocation_4)

    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_allocate_spill_smaller_blocks(tmp_path: Path):
    """
    Test that trying to allocate more memory than
    space available leads to spilling multiple smaller-size
    frames
    """

    # Allocate a very small pool for testing
    local_opt = StorageOptions(usable_size=5 * 1024 * 1024, location=bytes(tmp_path))
    options = BufferPoolOptions(
        memory_size=4, min_size_class=1024, storage_options=[local_opt]
    )
    pool: BufferPool = BufferPool.from_options(options)
    size_class_1MiB: SizeClass = pool.get_size_class(0)
    size_class_2MiB: SizeClass = pool.get_size_class(1)

    # Allocate 3.5 MiB
    allocation_1: BufferPoolAllocation = pool.allocate(1024 * 1024)
    allocation_2: BufferPoolAllocation = pool.allocate(1024 * 1024)
    allocation_3: BufferPoolAllocation = pool.allocate(1024 * 1024)
    allocation_4: BufferPoolAllocation = pool.allocate(512 * 1024)

    # Verify stats
    assert pool.bytes_pinned() == 3.5 * 1024 * 1024
    assert pool.bytes_allocated() == 3.5 * 1024 * 1024
    assert pool.max_memory() == 3.5 * 1024 * 1024
    assert size_class_1MiB.is_frame_pinned(0)
    assert size_class_1MiB.is_frame_pinned(1)
    assert size_class_1MiB.is_frame_pinned(2)

    # Unpinning 2MiB
    pool.unpin(allocation_2)
    pool.unpin(allocation_3)
    assert not size_class_1MiB.is_frame_pinned(1)
    assert not size_class_1MiB.is_frame_pinned(2)
    assert not allocation_2.is_nullptr() and allocation_2.is_in_memory()
    assert not allocation_3.is_nullptr() and allocation_3.is_in_memory()

    # Trying to allocate 2MiB should cause allocation_2 and allocation_3 to spill
    allocation_5: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)

    # Verify stats after allocation + spill
    assert pool.bytes_pinned() == 3.5 * 1024 * 1024
    assert pool.bytes_allocated() == 3.5 * 1024 * 1024
    assert pool.max_memory() == 3.5 * 1024 * 1024
    assert size_class_1MiB.is_frame_pinned(0)
    assert size_class_2MiB.is_frame_pinned(0)
    assert not allocation_2.is_nullptr() and not allocation_2.is_in_memory()
    assert not allocation_3.is_nullptr() and not allocation_3.is_in_memory()

    pool.free(allocation_1)
    pool.free(allocation_2)
    pool.free(allocation_3)
    pool.free(allocation_4)
    pool.free(allocation_5)

    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_allocate_spill_larger_block(tmp_path: Path):
    """
    Test that trying to allocate more memory than
    space available leads to spilling an larger-size frame
    """

    # Allocate a very small pool for testing
    local_opt = StorageOptions(usable_size=5 * 1024 * 1024, location=bytes(tmp_path))
    options = BufferPoolOptions(
        memory_size=4, min_size_class=1024, storage_options=[local_opt]
    )
    pool: BufferPool = BufferPool.from_options(options)
    size_class_1MiB: SizeClass = pool.get_size_class(0)
    size_class_2MiB: SizeClass = pool.get_size_class(1)

    # Allocate 3 MiB
    allocation_1: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    allocation_2: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)

    # Verify stats
    assert pool.bytes_pinned() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert size_class_2MiB.is_frame_pinned(0)
    assert size_class_2MiB.is_frame_pinned(1)

    # Unpinning 2MiB
    pool.unpin(allocation_1)
    assert not size_class_2MiB.is_frame_pinned(0)
    assert not allocation_1.is_nullptr() and allocation_1.is_in_memory()
    assert pool.bytes_pinned() == 2 * 1024 * 1024

    # Trying to allocate 1MiB should cause allocation_1 to spill
    allocation_3: BufferPoolAllocation = pool.allocate(1024 * 1024)

    # Verify stats after allocation + spill
    assert pool.bytes_pinned() == 3 * 1024 * 1024
    assert pool.bytes_allocated() == 3 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert size_class_2MiB.is_frame_pinned(1)  # allocation_2
    assert size_class_1MiB.is_frame_pinned(0)  # allocation_3
    assert not allocation_1.is_nullptr() and not allocation_1.is_in_memory()

    pool.free(allocation_1)
    pool.free(allocation_2)
    pool.free(allocation_3)

    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_repin_eviction(tmp_path: Path):
    """
    Test that unpinning, filling up the pool,
    and repinning leads to an eviction if possible
    """

    # Allocate a very small pool for testing
    local_opt = StorageOptions(usable_size=5 * 1024 * 1024, location=bytes(tmp_path))
    options = BufferPoolOptions(
        memory_size=4, min_size_class=1024, storage_options=[local_opt]
    )
    pool: BufferPool = BufferPool.from_options(options)
    size_class_1MiB: SizeClass = pool.get_size_class(0)
    size_class_2MiB: SizeClass = pool.get_size_class(1)

    # Allocate 3 MiB
    allocation_1: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    allocation_2: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)

    # Verify stats
    assert pool.bytes_pinned() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert size_class_2MiB.is_frame_pinned(0)
    assert size_class_2MiB.is_frame_pinned(1)

    # Unpinning 1MiB
    pool.unpin(allocation_1)
    assert not size_class_2MiB.is_frame_pinned(0)
    assert not allocation_1.is_nullptr() and allocation_1.is_in_memory()
    assert pool.bytes_pinned() == 2 * 1024 * 1024

    # Trying to allocate 1MiB should cause allocation_1 to spill
    allocation_3: BufferPoolAllocation = pool.allocate(1024 * 1024)

    # Verify stats after allocation + spill
    assert pool.bytes_pinned() == 3 * 1024 * 1024
    assert pool.bytes_allocated() == 3 * 1024 * 1024
    assert size_class_2MiB.is_frame_pinned(1)  # allocation_2
    assert size_class_1MiB.is_frame_pinned(0)  # allocation_3
    assert not allocation_1.is_nullptr() and not allocation_1.is_in_memory()

    # Repinning now should fail
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match="Pin failed. Not enough space in the buffer pool.",
    ):
        pool.pin(allocation_1)

    # Repin allocation_1 after unpinning allocation_3
    pool.unpin(allocation_3)
    pool.pin(allocation_1)

    # Verify stats after repin
    assert pool.bytes_pinned() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert size_class_2MiB.is_frame_pinned(0)  # allocation_1
    assert size_class_2MiB.is_frame_pinned(1)  # allocation_2
    assert not size_class_1MiB.is_frame_pinned(0)  # allocation_3
    assert not allocation_3.is_nullptr() and not allocation_3.is_in_memory()

    pool.free(allocation_1)
    pool.free(allocation_2)
    pool.free(allocation_3)

    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_eviction_readback_contents(tmp_path: Path):
    """
    Test that the contents of a block
    after being evicted and read-back into memory
    are still the same
    """

    # Allocate a very small pool for testing
    local_opt = StorageOptions(usable_size=5 * 1024 * 1024, location=bytes(tmp_path))
    options = BufferPoolOptions(
        memory_size=4, min_size_class=1024, storage_options=[local_opt]
    )
    pool: BufferPool = BufferPool.from_options(options)
    size_class_4MiB: SizeClass = pool.get_size_class(2)

    # Allocate 4 MiB frame and write contents
    write_bytes = b"Hello BufferPool from Bodo!"
    allocation_1: BufferPoolAllocation = pool.allocate(3.5 * 1024 * 1024)
    allocation_1.write_bytes(write_bytes)

    # Verify stats
    assert pool.bytes_pinned() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert size_class_4MiB.is_frame_pinned(0)

    # Unpin and evict frame
    pool.unpin(allocation_1)
    allocation_2: BufferPoolAllocation = pool.allocate(1024 * 1024)

    # Verify stats after allocation + spill
    assert pool.bytes_pinned() == 1 * 1024 * 1024
    assert pool.bytes_allocated() == 1 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert not allocation_1.is_nullptr() and not allocation_1.is_in_memory()

    # Repin allocation_1 and read contents
    pool.unpin(allocation_2)
    pool.pin(allocation_1)
    read_bytes: bytes = allocation_1.read_bytes(len(write_bytes))

    # Verify stats after repin
    assert pool.bytes_pinned() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert size_class_4MiB.is_frame_pinned(0)

    # Compare contents
    assert write_bytes == read_bytes

    pool.free(allocation_1)
    pool.free(allocation_2)

    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0

    # Cleanup and delete pool (to be conservative)
    pool.cleanup()
    del pool


def test_local_spill_file(tmp_path: Path):
    """
    Test if the location and contents of the local spill files
    are correct
    """

    # Allocate a very small pool for testing
    local_opt = StorageOptions(usable_size=4 * 1024 * 1024, location=bytes(tmp_path))
    options = BufferPoolOptions(
        memory_size=4, min_size_class=1024, storage_options=[local_opt]
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 4 MiB frame, write contents, and evict
    write_bytes = b"Hello BufferPool from Bodo!"
    allocation_1: BufferPoolAllocation = pool.allocate(3.5 * 1024 * 1024)
    allocation_1.write_bytes(write_bytes)
    pool.unpin(allocation_1)
    allocation_2: BufferPoolAllocation = pool.allocate(1024 * 1024)

    # Verify stats
    assert pool.bytes_pinned() == 1 * 1024 * 1024
    assert pool.bytes_allocated() == 1 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert not allocation_1.is_nullptr() and not allocation_1.is_in_memory()

    # Look for spilled file in expected location
    # First Folder Expected to Have Prepended Rank
    paths = list(tmp_path.glob(f"{bodo.get_rank()}-*"))
    assert len(paths) == 1
    inner_path = paths[0]
    # Block File should be in {size_class}/{block_id}
    block_file = inner_path / str(4 * 1024 * 1024) / "0"
    assert block_file.is_file()

    # Check file contents
    expected_contents = (
        "Hello BufferPool from Bodo!" + (4 * 1024 * 1024 - len(write_bytes)) * "\x00"
    )
    assert block_file.read_text() == expected_contents

    pool.free(allocation_1)
    pool.free(allocation_2)
    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0
    # Make sure free deletes any spill files as well
    assert len(list(elem for elem in tmp_path.iterdir() if elem.is_file())) == 0

    # Cleanup
    pool.cleanup()
    assert len(list(tmp_path.iterdir())) == 0

    # Delete pool (to be conservative)
    del pool
