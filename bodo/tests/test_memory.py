import pyarrow as pa
import pyarrow.lib
import pytest

from bodo.libs.memory import (
    BufferPool,
    BufferPoolAllocation,
    BufferPoolOptions,
    SizeClass,
    get_arrow_memory_pool_wrapper_for_buffer_pool,
    set_default_buffer_pool_as_arrow_memory_pool,
)
from bodo.tests.utils import temp_env_override


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
        assert options.ignore_max_limit_during_allocation

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
        assert options.memory_size == int(0.95 * total_mem)


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

    # Delete pool (to be conservative)
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

    # Verify stats
    assert pool.bytes_allocated() == 8 * 1024
    assert pool.max_memory() == 8 * 1024

    # Verify that the correct frame is allocated.
    assert size_class.is_in_range(alloc1)
    assert size_class.get_frame_index(alloc1) == 0
    alloc1_frame = size_class.get_alloc_for_frame(0)
    assert alloc1.has_same_ptr(alloc1_frame)

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
    alloc2_frame = size_class.get_alloc_for_frame(1)
    assert alloc2.has_same_ptr(alloc2_frame)

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

    # Delete pool (to be conservative)
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
    alloc1_frame = size_class_16.get_alloc_for_frame(0)
    assert alloc1.has_same_ptr(alloc1_frame)

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
    alloc2_frame = size_class_16.get_alloc_for_frame(1)
    assert alloc2.has_same_ptr(alloc2_frame)

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
    alloc3_frame = size_class_32.get_alloc_for_frame(0)
    assert alloc3.has_same_ptr(alloc3_frame)

    pool.free(alloc1)
    pool.free(alloc2)
    pool.free(alloc3)

    # No bytes should be allocated anymore
    assert pool.bytes_allocated() == 0

    # Max memory should still be the same
    assert pool.max_memory() == ((16 * 1024 * 2) + (32 * 1024))

    # Delete pool (to be conservative)
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

    # Verify stats
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024

    # Verify that the correct frame is allocated.
    assert size_class.is_in_range(alloc1)
    assert size_class.get_frame_index(alloc1) == 0
    alloc1_frame = size_class.get_alloc_for_frame(0)
    assert alloc1.has_same_ptr(alloc1_frame)

    # Verify that the correct bits are set
    assert size_class.is_frame_mapped(0)
    assert size_class.is_frame_pinned(0)

    pool.free(alloc1)

    # Max memory should still be the same.
    # Bytes allocated should now be 0
    assert pool.max_memory() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 0

    # Delete pool (to be conservative)
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

    # Delete pool (to be conservative)
    del pool


def test_larger_than_available_space_allocation():
    """
    Test that trying to allocate more memory than
    space available in the buffer pool raises the expected
    error.

    NOTE: This test will need to be modified once we have spill support.
    """
    # Allocate a very small pool for testing
    options = BufferPoolOptions(memory_size=4, min_size_class=4)
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 3.5 MiB
    allocation_1: BufferPoolAllocation = pool.allocate(1024 * 1024)
    allocation_2: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    allocation_3: BufferPoolAllocation = pool.allocate(512 * 1024)

    # Verify stats
    assert pool.bytes_allocated() == 3.5 * 1024 * 1024
    assert pool.max_memory() == 3.5 * 1024 * 1024

    # Trying to allocate 1MiB should fail
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match="Allocation failed. Not enough space in the buffer pool.",
    ):
        _: BufferPoolAllocation = pool.allocate(1024 * 1024)

    # Verify stats after failed allocation attempt
    assert pool.bytes_allocated() == 3.5 * 1024 * 1024
    assert pool.max_memory() == 3.5 * 1024 * 1024

    pool.free(allocation_1)
    pool.free(allocation_2)
    pool.free(allocation_3)

    # Delete pool (to be conservative)
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

    NOTE: This test might need to be modified once we have spill support.
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

    pool.free(allocation_1)
    pool.free(allocation_2)
    pool.free(allocation_3)
    pool.free(allocation_4)
    pool.free(allocation_5)
    pool.free(allocation_6)

    # Delete pool (to be conservative)
    del pool


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
    for i in range(14):
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

    # Delete pool (to be conservative)
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

    # Delete pool (to be conservative)
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

    # Delete pool (to be conservative)
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

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024
    assert allocation == orig_allocation_copy

    pool.free(allocation)

    # Delete pool (to be conservative)
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

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == (5 * 1024) + (16 * 1024)

    assert allocation != orig_allocation_copy

    pool.free(allocation)

    # Delete pool (to be conservative)
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

    # Delete pool (to be conservative)
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

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024

    # BufferPoolAllocation objects won't match exactly
    # since the size has changed (from a user perspective),
    # so we verify that the memory pointer is the same.
    assert allocation != orig_allocation_copy
    assert allocation.has_same_ptr(orig_allocation_copy)

    pool.free(allocation)

    # Delete pool (to be conservative)
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

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 2 * 16 * 1024

    assert allocation != orig_allocation_copy
    assert not allocation.has_same_ptr(orig_allocation_copy)

    pool.free(allocation)

    # Delete pool (to be conservative)
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

    # Delete pool (to be conservative)
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

    # Delete pool (to be conservative)
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

    # Delete pool (to be conservative)
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

    # Delete pool (to be conservative)
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

    assert pool.bytes_allocated() == 1024 * 1024
    assert pool.max_memory() == 1024 * 1024

    # Re-allocate to 12KiB
    pool.reallocate(12 * 1024, allocation)

    assert not size_class_1MiB.is_frame_mapped(0)
    assert not size_class_1MiB.is_frame_pinned(0)
    assert size_class_16KiB.is_frame_mapped(0)
    assert size_class_16KiB.is_frame_pinned(0)

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == (1024 * 1024) + (16 * 1024)

    assert allocation != orig_allocation_copy
    assert not allocation.has_same_ptr(orig_allocation_copy)

    pool.free(allocation)

    # Delete pool (to be conservative)
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
    assert not size_class_16KiB.is_frame_mapped(0)
    assert not size_class_16KiB.is_frame_pinned(0)

    assert pool.bytes_allocated() == 1024 * 1024
    assert pool.max_memory() == (1024 * 1024) + (16 * 1024)

    assert allocation != orig_allocation_copy
    assert not allocation.has_same_ptr(orig_allocation_copy)

    pool.free(allocation)

    # Delete pool (to be conservative)
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

        # Delete pool (to be conservative)
        del pool

    finally:
        # Restore default buffer pool as arrow memory pool
        set_default_buffer_pool_as_arrow_memory_pool()
