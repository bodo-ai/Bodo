import math
import mmap
import re
import sys
from pathlib import Path
from uuid import uuid4

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.lib
import pytest
import s3fs

import bodo
from bodo.memory import (
    default_buffer_pool_bytes_allocated,
    default_buffer_pool_bytes_pinned,
    set_default_buffer_pool_as_arrow_memory_pool,
)
from bodo.tests.memory_tester import (
    BufferPool,
    BufferPoolAllocation,
    BufferPoolOptions,
    OperatorBufferPool,
    OperatorScratchPool,
    SizeClass,
    StorageOptions,
    get_arrow_memory_pool_wrapper_for_buffer_pool,
)
from bodo.tests.utils import temp_env_override
from bodo.utils.typing import BodoWarning

# Python doesn't always raise exceptions, particularly
# when raised inside of `del` statements and sometimes
# when not handled correctly in Cython before v3.
# PyTest can capture them as warnings, but to be extra
# safe, we can treat those warnings as exceptions
pytestmark = [
    pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning"),
    # TODO[BSE-4556]: enable when bufferpool is enabled on Windows
    pytest.mark.skipif(
        sys.platform == "win32", reason="bufferpool disabled on Windows"
    ),
]


@pytest.fixture
def tmp_s3_path():
    """
    Create a temporary S3 path for testing.
    """

    s3 = boto3.resource("s3")
    bucket = s3.Bucket("engine-unit-tests-tmp-bucket")
    folder_name = str(uuid4())
    yield f"s3://engine-unit-tests-tmp-bucket/{folder_name}/"
    bucket.objects.filter(Prefix=folder_name).delete()


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
            "BODO_BUFFER_POOL_ENFORCE_MAX_ALLOCATION_LIMIT": None,
            "BODO_BUFFER_POOL_DEBUG_MODE": None,
            "BODO_BUFFER_POOL_MALLOC_FREE_TRIM_THRESHOLD_MiB": None,
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size > 0
        assert options.min_size_class == 64
        assert options.max_num_size_classes == 23
        assert not options.enforce_max_limit_during_allocation
        assert not options.debug_mode
        assert options.malloc_free_trim_threshold == 100 * 1024 * 1024

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
        assert options.max_num_size_classes == 23

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

    # Check that specifying enforce_max_limit_during_allocation
    # through env vars works as expected
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_ENFORCE_MAX_ALLOCATION_LIMIT": "1",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size > 0
        assert options.enforce_max_limit_during_allocation

    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_ENFORCE_MAX_ALLOCATION_LIMIT": "0",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size > 0
        assert not options.enforce_max_limit_during_allocation

    # Check that specifying malloc_free_trim_threshold
    # through env vars works as expected
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_MALLOC_FREE_TRIM_THRESHOLD_MiB": "20",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size > 0
        assert options.malloc_free_trim_threshold == 20 * 1024 * 1024


def test_default_memory_options(tmp_path: Path):
    """
    Test that BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT works as expected.
    """

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
        assert options.min_size_class == 16

    # Now check that the default is 95% when spilling is available.
    # Also check that the default min size class is 16KiB.
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": "100",
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size == int(0.95 * total_mem)
        assert len(options.storage_options) == 1
        assert options.min_size_class == 16

    # Now check that the default is 500% when spilling is not available.
    # Also check that the default min size class is 64KiB.
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": None,
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size == int(5.0 * total_mem)
        assert len(options.storage_options) == 0
        assert options.min_size_class == 64

    # Now check that the default is 500% when spilling is disabled
    # Also check that disabling spilling through env vars works as expected
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": "100",
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_DISABLE_SPILLING": "1",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size == int(5.0 * total_mem)
        assert len(options.storage_options) == 0
        assert not BufferPool.from_options(options).is_spilling_enabled()
        assert options.min_size_class == 64

    # Check that not disabling spilling through env vars works as expected
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": "100",
            "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
            "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": None,
            "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
            "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            "BODO_BUFFER_POOL_DISABLE_SPILLING": "0",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.memory_size == int(0.95 * total_mem)
        assert len(options.storage_options) == 1
        assert BufferPool.from_options(options).is_spilling_enabled()
        assert options.min_size_class == 16

    # Verify that it raises an error when spill-on-unpin is set but
    # no spilling locations are provided.
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_SPILL_ON_UNPIN": "1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": None,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": None,
        }
    ):
        with pytest.raises(
            RuntimeError,
            match="Must specify at least one storage location when setting spill_on_unpin",
        ):
            options = BufferPoolOptions.defaults()

    # Verify that setting spill-on-unpin works as expected.
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_SPILL_ON_UNPIN": "1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": "100",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert options.spill_on_unpin
        assert not options.move_on_unpin

    # Verify that setting move-on-unpin works as expected.
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_MOVE_ON_UNPIN": "1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": "100",
        }
    ):
        options = BufferPoolOptions.defaults()
        assert not options.spill_on_unpin
        assert options.move_on_unpin

    # Verify that setting debug-mode works as expected.
    with temp_env_override({"BODO_BUFFER_POOL_DEBUG_MODE": "1"}):
        options = BufferPoolOptions.defaults()
        assert options.debug_mode

    # Test that the default min SizeClass size is chosen correctly
    # based on the usable percent.
    for mem_percent in range(40, 150, 5):
        with temp_env_override(
            {
                "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
                "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": str(mem_percent),
                "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
                "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
            }
        ):
            options = BufferPoolOptions.defaults()
            # It should be 16KiB when mem_fraction is <= 1.0
            if mem_percent <= 100:
                assert options.min_size_class == 16
            # and 64KiB otherwise
            else:
                assert options.min_size_class == 64

    # Test the same thing, but with spilling available.
    # The default min SizeClass size should still be chosen
    # based on the usable percent.
    for mem_percent in range(40, 150, 5):
        with temp_env_override(
            {
                "BODO_BUFFER_POOL_MEMORY_SIZE_MiB": None,
                "BODO_BUFFER_POOL_MEMORY_USABLE_PERCENT": str(mem_percent),
                "BODO_BUFFER_POOL_MIN_SIZE_CLASS_KiB": None,
                "BODO_BUFFER_POOL_MAX_NUM_SIZE_CLASSES": None,
                "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": str(tmp_path),
                "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "1",
                "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": "100",
            }
        ):
            options = BufferPoolOptions.defaults()
            # It should be 16KiB when mem_fraction is <= 1.0
            if mem_percent <= 100:
                assert options.min_size_class == 16
            # and 64KiB otherwise
            else:
                assert options.min_size_class == 64


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


@pytest.mark.s3
def test_default_s3_storage_options(tmp_s3_path: str):
    """
    Test that StorageOptions detection for S3 locations works
    as expected.
    """

    with temp_env_override(
        {
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": tmp_s3_path,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "-1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": "100",
        }
    ):
        options = StorageOptions.defaults(1)
        assert options.location == bytes(tmp_s3_path, "utf-8")
        assert options.usable_size == -1

    # Check that when we explicitly disable remote spilling, we don't
    # use the provided S3 config.
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_DISABLE_REMOTE_SPILLING": "1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": tmp_s3_path,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "-1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": "100",
        }
    ):
        options = StorageOptions.defaults(1)
        assert options is None


def test_default_azure_storage_options(tmp_abfs_path: str):
    """
    Test that StorageOptions detection for Azure storage locations
    works as expected.
    """

    with temp_env_override(
        {
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": tmp_abfs_path,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "-1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": "100",
        }
    ):
        options = StorageOptions.defaults(1)
        assert options.location == bytes(tmp_abfs_path, "utf-8")
        assert options.usable_size == -1

    # Check that when we explicitly disable remote spilling, we don't
    # use the provided S3 config.
    with temp_env_override(
        {
            "BODO_BUFFER_POOL_DISABLE_REMOTE_SPILLING": "1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_DRIVES": tmp_abfs_path,
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_SPACE_PER_DRIVE_GiB": "-1",
            "BODO_BUFFER_POOL_STORAGE_CONFIG_1_USABLE_PERCENTAGE": "100",
        }
    ):
        options = StorageOptions.defaults(1)
        assert options is None


def test_malloc_allocation():
    """
    Test that small allocations that go through
    malloc work as expected.
    """
    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 5KiB which is <(3/4)*8KiB and therefore should
    # go through malloc
    allocation: BufferPoolAllocation = pool.allocate(5 * 1024)

    # Verify that default allocation is 64B aligned
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
            assert not size_class.is_frame_mapped(frame_idx), (
                f"Frame at index {frame_idx} is mapped even though it shouldn't be!"
            )
            assert not size_class.is_frame_pinned(frame_idx), (
                f"Frame at index {frame_idx} is pinned even though it shouldn't be!"
            )

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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
    )
    pool: BufferPool = BufferPool.from_options(options)

    size_class: SizeClass = pool.get_size_class(0)

    # Allocate 6KiB+1 (minimum amount to allocate through mmap)
    alloc1: BufferPoolAllocation = pool.allocate((6 * 1024) + 1)

    # Verify that default allocation is 64B aligned
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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 12KiB (in the 16KiB SizeClass)
    alloc1: BufferPoolAllocation = pool.allocate(12 * 1024)

    # Verify that default allocation is 64B aligned
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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
    )
    pool: BufferPool = BufferPool.from_options(options)

    # The largest size-class (index 9) will have
    # capacity: 2 and block_size: 4194304 (4MiB)
    size_class: SizeClass = pool.get_size_class(9)

    # Allocate 3MiB (in the 4MiB SizeClass)
    alloc1: BufferPoolAllocation = pool.allocate(3 * 1024 * 1024)

    # Verify that default allocation is 64B aligned
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
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=4,
        enforce_max_limit_during_allocation=True,
    )
    pool: BufferPool = BufferPool.from_options(options)

    alloc_size = (4 * 1024 * 1024) + 1
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Request allocation size ({alloc_size}) is larger than the largest block-size available!"
        ),
    ):
        _: BufferPoolAllocation = pool.allocate(alloc_size)

    alloc_size = 5 * 1024 * 1024
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Request allocation size ({alloc_size}) is larger than the largest block-size available!"
        ),
    ):
        _: BufferPoolAllocation = pool.allocate(alloc_size)

    # Verify stats after allocation
    assert pool.bytes_allocated() == 0
    assert pool.max_memory() == 0

    # Verify that none of the bits are set in either bitmap
    # of any of the SizeClass-es
    for size_class_idx in range(pool.num_size_classes()):
        size_class: SizeClass = pool.get_size_class(size_class_idx)
        num_blocks = size_class.get_num_blocks()
        for frame_idx in range(num_blocks):
            assert not size_class.is_frame_mapped(frame_idx), (
                f"Frame at index {frame_idx} is mapped even though it shouldn't be!"
            )
            assert not size_class.is_frame_pinned(frame_idx), (
                f"Frame at index {frame_idx} is pinned even though it shouldn't be!"
            )

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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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
    page_size = mmap.PAGESIZE
    req_alignment = page_size * 2
    with pytest.raises(
        ValueError,
        match=r"Requested alignment \(.*\) higher than max supported alignment \(.*\).",
    ):
        _: BufferPoolAllocation = pool.allocate(20 * 1024, req_alignment)

    # Trying to allocate with non power of 2 alignment should error out
    with pytest.raises(
        ValueError,
        match=r"Alignment \(.*\) must be a positive number and a power of 2.",
    ):
        # Malloc
        _: BufferPoolAllocation = pool.allocate(10, 48)

    with pytest.raises(
        ValueError,
        match=r"Alignment \(.*\) must be a positive number and a power of 2.",
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
    options = BufferPoolOptions(
        memory_size=7,
        min_size_class=4,
        enforce_max_limit_during_allocation=True,
    )
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
    options = BufferPoolOptions(
        memory_size=7,
        min_size_class=4,
        max_num_size_classes=5,
        enforce_max_limit_during_allocation=True,
    )
    pool: BufferPool = BufferPool.from_options(options)
    assert pool.num_size_classes() == 5
    del pool

    options = BufferPoolOptions(
        memory_size=7,
        min_size_class=4,
        max_num_size_classes=15,
        enforce_max_limit_during_allocation=True,
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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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
    This reuses the same frame as before
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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

    assert size_class.is_frame_mapped(0)
    assert size_class.is_frame_pinned(0)
    assert not size_class.is_frame_mapped(1)
    assert not size_class.is_frame_pinned(1)

    for frame_idx in range(2, size_class.get_num_blocks()):
        assert not size_class.is_frame_mapped(frame_idx)
        assert not size_class.is_frame_pinned(frame_idx)

    assert size_class.get_swip_at_frame(0) == allocation.get_swip_as_int()
    assert size_class.get_swip_at_frame(0) != orig_allocation_copy.get_swip_as_int()

    assert pool.bytes_allocated() == 16 * 1024
    assert pool.max_memory() == 16 * 1024

    assert allocation != orig_allocation_copy
    assert allocation.has_same_ptr(orig_allocation_copy)

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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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
            assert not size_class.is_frame_mapped(frame_idx), (
                f"Frame at index {frame_idx} is mapped even though it shouldn't be!"
            )
            assert not size_class.is_frame_pinned(frame_idx), (
                f"Frame at index {frame_idx} is pinned even though it shouldn't be!"
            )

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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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
            assert not size_class.is_frame_mapped(frame_idx), (
                f"Frame at index {frame_idx} is mapped even though it shouldn't be!"
            )
            assert not size_class.is_frame_pinned(frame_idx), (
                f"Frame at index {frame_idx} is pinned even though it shouldn't be!"
            )

    assert pool.bytes_allocated() == 0
    assert pool.max_memory() == 0

    # Re-allocate to 5KiB (will go through mmap)
    pool.reallocate(5 * 1024, allocation)

    for size_class_idx in range(pool.num_size_classes()):
        size_class: SizeClass = pool.get_size_class(size_class_idx)
        num_blocks = size_class.get_num_blocks()
        for frame_idx in range(num_blocks):
            assert not size_class.is_frame_mapped(frame_idx), (
                f"Frame at index {frame_idx} is mapped even though it shouldn't be!"
            )
            assert not size_class.is_frame_pinned(frame_idx), (
                f"Frame at index {frame_idx} is pinned even though it shouldn't be!"
            )

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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
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


def test_reallocate_unpinned_frame():
    """
    Verify that reallocating an unpinned frame works as expected.
    We also test the edge case where the new allocation cannot
    be successfully made. In those cases, the original frame
    must remain unpinned. In our reallocate implementation, we
    pin the original and then allocate the new memory block, so
    we have to unpin the original again before returning in
    case of an allocation failure.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=16,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
        debug_mode=True,
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 50KiB
    allocation: BufferPoolAllocation = pool.allocate(50 * 1024)
    assert pool.bytes_allocated() == 64 * 1024
    assert pool.bytes_pinned() == 64 * 1024

    # Unpin the allocation
    pool.unpin(allocation)
    assert pool.bytes_allocated() == 64 * 1024
    assert pool.bytes_pinned() == 0
    assert not pool.is_pinned(allocation)

    # Reallocate to 2MiB
    pool.reallocate(2 * 1024 * 1024, allocation)
    assert pool.bytes_allocated() == 2 * 1024 * 1024
    assert pool.bytes_pinned() == 2 * 1024 * 1024
    assert pool.is_pinned(allocation)

    # Unpin it
    pool.unpin(allocation)
    assert pool.bytes_allocated() == 2 * 1024 * 1024
    assert pool.bytes_pinned() == 0
    assert not pool.is_pinned(allocation)

    # Test the case where new allocation fails
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match=re.escape(
            "BufferPool::Reallocate: Allocation of new memory failed: Out of memory: Allocation canceled beforehand. Not enough space in the buffer pool to allocate (requested 3145728 bytes, aligned 3145728 bytes, available 2097152 bytes)."
        ),
    ):
        pool.reallocate(3 * 1024 * 1024, allocation)

    # Verify that the original remains unpinned
    assert pool.bytes_allocated() == 2 * 1024 * 1024
    assert pool.bytes_pinned() == 0
    assert not pool.is_pinned(allocation)

    pool.free(allocation)
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    del pool


def test_pyarrow_allocation():
    """
    Test that setting our BufferPool as the PyArrow
    MemoryPool works by verifying that subsequent
    allocations go through the BufferPool as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=4,
        enforce_max_limit_during_allocation=True,
    )
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
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=True,
    )
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
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=True,
        debug_mode=True,
    )
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
        match="Malloc canceled beforehand. Not enough space in the buffer pool.",
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
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=True,
        debug_mode=True,
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
    pool.unpin(allocation_1)

    # Trying to allocate 2MiB should fail
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match="Allocation canceled beforehand. Not enough space in the buffer pool.",
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
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=True,
        debug_mode=True,
    )
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
        match="Spilling is not available to free up sufficient space in memory",
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


def test_larger_than_available_space_allocation_limit_ignored(capfd):
    """
    Test that trying to allocate more memory than
    space available in the buffer pool doesn't raise
    an error when enforce_max_limit_during_allocation is False
    and there actually is enough space in physical memory (and
    a buffer frame of appropriate size is available in the pool).
    Also verifies that if all frames are taken up, the appropriate
    error is raised even if enforce_max_limit_during_allocation is
    False.
    """
    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=4,
        enforce_max_limit_during_allocation=False,
        # Print warnings so we can verify correct behavior
        debug_mode=True,
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
    # a free 1MiB frame available. However, a warning should be displayed
    # on stderr.
    allocation_4: BufferPoolAllocation = pool.allocate(1024 * 1024)
    _, err = capfd.readouterr()
    assert "We will try to allocate anyway. This may invoke the OOM killer" in err

    # Verify stats after allocation attempt
    assert pool.bytes_allocated() == 4.5 * 1024 * 1024
    assert pool.max_memory() == 4.5 * 1024 * 1024

    # Try to fill up all 1MiB frames (there are 8 of these since we over-allocate
    # frames from the largest SizeClass-es).
    alloc_1MiB_frames = []
    for i in range(6):
        alloc_1MiB_frame: BufferPoolAllocation = pool.allocate(1024 * 1024)
        _, err = capfd.readouterr()
        assert (
            "We will try to allocate anyway. This may invoke the OOM killer" in err
        ), i
        alloc_1MiB_frames.append(alloc_1MiB_frame)

    # Try allocating again:
    # Trying to allocate 1MiB should fail
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match=re.escape("Could not find an empty frame of required size (1048576)!"),
    ):
        _: BufferPoolAllocation = pool.allocate(1024 * 1024)
    # Verify that a warning was displayed on stderr before it raised the error
    _, err = capfd.readouterr()
    assert "We will try to allocate anyway. This may invoke the OOM killer" in err

    # Verify stats after allocation attempt
    assert pool.bytes_allocated() == 10.5 * 1024 * 1024
    assert pool.max_memory() == 10.5 * 1024 * 1024

    # Unpinning and pinning should continue working
    pool.unpin(alloc_1MiB_frames[1])
    pool.pin(alloc_1MiB_frames[1])

    # Unpinning and a new allocation should be fine
    pool.unpin(alloc_1MiB_frames[1])

    allocation_11: BufferPoolAllocation = pool.allocate(512 * 1024)
    _, err = capfd.readouterr()
    assert "We will try to allocate anyway. This may invoke the OOM killer" in err

    pool.pin(alloc_1MiB_frames[1])

    pool.free(allocation_1)
    pool.free(allocation_2)
    pool.free(allocation_3)
    pool.free(allocation_4)
    for alloc in alloc_1MiB_frames:
        pool.free(alloc)
    pool.free(allocation_11)

    # Delete pool (to be conservative)
    del pool


def test_allocate_cannot_evict_sufficient_bytes_no_enforcement(tmp_path: Path, capfd):
    """
    Test that trying to allocate when there's not enough
    space in the buffer-pool even after evicting all possible
    frames, works correctly when max limit enforcement is
    disabled.
    Tests for both the frame available and not available
    cases.
    """
    # Allocate a very small pool for testing
    local_opt = StorageOptions(usable_size=16 * 1024 * 1024, location=bytes(tmp_path))
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=False,
        # Print warnings so we can verify correct behavior
        debug_mode=True,
        storage_options=[local_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)

    allocation1: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    allocation2: BufferPoolAllocation = pool.allocate(1 * 1024 * 1024)
    allocation3: BufferPoolAllocation = pool.allocate(1 * 1024 * 1024)
    pool.unpin(allocation2)

    # Verify stats after allocation + unpin
    assert pool.bytes_pinned() == 3 * 1024 * 1024
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024

    # At this point, there's no space left, so allocating 2MiB will
    # trigger a best effort spill. We will only be able to spill
    # 1MiB though.
    allocation4: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    # Verify that a warning is displayed on stderr
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )
    # We should've triggered a spill of the unpinned frame
    assert not allocation2.is_in_memory()

    # Allocate the remaining 2MiB frames (we allocate extra frames when enforcement is turned off)
    allocation5: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )
    allocation6: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )

    # Verify stats
    assert pool.bytes_pinned() == 9 * 1024 * 1024
    assert pool.bytes_in_memory() == 9 * 1024 * 1024
    assert pool.bytes_allocated() == 10 * 1024 * 1024
    assert pool.max_memory() == 10 * 1024 * 1024

    # Reduce memory pressure further by making 1MiB eligible
    # for eviction.
    pool.unpin(allocation3)

    # Test for the frame not available case (we're out of 2MiB frames at this point):
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match=re.escape("Could not find an empty frame of required size (2097152)!"),
    ):
        _: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    # Verify that a warning is displayed on stderr
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )

    # Verify stats
    assert pool.bytes_pinned() == 8 * 1024 * 1024
    assert pool.bytes_in_memory() == 8 * 1024 * 1024
    assert pool.bytes_allocated() == 10 * 1024 * 1024
    assert pool.max_memory() == 10 * 1024 * 1024
    # We should've triggered a spill of the unpinned frame
    assert not allocation3.is_in_memory()

    # Do a similar test for malloc
    allocation7: BufferPoolAllocation = pool.allocate(2 * 1024)
    # Verify that a warning was displayed on stderr before it raised the error
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )
    # These should stay evicted.
    assert not allocation2.is_in_memory()
    assert not allocation3.is_in_memory()

    pool.free(allocation1)
    pool.free(allocation2)
    pool.free(allocation3)
    pool.free(allocation4)
    pool.free(allocation5)
    pool.free(allocation6)
    pool.free(allocation7)

    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0

    pool.cleanup()
    del pool


def test_allocate_cannot_evict_sufficient_bytes_no_enforcement_very_large_spill(
    tmp_path: Path, capfd
):
    """
    Test that trying to allocate when there's not enough
    space in the buffer-pool even after evicting all possible
    frames, works correctly when max limit enforcement is
    disabled. In particular, this tests for the case where
    the amount to be spilled to relieve memory pressure is
    larger than any SizeClass.
    """
    # Allocate a very small pool for testing
    local_opt = StorageOptions(usable_size=16 * 1024 * 1024, location=bytes(tmp_path))
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=False,
        # Print warnings so we can verify correct behavior
        debug_mode=True,
        storage_options=[local_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Fill up all the frames
    allocation1: BufferPoolAllocation = pool.allocate(4 * 1024 * 1024)
    _, err = capfd.readouterr()
    # No warning on this one
    assert len(err) == 0

    allocation2: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )
    allocation3: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )
    allocation4: BufferPoolAllocation = pool.allocate(1 * 1024 * 1024)
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )
    allocation5: BufferPoolAllocation = pool.allocate(1 * 1024 * 1024)
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )
    allocation6: BufferPoolAllocation = pool.allocate(1 * 1024 * 1024)
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )
    allocation7: BufferPoolAllocation = pool.allocate(1 * 1024 * 1024)
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )

    # Verify stats
    assert pool.bytes_pinned() == 12 * 1024 * 1024
    assert pool.bytes_allocated() == 12 * 1024 * 1024
    assert pool.max_memory() == 12 * 1024 * 1024

    # Unpin some to make them eligible for eviction
    pool.unpin(allocation1)
    pool.unpin(allocation2)
    pool.unpin(allocation3)
    pool.unpin(allocation4)
    pool.unpin(allocation5)

    # Verify stats
    assert pool.bytes_pinned() == 2 * 1024 * 1024
    assert pool.bytes_allocated() == 12 * 1024 * 1024
    assert pool.max_memory() == 12 * 1024 * 1024

    # We're 8MiB over-allocated currently (4MiB vs 12MiB). Trying to allocate another 4MiB
    # will try to spill as much as 4 - (-8) = 12MiB. We'll only be able
    # to spill 10MiB though.
    allocation8: BufferPoolAllocation = pool.allocate(4 * 1024 * 1024)
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )

    # Verify stats
    assert pool.bytes_pinned() == 6 * 1024 * 1024
    assert pool.bytes_in_memory() == 6 * 1024 * 1024
    assert pool.bytes_allocated() == 16 * 1024 * 1024
    assert pool.max_memory() == 16 * 1024 * 1024
    assert not allocation1.is_in_memory()
    assert not allocation2.is_in_memory()
    assert not allocation3.is_in_memory()
    assert not allocation4.is_in_memory()
    assert not allocation5.is_in_memory()
    assert allocation6.is_in_memory()
    assert allocation7.is_in_memory()
    assert allocation8.is_in_memory()

    pool.free(allocation1)
    pool.free(allocation2)
    pool.free(allocation3)
    pool.free(allocation4)
    pool.free(allocation5)
    pool.free(allocation6)
    pool.free(allocation7)
    pool.free(allocation8)

    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0

    pool.cleanup()
    del pool


def test_pin_evicted_block_not_evicted_sufficient_bytes_no_enforcement(
    tmp_path: Path, capfd
):
    """
    Test that trying to pin an evicted block when there's
    not enough space in the buffer-pool even after evicting
    all possible frames, works correctly when the max limit
    enforcement is disabled.
    Tests for both the frame available and not available
    cases.
    """
    # Allocate a very small pool for testing
    local_opt = StorageOptions(usable_size=16 * 1024 * 1024, location=bytes(tmp_path))
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=False,
        # Print warnings so we can verify correct behavior
        debug_mode=True,
        storage_options=[local_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)

    allocation1: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    allocation2: BufferPoolAllocation = pool.allocate(1 * 1024 * 1024)
    allocation3: BufferPoolAllocation = pool.allocate(1 * 1024 * 1024)
    pool.unpin(allocation1)
    allocation4: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)

    # Verify stats after allocation + unpin
    assert pool.bytes_pinned() == 4 * 1024 * 1024
    assert pool.bytes_in_memory() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 6 * 1024 * 1024
    assert pool.max_memory() == 6 * 1024 * 1024
    assert not allocation1.is_in_memory()

    # Make 1MiB eligible for eviction
    pool.unpin(allocation2)

    # Try to pin allocation1 back:
    pool.pin(allocation1)
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )

    assert allocation1.is_in_memory()
    assert not allocation2.is_in_memory()
    assert pool.bytes_pinned() == 5 * 1024 * 1024
    assert pool.bytes_in_memory() == 5 * 1024 * 1024
    assert pool.bytes_allocated() == 6 * 1024 * 1024
    assert pool.max_memory() == 6 * 1024 * 1024

    # Reduce memory pressure further
    pool.unpin(allocation3)
    pool.unpin(allocation4)

    # Verify stats
    assert pool.bytes_pinned() == 2 * 1024 * 1024
    assert pool.bytes_in_memory() == 5 * 1024 * 1024
    assert pool.bytes_allocated() == 6 * 1024 * 1024
    assert pool.max_memory() == 6 * 1024 * 1024
    assert not allocation2.is_in_memory()

    allocation5: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    _, err = capfd.readouterr()
    # No warning should be raised here. We need to free up 3MiB, and
    # we can do that by spilling allocation3 and allocation4.
    assert len(err) == 0

    # Verify stats
    assert pool.bytes_pinned() == 4 * 1024 * 1024
    assert pool.bytes_in_memory() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 8 * 1024 * 1024
    assert pool.max_memory() == 8 * 1024 * 1024
    assert not allocation2.is_in_memory()
    # Best effort spill will spill as much as possible
    # to get memory pressure under control:
    assert not allocation3.is_in_memory()
    assert not allocation4.is_in_memory()

    # Use up the remaining 2MiB frames (these exist because we allocate extra
    # frames for the larger SizeClass-es).
    allocation6: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    allocation7: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)

    # Test for the frame not available case (we're out of 2MiB frames at this point):
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match="Pin failed. Unable to find available frame",
    ):
        pool.pin(allocation4)
    # Verify that a warning is displayed on stderr
    _, err = capfd.readouterr()
    assert (
        "Could not spill sufficient bytes. We will try to allocate anyway. This may invoke the OOM killer"
        in err
    )

    # Verify stats
    assert pool.bytes_pinned() == 8 * 1024 * 1024
    assert pool.bytes_in_memory() == 8 * 1024 * 1024
    assert pool.bytes_allocated() == 12 * 1024 * 1024
    assert pool.max_memory() == 12 * 1024 * 1024
    assert not allocation2.is_in_memory()
    assert not allocation3.is_in_memory()
    assert not allocation4.is_in_memory()

    pool.free(allocation1)
    pool.free(allocation2)
    pool.free(allocation3)
    pool.free(allocation4)
    pool.free(allocation5)
    pool.free(allocation6)
    pool.free(allocation7)

    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0

    pool.cleanup()
    del pool


def test_extra_frames_no_enforcement():
    """
    Test that extra frames are allocated in the larger SizeClass-es
    when threshold enforcement is disabled.
    """

    ## 1. When there are >16 SizeClass-es
    options = BufferPoolOptions(
        memory_size=128,  # MiB
        min_size_class=1,  # KiB
        enforce_max_limit_during_allocation=False,
        debug_mode=False,
    )
    pool: BufferPool = BufferPool.from_options(options)

    for idx, exp_num_frames in [
        # 8 largest (1MiB, ..., 128MiB) should have 2x frames:
        (10, 256),
        (11, 128),
        (12, 64),
        (13, 32),
        (14, 16),
        (15, 8),
        (16, 4),
        (17, 2),
        # Next 8 (512KiB, ..., 4KiB) should have 1x frames:
        (2, 49152),
        (3, 24576),
        (4, 12288),
        (5, 6144),
        (6, 3072),
        (7, 1536),
        (8, 768),
        (9, 384),
        # Rest (2KiB & 1KiB) don't have extra frames:
        (0, 131072),
        (1, 65536),
    ]:
        size_class_: SizeClass = pool.get_size_class(idx)
        n_blocks = size_class_.get_num_blocks()
        assert n_blocks == exp_num_frames, (
            f"Expected SizeClass at idx {idx} to have {exp_num_frames} frames but it has {n_blocks} frames instead!"
        )

    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0

    pool.cleanup()
    del pool

    ## 2. Test when there are fewer than 16 SizeClass-es
    options = BufferPoolOptions(
        memory_size=16,  # MiB
        min_size_class=2,  # KiB
        enforce_max_limit_during_allocation=False,
        debug_mode=False,
    )
    pool: BufferPool = BufferPool.from_options(options)

    for idx, exp_num_frames in [
        # Smallest SizeClass (2KiB) shouldn't have extra frames
        (0, 8192),
        # Next 5 (4KiB, ..., 64KiB) should have 1.5x frames
        (1, 6144),
        (2, 3072),
        (3, 1536),
        (4, 768),
        (5, 384),
        # 8 largest (128KiB, ..., 16MiB) should have 2x frames
        (6, 256),
        (7, 128),
        (8, 64),
        (9, 32),
        (10, 16),
        (11, 8),
        (12, 4),
        (13, 2),
    ]:
        size_class_: SizeClass = pool.get_size_class(idx)
        n_blocks = size_class_.get_num_blocks()
        assert n_blocks == exp_num_frames, (
            f"Expected SizeClass at idx {idx} to have {exp_num_frames} frames but it has {n_blocks} frames instead!"
        )

    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0

    pool.cleanup()
    del pool

    ## 3. Test edge cases with very few SizeClass-es
    options = BufferPoolOptions(
        memory_size=4,  # MiB
        min_size_class=512,  # KiB
        enforce_max_limit_during_allocation=False,
        debug_mode=False,
    )
    pool: BufferPool = BufferPool.from_options(options)

    for idx, exp_num_frames in [
        # Smallest SizeClass (512KiB) shouldn't have extra frames
        (0, 8),
        # 3 largest (1MiB, ..., 4MiB) should have 2x frames
        (1, 8),
        (2, 4),
        (3, 2),
    ]:
        size_class_: SizeClass = pool.get_size_class(idx)
        n_blocks = size_class_.get_num_blocks()
        assert n_blocks == exp_num_frames, (
            f"Expected SizeClass at idx {idx} to have {exp_num_frames} frames but it has {n_blocks} frames instead!"
        )

    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0

    pool.cleanup()
    del pool


def test_no_warning_debug_mode_disabled(tmp_path: Path, capfd):
    """
    Test that no warnings are displayed when debug mode
    is disabled.
    """
    # Allocate a very small pool for testing
    local_opt = StorageOptions(usable_size=16 * 1024 * 1024, location=bytes(tmp_path))
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=False,
        debug_mode=False,
        storage_options=[local_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)

    allocation1: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)
    allocation2: BufferPoolAllocation = pool.allocate(1 * 1024 * 1024)
    allocation3: BufferPoolAllocation = pool.allocate(1 * 1024 * 1024)
    pool.unpin(allocation1)
    allocation4: BufferPoolAllocation = pool.allocate(2 * 1024 * 1024)

    # Verify stats after allocation + unpin
    assert pool.bytes_pinned() == 4 * 1024 * 1024
    assert pool.bytes_in_memory() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 6 * 1024 * 1024
    assert pool.max_memory() == 6 * 1024 * 1024
    assert not allocation1.is_in_memory()

    # Make 1MiB eligible for eviction
    pool.unpin(allocation2)

    # Try to pin allocation1 back. This would display a warning
    # if debug mode was enabled.
    pool.pin(allocation1)
    _, err = capfd.readouterr()
    assert len(err) == 0

    assert allocation1.is_in_memory()
    assert not allocation2.is_in_memory()
    assert pool.bytes_pinned() == 5 * 1024 * 1024
    assert pool.bytes_in_memory() == 5 * 1024 * 1024
    assert pool.bytes_allocated() == 6 * 1024 * 1024
    assert pool.max_memory() == 6 * 1024 * 1024

    pool.free(allocation1)
    pool.free(allocation2)
    pool.free(allocation3)
    pool.free(allocation4)

    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0

    pool.cleanup()
    del pool


def test_allocate_spill_eq_block(tmp_path: Path):
    """
    Test that trying to allocate more memory than
    space available leads to spilling an equal-size frame
    """

    # Allocate a very small pool for testing
    local_opt = StorageOptions(usable_size=5 * 1024 * 1024, location=bytes(tmp_path))
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=True,
        storage_options=[local_opt],
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
    assert pool.bytes_in_memory() == 3.5 * 1024 * 1024
    # assert pool.max_memory() == 3.5 * 1024 * 1024
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
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=True,
        storage_options=[local_opt],
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
    assert pool.bytes_in_memory() == 3.5 * 1024 * 1024
    assert pool.bytes_allocated() == 5.5 * 1024 * 1024
    assert pool.max_memory() == 5.5 * 1024 * 1024
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
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=True,
        storage_options=[local_opt],
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
    assert pool.bytes_in_memory() == 3 * 1024 * 1024
    assert pool.bytes_allocated() == 5 * 1024 * 1024
    assert pool.max_memory() == 5 * 1024 * 1024
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
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=True,
        storage_options=[local_opt],
        debug_mode=True,
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
    assert pool.bytes_in_memory() == 3 * 1024 * 1024
    assert pool.bytes_allocated() == 5 * 1024 * 1024
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
    assert pool.bytes_in_memory() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 5 * 1024 * 1024
    assert pool.max_memory() == 5 * 1024 * 1024
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
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=True,
        storage_options=[local_opt],
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
    assert pool.bytes_in_memory() == 1 * 1024 * 1024
    assert pool.bytes_allocated() == 5 * 1024 * 1024
    assert pool.max_memory() == 5 * 1024 * 1024
    assert not allocation_1.is_nullptr() and not allocation_1.is_in_memory()

    # Repin allocation_1 and read contents
    pool.unpin(allocation_2)
    pool.pin(allocation_1)
    read_bytes: bytes = allocation_1.read_bytes(len(write_bytes))

    # Verify stats after repin
    assert pool.bytes_pinned() == 4 * 1024 * 1024
    assert pool.bytes_in_memory() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 5 * 1024 * 1024
    assert pool.max_memory() == 5 * 1024 * 1024
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
        memory_size=4,
        min_size_class=1024,
        enforce_max_limit_during_allocation=True,
        storage_options=[local_opt],
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
    assert pool.bytes_in_memory() == 1 * 1024 * 1024
    # assert pool.max_memory() == 4 * 1024 * 1024
    assert not allocation_1.is_nullptr() and not allocation_1.is_in_memory()

    # Look for spilled file in expected location
    # First Folder Expected to Have Prepended Rank
    paths = list(tmp_path.glob(f"{bodo.get_rank()}-*"))
    assert len(paths) == 1
    inner_path = paths[0]
    # Block File should be in {size_class_bytes} folder
    block_file = inner_path / str(4 * 1024 * 1024)
    assert block_file.is_file()

    # Check file contents
    expected_contents = (
        write_bytes
        + b"\x00"
        + (255 - len(write_bytes)) * b"\xcb"
        + (4 * 1024 * 1024 - 256) * b"\x00"
    )
    assert block_file.read_bytes() == expected_contents

    pool.free(allocation_1)
    pool.free(allocation_2)
    assert pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0
    # Make sure free deletes any spill files as well
    assert len([elem for elem in tmp_path.iterdir() if elem.is_file()]) == 0

    # Cleanup
    pool.cleanup()
    assert len(list(tmp_path.iterdir())) == 0

    # Delete pool (to be conservative)
    del pool


def test_reallocate_spilled_block(tmp_path: Path):
    """
    Verify that reallocating a spilled block works as expected.
    We also test the edge case where the new allocation cannot
    be successfully made. In those cases, the original frame
    must be in memory but unpinned. In our reallocate
    implementation, we pin the original block and therefore
    read it from disk. We then allocate the new memory block.
    In the case where this allocation fails, we must at least
    unpin the original frame back. We don't spill it back to disk
    since future allocations can do that if required. If we
    were able to read it into memory, there must be sufficient
    space in the buffer pool.
    """
    # Allocate a very small pool for testing
    local_opt = StorageOptions(usable_size=100 * 1024 * 1024, location=bytes(tmp_path))
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=16,
        enforce_max_limit_during_allocation=True,
        storage_options=[local_opt],
        debug_mode=True,
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 1MiB
    allocation1: BufferPoolAllocation = pool.allocate(1024 * 1024)
    assert pool.bytes_allocated() == 1024 * 1024
    assert pool.bytes_pinned() == 1024 * 1024
    assert pool.is_pinned(allocation1)

    # Unpin it
    pool.unpin(allocation1)
    assert pool.bytes_allocated() == 1024 * 1024
    assert pool.bytes_pinned() == 0
    assert not pool.is_pinned(allocation1)

    # Allocate 3MiB
    allocation2: BufferPoolAllocation = pool.allocate(3 * 1024 * 1024)
    assert pool.bytes_in_memory() == 4 * 1024 * 1024
    assert pool.bytes_allocated() == 5 * 1024 * 1024
    assert pool.bytes_pinned() == 4 * 1024 * 1024
    assert pool.is_pinned(allocation2)

    # Verify that the first allocation was spilled
    assert not pool.is_pinned(allocation1)
    assert not allocation1.is_in_memory()

    # Free the 3MiB allocation
    pool.free(allocation2)
    assert pool.bytes_allocated() == 1024 * 1024
    assert pool.bytes_in_memory() == 0
    assert pool.bytes_pinned() == 0
    assert not pool.is_pinned(allocation1)
    assert not allocation1.is_in_memory()

    # Reallocate 1MiB block to 2MiB
    pool.reallocate(2 * 1024 * 1024, allocation1)
    assert pool.bytes_allocated() == 2 * 1024 * 1024
    assert pool.bytes_pinned() == 2 * 1024 * 1024

    # Verify that it is pinned
    assert pool.is_pinned(allocation1)

    # Unpin it
    pool.unpin(allocation1)
    assert pool.bytes_allocated() == 2 * 1024 * 1024
    assert pool.bytes_pinned() == 0
    assert not pool.is_pinned(allocation1)
    assert allocation1.is_in_memory()

    # Allocate 3MiB
    allocation3: BufferPoolAllocation = pool.allocate(3 * 1024 * 1024)
    assert pool.bytes_allocated() == 6 * 1024 * 1024
    assert pool.bytes_in_memory() == 4 * 1024 * 1024
    assert pool.bytes_pinned() == 4 * 1024 * 1024
    assert pool.is_pinned(allocation2)

    # Verify that the first allocation was spilled
    assert not pool.is_pinned(allocation1)
    assert not allocation1.is_in_memory()

    # Free the 3MiB allocation
    pool.free(allocation3)
    assert pool.bytes_allocated() == 2 * 1024 * 1024
    assert pool.bytes_in_memory() == 0
    assert pool.bytes_pinned() == 0
    assert not pool.is_pinned(allocation1)
    assert not allocation1.is_in_memory()

    # Reallocate 2MiB block to 3MiB -- should fail
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match=re.escape(
            "BufferPool::Reallocate: Allocation of new memory failed: Out of memory: Allocation canceled beforehand. Not enough space in the buffer pool to allocate (requested 3145728 bytes, aligned 3145728 bytes, available 2097152 bytes)."
        ),
    ):
        pool.reallocate(3 * 1024 * 1024, allocation1)

    # Verify that the 2MiB block is in memory but unpinned
    assert pool.bytes_allocated() == 2 * 1024 * 1024
    assert pool.bytes_pinned() == 0
    assert not pool.is_pinned(allocation1)
    assert allocation1.is_in_memory()

    # Cleanup
    pool.free(allocation1)
    pool.cleanup()
    assert len(list(tmp_path.iterdir())) == 0
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    del pool


def test_buffer_pool_eq():
    """
    Test that the == and != operators work as expected.
    """

    default_pool1: BufferPool = BufferPool.default()
    default_pool2: BufferPool = BufferPool.default()
    options = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
    )
    pool1: BufferPool = BufferPool.from_options(options)
    pool2: BufferPool = BufferPool.from_options(options)
    pool3 = pool1

    # Test with "not" of inverse everywhere to ensure
    # that both eq and ne are working correctly.
    assert default_pool1 == default_pool2
    assert not (default_pool1 != default_pool2)
    assert pool1 == pool3
    assert not (pool1 != pool3)
    assert pool1 != pool2
    assert not (pool1 == pool2)
    assert default_pool1 != pool1
    assert not (default_pool1 == pool1)

    del pool1
    del pool2
    del pool3


def test_spill_on_unpin(tmp_path: Path):
    """
    Test that spill on unpin functionality
    works as expected.
    """

    # Create a pool with spill_on_unpin=True
    local_opt = StorageOptions(usable_size=100 * 1024 * 1024, location=bytes(tmp_path))
    options: BufferPoolOptions = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        spill_on_unpin=True,
        enforce_max_limit_during_allocation=True,
        storage_options=[local_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Make an allocation that goes through a SizeClass
    write_bytes = b"Hello BufferPool from Bodo!"
    allocation_1: BufferPoolAllocation = pool.allocate(3.5 * 1024 * 1024)
    allocation_1.write_bytes(write_bytes)

    # Unpin it
    pool.unpin(allocation_1)

    # Verify that it's been spilled to disk and contents are correct
    assert pool.bytes_pinned() == 0
    assert pool.bytes_in_memory() == 0
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert not allocation_1.is_nullptr() and not allocation_1.is_in_memory()

    # Look for spilled file in expected location
    # First Folder Expected to Have Prepended Rank
    paths = list(tmp_path.glob(f"{bodo.get_rank()}-*"))
    assert len(paths) == 1
    inner_path = paths[0]
    # Block File should be in {size_class_bytes} folder
    block_file = inner_path / str(4 * 1024 * 1024)
    assert block_file.is_file()

    # Check file contents
    expected_contents = (
        write_bytes
        + b"\x00"
        + (255 - len(write_bytes)) * b"\xcb"
        + (4 * 1024 * 1024 - 256) * b"\x00"
    )
    assert block_file.read_bytes() == expected_contents

    # Pin it back and verify contents
    pool.pin(allocation_1)
    contents_after_pin_back = allocation_1.read_bytes(int(3.5 * 1024 * 1024))
    assert contents_after_pin_back == write_bytes

    pool.free(allocation_1)
    pool.cleanup()
    del pool


def test_move_on_unpin_move_frame_case(tmp_path: Path):
    """
    Test that the "moveFrame" case of move on
    unpin works as expected.
    """
    # Create a pool with move_on_unpin=True
    local_opt = StorageOptions(usable_size=100 * 1024 * 1024, location=bytes(tmp_path))
    options: BufferPoolOptions = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=15,
        move_on_unpin=True,
        enforce_max_limit_during_allocation=True,
        storage_options=[local_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate from SizeClass other largest one (i.e. at least 2 frames).
    write_bytes = b"Hello BufferPool from Bodo!"
    allocation: BufferPoolAllocation = pool.allocate(1.5 * 1024 * 1024)
    allocation.write_bytes(write_bytes)
    orig_ptr_alloc = allocation.get_ptr_as_int()

    # Unpin allocation so that it moves it. Verify that
    # it's been moved.
    pool.unpin(allocation)
    assert orig_ptr_alloc != allocation.get_ptr_as_int()
    # Verify that it's at the next frame
    assert allocation.get_ptr_as_int() == (orig_ptr_alloc + 2 * 1024 * 1024)
    # Verify contents -- this is technically unsafe to do when a frame is
    # unpinned, but in this case we're sure it's still in memory.
    assert write_bytes == allocation.read_bytes(int(1.5 * 1024 * 1024))

    # Pin back and verify that it's still at another frame
    pool.pin(allocation)
    assert orig_ptr_alloc != allocation.get_ptr_as_int()
    assert allocation.get_ptr_as_int() == (orig_ptr_alloc + 2 * 1024 * 1024)
    assert write_bytes == allocation.read_bytes(int(1.5 * 1024 * 1024))

    pool.free(allocation)
    pool.cleanup()
    del pool


def test_move_on_unpin_swap_frames_case(tmp_path: Path):
    """
    Test that the "swapFrames" case of move on
    unpin works as expected.
    """

    # Create a pool with move_on_unpin=True
    local_opt = StorageOptions(usable_size=100 * 1024 * 1024, location=bytes(tmp_path))
    options: BufferPoolOptions = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=15,
        move_on_unpin=True,
        enforce_max_limit_during_allocation=True,
        storage_options=[local_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate from SizeClass with two frames (4MiB) and unpin it
    write_bytes_1 = b"[1] Hello BufferPool from Bodo!"
    allocation_1: BufferPoolAllocation = pool.allocate(3.5 * 1024 * 1024)
    allocation_1.write_bytes(write_bytes_1)
    orig_ptr_alloc_1 = allocation_1.get_ptr_as_int()
    pool.unpin(allocation_1)
    ptr_after_unpin_alloc_1 = allocation_1.get_ptr_as_int()
    assert write_bytes_1 == allocation_1.read_bytes(int(3.5 * 1024 * 1024))
    assert ptr_after_unpin_alloc_1 != orig_ptr_alloc_1
    assert ptr_after_unpin_alloc_1 == (orig_ptr_alloc_1 + 4 * 1024 * 1024)

    # Allocate again from the same SizeClass and unpin it. This will swap it.
    write_bytes_2 = b"[2] Hello BufferPool from Bodo!"
    allocation_2: BufferPoolAllocation = pool.allocate(3.5 * 1024 * 1024)
    allocation_2.write_bytes(write_bytes_2)
    orig_ptr_alloc_2 = allocation_2.get_ptr_as_int()
    pool.unpin(allocation_2)
    ptr_after_unpin_alloc_2 = allocation_2.get_ptr_as_int()
    # Update alloc_1 pointer
    ptr_after_unpin_alloc_1 = allocation_1.get_ptr_as_int()
    assert write_bytes_2 == allocation_2.read_bytes(int(3.5 * 1024 * 1024))
    assert ptr_after_unpin_alloc_2 != orig_ptr_alloc_2
    # Confirm the swap
    assert ptr_after_unpin_alloc_1 == orig_ptr_alloc_1 == orig_ptr_alloc_2
    assert ptr_after_unpin_alloc_2 != orig_ptr_alloc_1

    # Pin back both and verify addresses and contents
    pool.pin(allocation_1)
    ptr_after_pin_alloc_1 = allocation_1.get_ptr_as_int()
    assert write_bytes_1 == allocation_1.read_bytes(int(3.5 * 1024 * 1024))
    assert ptr_after_pin_alloc_1 == ptr_after_unpin_alloc_1
    pool.pin(allocation_2)
    ptr_after_pin_alloc_2 = allocation_2.get_ptr_as_int()
    assert write_bytes_2 == allocation_2.read_bytes(int(3.5 * 1024 * 1024))
    assert ptr_after_pin_alloc_2 == ptr_after_unpin_alloc_2

    pool.free(allocation_1)
    pool.free(allocation_2)

    pool.cleanup()
    del pool


def test_move_on_unpin_spill_case(tmp_path: Path):
    """
    Test that the "spillFrame" case of move on
    unpin works as expected.
    """

    # Create a pool with move_on_unpin=True
    local_opt = StorageOptions(usable_size=100 * 1024 * 1024, location=bytes(tmp_path))
    options: BufferPoolOptions = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=15,
        move_on_unpin=True,
        enforce_max_limit_during_allocation=True,
        storage_options=[local_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate from largest size class with a single frame
    write_bytes = b"[4] Hello BufferPool from Bodo!"
    allocation: BufferPoolAllocation = pool.allocate(6 * 1024 * 1024)
    allocation.write_bytes(write_bytes)

    # Unpin it
    pool.unpin(allocation)

    # Verify that it's been spilled to disk and contents are correct
    assert not allocation.is_nullptr() and not allocation.is_in_memory()

    # Look for spilled file in expected location
    # First Folder Expected to Have Prepended Rank
    paths = list(tmp_path.glob(f"{bodo.get_rank()}-*"))
    assert len(paths) == 1
    inner_path = paths[0]
    # Block File should be in {size_class_bytes} folder
    block_file = inner_path / str(8 * 1024 * 1024)
    assert block_file.is_file()

    # Check file contents
    expected_contents = (
        write_bytes
        + b"\x00"
        + (255 - len(write_bytes)) * b"\xcb"
        + (8 * 1024 * 1024 - 256) * b"\x00"
    )
    assert block_file.read_bytes() == expected_contents

    # Pin back
    pool.pin(allocation)
    contents_after_pin_back = allocation.read_bytes(6 * 1024 * 1024)
    assert contents_after_pin_back == write_bytes

    pool.free(allocation)

    pool.cleanup()
    del pool


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="malloc_trim is a Linux only feature"
)
def test_malloc_trim():
    """
    Test that the malloc_trim functionality works as expected.
    """
    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=16,
        min_size_class=256,
        max_num_size_classes=5,
        # Set a 1MiB threshold for malloc_trim being called
        malloc_free_trim_threshold=1024 * 1024,
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Make allocations through malloc (<192KiB)
    allocation1: BufferPoolAllocation = pool.allocate(150 * 1024)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0
    allocation2: BufferPoolAllocation = pool.allocate(150 * 1024)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0
    allocation3: BufferPoolAllocation = pool.allocate(150 * 1024)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0
    allocation4: BufferPoolAllocation = pool.allocate(150 * 1024)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0
    allocation5: BufferPoolAllocation = pool.allocate(150 * 1024)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0
    allocation6: BufferPoolAllocation = pool.allocate(150 * 1024)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0
    allocation7: BufferPoolAllocation = pool.allocate(150 * 1024)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0
    allocation8: BufferPoolAllocation = pool.allocate(150 * 1024)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0

    # Make allocations through mmap (>192KiB) -- we will use them to
    # verify that bytes_freed_through_malloc_since_last_trim doesn't
    # increase when non-malloc allocations are freed.
    non_malloc_allocation1: BufferPoolAllocation = pool.allocate(512 * 1024)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0
    non_malloc_allocation2: BufferPoolAllocation = pool.allocate(512 * 1024)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0
    non_malloc_allocation3: BufferPoolAllocation = pool.allocate(512 * 1024)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0

    # Start free-ing the allocations and check that
    # bytes_freed_through_malloc_since_last_trim is as expected after each.
    # It should reset after 1MiB worth of malloc allocations.
    pool.free(allocation1)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 150 * 1024
    pool.free(allocation2)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 300 * 1024
    pool.free(allocation3)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 450 * 1024
    # Free-ing a non-malloc allocation shouldn't increase
    # bytes_freed_through_malloc_since_last_trim
    pool.free(non_malloc_allocation1)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 450 * 1024
    pool.free(allocation4)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 600 * 1024
    pool.free(allocation5)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 750 * 1024
    pool.free(allocation6)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 900 * 1024
    pool.free(non_malloc_allocation2)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 900 * 1024
    # This free will take it over 1MiB, so bytes_freed_through_malloc_since_last_trim
    # should get reset to 0.
    pool.free(allocation7)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0
    pool.free(non_malloc_allocation3)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 0
    # It should go up again now.
    pool.free(allocation8)
    assert pool.bytes_freed_through_malloc_since_last_trim() == 150 * 1024

    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    del pool


def test_sparse_file_spill_cleanup_overload(tmp_path, capfd):
    """
    Test that deletes for sparse file storage managers
    are only called when we need to spill more data
    than what is openly available
    """

    # Allocate 1MiB in Pool, 3MiB in storage for testing
    local_opt = StorageOptions(
        usable_size=3 * 1024 * 1024, tracing_mode=True, location=bytes(tmp_path)
    )
    options: BufferPoolOptions = BufferPoolOptions(
        memory_size=1,
        min_size_class=512,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
        spill_on_unpin=True,
        trace_level=2,
        storage_options=[local_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Allocate 3 1MiB frames and immediately spill
    allocation_1: BufferPoolAllocation = pool.allocate(1024 * 1024)
    pool.unpin(allocation_1)
    allocation_2: BufferPoolAllocation = pool.allocate(1024 * 1024)
    pool.unpin(allocation_2)
    allocation_3: BufferPoolAllocation = pool.allocate(1024 * 1024)
    pool.unpin(allocation_3)

    # Free all 3
    pool.free(allocation_1)
    pool.free(allocation_2)
    pool.free(allocation_3)

    # Allocate and spill a 512KiB frame
    # Expect a delete operation to occur to make space
    allocation_4: BufferPoolAllocation = pool.allocate(512 * 1024)
    pool.unpin(allocation_4)

    pool.cleanup()
    del pool

    # Check if delete action occured
    _, err = capfd.readouterr()
    errlines: list[str] = err.split("\n")
    errlines = [" ".join(e.split()) for e in errlines]
    assert "Total Num Delete Calls │ 1" in errlines


def test_sparse_file_spill_cleanup_threshold(tmp_path, capfd):
    """
    Test that deletes for sparse file storage managers
    are only called when the threshold is reached
    """

    # Allocate 1MiB in Pool, 3MiB in storage for testing
    local_opt = StorageOptions(
        usable_size=3 * 1024 * 1024, tracing_mode=True, location=bytes(tmp_path)
    )
    options: BufferPoolOptions = BufferPoolOptions(
        memory_size=1,
        min_size_class=1024,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
        spill_on_unpin=True,
        trace_level=2,
        storage_options=[local_opt],
    )

    # Delete leftover blocks if # per size class >= 2
    with temp_env_override({"BODO_BUFFER_POOL_SPARSE_DELETE_THRESHOLD": "4"}):
        pool: BufferPool = BufferPool.from_options(options)

    # Allocate 2 1MiB frames and immediately spill
    allocation_1: BufferPoolAllocation = pool.allocate(1024 * 1024)
    pool.unpin(allocation_1)
    allocation_2: BufferPoolAllocation = pool.allocate(1024 * 1024)
    pool.unpin(allocation_2)

    # Free both
    pool.free(allocation_1)
    pool.free(allocation_2)

    pool.cleanup()
    del pool

    # Check if delete action occurred
    _, err = capfd.readouterr()
    errlines: list[str] = err.split("\n")
    errlines = [" ".join(e.split()) for e in errlines]
    assert "Total Num Delete Calls │ 1" in errlines


@pytest.mark.s3
def test_spill_to_s3(tmp_s3_path: str):
    """
    Test that data spilled to S3 and read back is
    correctly preserved.
    """

    s3_opt = StorageOptions(
        usable_size=-1, storage_type=1, location=tmp_s3_path.encode()
    )
    options: BufferPoolOptions = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        spill_on_unpin=True,
        enforce_max_limit_during_allocation=True,
        storage_options=[s3_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Make an allocation that goes through a SizeClass
    write_bytes = b"Hello BufferPool from Bodo!"
    allocation_1: BufferPoolAllocation = pool.allocate(3.5 * 1024 * 1024)
    allocation_1.write_bytes(write_bytes)

    # Unpin it
    pool.unpin(allocation_1)

    # Verify that it's been spilled to S3 and contents are correct
    assert pool.bytes_pinned() == 0
    assert pool.bytes_in_memory() == 0
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert not allocation_1.is_nullptr() and not allocation_1.is_in_memory()

    # Look for spilled file in expected location
    # First Folder Expected to Have Prepended Rank
    fs = s3fs.S3FileSystem()
    paths = fs.glob(tmp_s3_path + str(bodo.get_rank()) + "-*")
    assert len(paths) == 1
    inner_path: str = paths[0]  # type: ignore
    # Block File should be in {size_class_bytes} folder
    file_path = inner_path + "/" + str(4 * 1024 * 1024) + "/" + "0"

    # Check file contents
    expected_contents = (
        write_bytes
        + b"\x00"
        + (255 - len(write_bytes)) * b"\xcb"
        + (4 * 1024 * 1024 - 256) * b"\x00"
    )
    assert fs.cat_file(file_path) == expected_contents

    # Pin it back and verify contents
    pool.pin(allocation_1)
    contents_after_pin_back = allocation_1.read_bytes(int(3.5 * 1024 * 1024))
    assert contents_after_pin_back == write_bytes

    pool.free(allocation_1)
    pool.cleanup()
    del pool


@pytest.mark.s3
def test_spill_to_s3_empty(tmp_s3_path: str):
    """
    Ensure that cleaning / deleting an unused S3 storage location
    does not fail
    """

    s3_opt = StorageOptions(
        usable_size=-1, storage_type=1, location=tmp_s3_path.encode()
    )
    options: BufferPoolOptions = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
        storage_options=[s3_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)
    pool.cleanup()
    del pool


def test_spill_to_azure(abfs_fs, tmp_abfs_path: str):
    """
    Test that data spilled to ABFS and read back is
    correctly preserved.
    """

    azure_opt = StorageOptions(
        usable_size=-1, storage_type=2, location=tmp_abfs_path.encode()
    )
    options: BufferPoolOptions = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        spill_on_unpin=True,
        enforce_max_limit_during_allocation=True,
        storage_options=[azure_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)

    # Make an allocation that goes through a SizeClass
    write_bytes = b"Hello BufferPool from Bodo!"
    allocation_1: BufferPoolAllocation = pool.allocate(3.5 * 1024 * 1024)
    allocation_1.write_bytes(write_bytes)

    # Unpin it
    pool.unpin(allocation_1)

    # Verify that it's been spilled to S3 and contents are correct
    assert pool.bytes_pinned() == 0
    assert pool.bytes_in_memory() == 0
    assert pool.bytes_allocated() == 4 * 1024 * 1024
    assert pool.max_memory() == 4 * 1024 * 1024
    assert not allocation_1.is_nullptr() and not allocation_1.is_in_memory()

    # Look for spilled file in expected location
    # First Folder Expected to Have Prepended Rank
    paths = abfs_fs.glob(tmp_abfs_path + str(bodo.get_rank()) + "-*")
    assert len(paths) == 1
    inner_path: str = paths[0]  # type: ignore
    # Block File should be in {size_class_bytes} folder
    file_path = inner_path + "/" + str(4 * 1024 * 1024) + "/" + "0"

    # Check file contents
    expected_contents = (
        write_bytes
        + b"\x00"
        + (255 - len(write_bytes)) * b"\xcb"
        + (4 * 1024 * 1024 - 256) * b"\x00"
    )
    assert abfs_fs.cat_file(file_path) == expected_contents

    # Pin it back and verify contents
    pool.pin(allocation_1)
    contents_after_pin_back = allocation_1.read_bytes(int(3.5 * 1024 * 1024))
    assert contents_after_pin_back == write_bytes

    pool.free(allocation_1)
    pool.cleanup()
    del pool


def test_spill_to_azure_empty(tmp_abfs_path: str):
    """
    Ensure that cleaning / deleting an unused Azure storage location
    does not fail
    """

    s3_opt = StorageOptions(
        usable_size=-1, storage_type=2, location=tmp_abfs_path.encode()
    )
    options: BufferPoolOptions = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
        storage_options=[s3_opt],
    )
    pool: BufferPool = BufferPool.from_options(options)
    pool.cleanup()
    del pool


@pytest.mark.s3
def test_print_spilling_metrics(capfd, tmp_path: Path, tmp_s3_path: str):
    """
    Test tracing spilling metrics are correctly enabled
    and printed out in tracing mode
    """

    local_opt = StorageOptions(
        usable_size=5 * 1024 * 1024, location=bytes(tmp_path), tracing_mode=True
    )
    s3_opt = StorageOptions(
        usable_size=-1, storage_type=1, location=tmp_s3_path.encode(), tracing_mode=True
    )

    options: BufferPoolOptions = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        spill_on_unpin=True,
        enforce_max_limit_during_allocation=True,
        storage_options=[local_opt, s3_opt],
        trace_level=2,
    )
    pool: BufferPool = BufferPool.from_options(options)

    allocation_1: BufferPoolAllocation = pool.allocate(3.5 * 1024 * 1024)
    pool.unpin(allocation_1)
    allocation_2 = pool.allocate(4 * 1024 * 1024)
    pool.unpin(allocation_2)
    allocation_3 = pool.allocate(2.5 * 1024 * 1024)
    pool.unpin(allocation_3)

    pool.free(allocation_1)
    pool.pin(allocation_2)
    pool.free(allocation_2)
    pool.pin(allocation_3)
    pool.unpin(allocation_3)

    pool.cleanup()
    _, err = capfd.readouterr()
    errlines: list[str] = err.split("\n")
    errlines = [" ".join(e.split()) for e in errlines]

    # Top Level Pool Stats
    assert "Curr Bytes Allocated: 4.0MiB" in errlines
    assert "Curr Bytes In Memory: 0 bytes" in errlines
    assert "Curr Bytes Malloced: 0 bytes" in errlines
    assert "Curr Bytes Pinned: 0 bytes" in errlines
    assert "Curr Num Allocations: 2" in errlines

    assert "Total Num Allocations: 3" in errlines
    assert "Total Num Reallocs: 0" in errlines
    assert "Total Num Pins: 2" in errlines
    assert "Total Num Unpins: 4" in errlines
    assert "Total Num Frees from Spill: 1" in errlines
    assert "Total Num Reallocs Reused: 0" in errlines

    assert "Total Bytes Allocated: 12.0MiB" in errlines
    assert "Total Bytes Requested: 10.0MiB" in errlines
    assert "Total Bytes Malloced: 0 bytes" in errlines
    assert "Total Bytes Pinned: 20.0MiB" in errlines
    assert "Total Bytes Unpinned: 0 bytes" in errlines
    assert "Total Bytes Reused in Realloc: 0 bytes" in errlines

    assert "Peak Bytes Allocated: 12.0MiB" in errlines
    assert "Peak Bytes In Memory: 4.0MiB" in errlines
    assert "Peak Bytes Malloced: 0 bytes" in errlines
    assert "Peak Bytes Pinned: 4.0MiB" in errlines

    # Spilling metrics per SizeClass
    size_class_spill_count: str = next(l for l in errlines if l.startswith("4.0MiB"))
    assert bool(
        re.match(
            r"4\.0MiB │ 4 │ [0-9.ms]+ │ 2 │ [0-9.ms]+ │ 5 │ [0-9.ms]+ │ [0-9.ms]+",
            size_class_spill_count,
        )
    )
    for class_name in [
        "8KiB",
        "16KiB",
        "32KiB",
        "64KiB",
        "128KiB",
        "256KiB",
        "512KiB",
        "1.0MiB",
        "2.0MiB",
    ]:
        size_class_spill_count: str = next(
            l for l in errlines if l.startswith(class_name)
        )
        size_class_spill_count = size_class_spill_count[(len(class_name) + 3) :]
        assert bool(
            re.match(
                r"0 │ [0-9.ms]+ │ 0 │ [0-9.ms]+ │ 0 │ [0-9.ms]+ │ [0-9.ms]+",
                size_class_spill_count,
            )
        )

    # Spilling metrics per StorageManager
    curr_spilled_bytes: str = next(l for l in errlines if "Current Spilled Bytes" in l)
    assert curr_spilled_bytes == "Current Spilled Bytes │ 4.0MiB │ 0 bytes"
    curr_blocks_spilled: str = next(
        l for l in errlines if "Current Blocks Spilled" in l
    )
    assert curr_blocks_spilled == "Current Blocks Spilled │ 1 │ 0"

    total_blocks_spilled: str = next(l for l in errlines if "Total Blocks Spilled" in l)
    assert total_blocks_spilled == "Total Blocks Spilled │ 2 │ 2"
    total_blocks_read: str = next(l for l in errlines if "Total Blocks Read" in l)
    assert total_blocks_read == "Total Blocks Read │ 0 │ 2"
    total_blocks_deleted: str = next(
        l for l in errlines if "Total Num Delete Calls" in l
    )
    assert total_blocks_deleted == "Total Num Delete Calls │ 0 │ 2"

    total_bytes_spilled: str = next(l for l in errlines if "Total Bytes Spilled" in l)
    assert total_bytes_spilled == "Total Bytes Spilled │ 8.0MiB │ 8.0MiB"
    total_bytes_read: str = next(l for l in errlines if "Total Bytes Read" in l)
    assert total_bytes_read == "Total Bytes Read │ 0 bytes │ 8.0MiB"
    total_bytes_deleted: str = next(l for l in errlines if "Total Bytes Deleted" in l)
    assert total_bytes_deleted == "Total Bytes Deleted │ 0 bytes │ 8.0MiB"

    max_spilled_bytes: str = next(l for l in errlines if "Max Spilled Bytes" in l)
    assert max_spilled_bytes == "Max Spilled Bytes │ 4.0MiB │ 8.0MiB"


## OperatorBufferPool tests


def test_operator_pool_attributes():
    """
    Verify that the operator pool attributes are
    initialized correctly (including the
    defaults).
    """

    # Create one with default BufferPool as its parent
    pool = BufferPool.default()
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool, 0.4)
    assert op_pool.parent_pool == pool
    assert op_pool.backend_name == pool.backend_name
    assert op_pool.operator_budget_bytes == 512 * 1024
    assert op_pool.error_threshold == 0.4
    assert op_pool.memory_error_threshold == int(512 * 1024 * 0.4)
    assert op_pool.threshold_enforcement_enabled  # default behavior
    del op_pool

    # Create one with a custom pool
    options = BufferPoolOptions(
        memory_size=8,
        min_size_class=8,
        max_num_size_classes=10,
        enforce_max_limit_during_allocation=True,
    )
    pool: BufferPool = BufferPool.from_options(options)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool, 0.8)
    assert op_pool.parent_pool == pool
    assert op_pool.backend_name == pool.backend_name
    assert op_pool.operator_budget_bytes == 512 * 1024
    assert op_pool.error_threshold == 0.8
    assert op_pool.memory_error_threshold == int(512 * 1024 * 0.8)
    assert op_pool.threshold_enforcement_enabled  # default behavior
    del op_pool
    del pool

    # Create one without specifying the pool and threshold (verify defaults)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024)
    assert op_pool.parent_pool == BufferPool.default()
    assert op_pool.backend_name == BufferPool.default().backend_name
    assert op_pool.operator_budget_bytes == 512 * 1024
    assert op_pool.error_threshold == 0.5
    assert op_pool.memory_error_threshold == int(512 * 1024 * 0.5)  # default is 0.5
    assert op_pool.threshold_enforcement_enabled  # default behavior
    del op_pool


def test_operator_pool_enable_disable_enforcement():
    """
    Test that threshold enforcement works as expected.
    Only tests the basic functionality. The more advanced
    functionality and edge cases are tested in the
    other tests.
    """

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024)

    # Verify default
    assert op_pool.threshold_enforcement_enabled

    # Check that disabling works as expected:
    op_pool.disable_threshold_enforcement()
    assert not op_pool.threshold_enforcement_enabled

    # Check that turning it back on works as expected:
    op_pool.enable_threshold_enforcement()
    assert op_pool.threshold_enforcement_enabled

    # Test that if number of pinned bytes is above error
    # threshold (256KiB) and we try to enable enforcement, it errors out:
    op_pool.disable_threshold_enforcement()  # Disable temporarily
    allocation: BufferPoolAllocation = op_pool.allocate(
        300 * 1024
    )  # Will go through while enforcement is disabled
    with pytest.raises(
        RuntimeError,
        match="OperatorPoolThresholdExceededError: Tried allocating more space than what's allowed to be pinned!",
    ):
        op_pool.enable_threshold_enforcement()
    assert not op_pool.threshold_enforcement_enabled

    op_pool.free(allocation)

    # Allocating 300KiB through the scratch portion and
    # 100KiB through the main portion should not raise
    # the exception
    allocation1 = op_pool.allocate_scratch(300 * 1024)
    allocation2 = op_pool.allocate_scratch(100 * 1024)
    op_pool.enable_threshold_enforcement()
    assert op_pool.threshold_enforcement_enabled

    op_pool.free(allocation1)
    op_pool.free(allocation2)
    del op_pool


def test_operator_pool_set_threshold():
    """
    Test that setting the threshold works as expected.
    """

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024)

    # Verify defaults
    assert op_pool.error_threshold == 0.5
    assert op_pool.memory_error_threshold == 256 * 1024

    # Verify that OperatorPoolThresholdExceededError is raised
    # when trying to allocate 300KiB
    with pytest.raises(
        RuntimeError,
        match="OperatorPoolThresholdExceededError: Tried allocating more space than what's allowed to be pinned!",
    ):
        op_pool.allocate(300 * 1024)

    # Change the threshold ratio to 0.8
    op_pool.set_error_threshold(0.8)

    # Verify attributes
    assert op_pool.error_threshold == 0.8
    assert op_pool.memory_error_threshold == int(512 * 1024 * 0.8)

    # Trying to allocate 300KiB should go through now:
    allocation: BufferPoolAllocation = op_pool.allocate(300 * 1024)

    # Trying to set the threshold back to 0.5 should now raise
    # an OperatorPoolThresholdExceededError.
    with pytest.raises(
        RuntimeError,
        match="OperatorPoolThresholdExceededError: Tried allocating more space than what's allowed to be pinned!",
    ):
        op_pool.set_error_threshold(0.5)

    # Verify no change to attributes
    assert op_pool.error_threshold == 0.8
    assert op_pool.memory_error_threshold == int(512 * 1024 * 0.8)

    # Free allocation
    op_pool.free(allocation)
    assert op_pool.bytes_pinned() == 0
    assert op_pool.main_mem_bytes_pinned() == 0
    assert op_pool.scratch_mem_bytes_pinned() == 0

    # Verify that we can set the threshold back to 0.5 now.
    op_pool.set_error_threshold(0.5)

    # Verify attributes
    assert op_pool.error_threshold == 0.5
    assert op_pool.memory_error_threshold == 256 * 1024
    assert op_pool.bytes_pinned() == 0
    assert op_pool.bytes_allocated() == 0

    # Cleanup
    del op_pool


def test_operator_pool_set_budget():
    """
    Test that updating/reducing the budget works as expected.
    """

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024)

    # Verify defaults
    assert op_pool.operator_budget_bytes == 512 * 1024
    assert op_pool.error_threshold == 0.5
    assert op_pool.memory_error_threshold == 256 * 1024

    # Trying to increase budget should raise an exception
    with pytest.raises(
        RuntimeError,
        match="OperatorBufferPool::SetBudget: Increasing the budget is not supported through this API.",
    ):
        op_pool.set_budget(1024 * 1024)

    # Reduce the budget and verify that the attributes are as expected
    op_pool.set_budget(500 * 1024)
    assert op_pool.operator_budget_bytes == 500 * 1024
    assert op_pool.error_threshold == 0.5
    assert op_pool.memory_error_threshold == 250 * 1024

    # If threshold enforcement is enabled and more bytes are pinned
    # than new memory error threshold, it should raise
    # OperatorPoolThresholdExceededError and not update budget.
    allocation: BufferPoolAllocation = op_pool.allocate(225 * 1024)
    assert op_pool.threshold_enforcement_enabled
    with pytest.raises(
        RuntimeError,
        match="OperatorPoolThresholdExceededError: Tried allocating more space than what's allowed to be pinned!",
    ):
        op_pool.set_budget(400 * 1024)
    assert op_pool.operator_budget_bytes == 500 * 1024
    assert op_pool.error_threshold == 0.5
    assert op_pool.memory_error_threshold == 250 * 1024

    # The same should be fine to do if threshold enforcement is disabled.
    op_pool.disable_threshold_enforcement()
    op_pool.set_budget(400 * 1024)
    assert op_pool.operator_budget_bytes == 400 * 1024
    assert op_pool.error_threshold == 0.5
    assert op_pool.memory_error_threshold == 200 * 1024

    # Cleanup
    op_pool.free(allocation)
    assert op_pool.bytes_pinned() == 0
    assert op_pool.bytes_allocated() == 0
    op_pool.enable_threshold_enforcement()
    assert op_pool.threshold_enforcement_enabled

    # Allocate through the scratch portion. Trying to reduce
    # the budget below this amount should raise an exception:
    allocation = op_pool.allocate_scratch(300 * 1024)
    with pytest.raises(
        RuntimeError,
        match="OperatorPoolThresholdExceededError: Tried allocating more space than what's allowed to be pinned!",
    ):
        op_pool.set_budget(200 * 1024)
    assert op_pool.operator_budget_bytes == 400 * 1024
    assert op_pool.error_threshold == 0.5
    assert op_pool.memory_error_threshold == 200 * 1024

    # The same should be fine to do when threshold enforcement is disabled
    op_pool.disable_threshold_enforcement()
    op_pool.set_budget(200 * 1024)
    assert op_pool.operator_budget_bytes == 200 * 1024
    assert op_pool.error_threshold == 0.5
    assert op_pool.memory_error_threshold == 100 * 1024

    # Cleanup
    op_pool.free(allocation)
    assert op_pool.bytes_pinned() == 0
    assert op_pool.bytes_allocated() == 0

    del op_pool


def test_operator_pool_allocation():
    """
    Test multiple allocation/free scenarios, including
    ensuring that threshold enforcement and limit
    enforcement work as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=8,
        enforce_max_limit_during_allocation=True,
        debug_mode=True,
    )  # 4MiB, 8KiB
    pool: BufferPool = BufferPool.from_options(options)

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool, 0.5)

    ##
    ## ------ 1: Basic allocation test (allocate and then free) ------ ##
    ##

    allocation: BufferPoolAllocation = op_pool.allocate(5 * 1024)

    # Verify that default allocation is 64B aligned
    assert allocation.alignment == 64
    assert allocation.get_ptr_as_int() % 64 == 0

    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 5 * 1024
    assert op_pool.bytes_pinned() == 5 * 1024
    assert op_pool.max_memory() == 5 * 1024
    assert pool.bytes_allocated() == 5 * 1024
    assert pool.bytes_pinned() == 5 * 1024
    assert pool.max_memory() == 5 * 1024

    # Free and verify stats
    op_pool.free(allocation)
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    # Max memory should still be the same
    assert op_pool.max_memory() == 5 * 1024
    assert pool.max_memory() == 5 * 1024

    ##
    ## --------------- 2: Allocate, unpin and then free -------------- ##
    ##

    # Allocate 10KiB. Anything above 6KiB will go through
    # the SizeClasses, so we'll be able to actually pin and unpin
    allocation: BufferPoolAllocation = op_pool.allocate(10 * 1024)

    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 10 * 1024
    assert op_pool.bytes_pinned() == 10 * 1024
    assert op_pool.max_memory() == 10 * 1024
    assert pool.bytes_allocated() == 16 * 1024  # Comes from the 16KiB SizeClass
    assert pool.bytes_pinned() == 16 * 1024
    assert pool.max_memory() == 16 * 1024
    # Verify through the parent pool that it is pinned
    assert pool.is_pinned(allocation)

    op_pool.unpin(allocation)

    # Verify stats after unpinning
    assert op_pool.bytes_allocated() == 10 * 1024
    assert op_pool.bytes_pinned() == 0
    assert op_pool.max_memory() == 10 * 1024
    assert pool.bytes_allocated() == 16 * 1024
    assert pool.bytes_pinned() == 0
    assert pool.max_memory() == 16 * 1024
    # Verify through the parent pool that it is indeed unpinned
    assert not pool.is_pinned(allocation)

    # Free and verify stats
    op_pool.free(allocation)
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.max_memory() == 10 * 1024
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    assert pool.max_memory() == 16 * 1024

    ##
    ## --- 3: Same as (2), but where allocation goes through malloc -- ##
    ##

    # Allocate 4KiB. Anything below 6KiB will go through
    # malloc.
    allocation: BufferPoolAllocation = op_pool.allocate(4 * 1024)

    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 4 * 1024
    assert op_pool.bytes_pinned() == 4 * 1024
    assert op_pool.max_memory() == 10 * 1024
    assert pool.bytes_allocated() == 4 * 1024
    assert pool.bytes_pinned() == 4 * 1024
    assert pool.max_memory() == 16 * 1024
    # Verify through the parent pool that it is pinned
    assert pool.is_pinned(allocation)

    op_pool.unpin(allocation)

    # Verify stats after unpinning
    assert op_pool.bytes_allocated() == 4 * 1024
    assert pool.bytes_allocated() == 4 * 1024
    # Still 4KiB since malloc allocations cannot be unpinned
    assert op_pool.bytes_pinned() == 4 * 1024
    assert op_pool.max_memory() == 10 * 1024
    assert pool.bytes_pinned() == 4 * 1024
    assert pool.max_memory() == 16 * 1024
    # Verify through the parent pool that it is indeed still pinned
    assert pool.is_pinned(allocation)

    # Free and verify stats
    op_pool.free(allocation)
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.max_memory() == 10 * 1024
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    assert pool.max_memory() == 16 * 1024

    ##
    ## ----- 4: Allocate small, then try to allocate amount that ----- ##
    ## -----    would cross threshold (w/ and w/o threshold      ----- ##
    ## -----    enforcement)                                     ----- ##
    ##

    allocation1: BufferPoolAllocation = op_pool.allocate(50 * 1024)
    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 50 * 1024
    assert op_pool.bytes_pinned() == 50 * 1024
    assert op_pool.max_memory() == 50 * 1024
    assert pool.bytes_allocated() == 64 * 1024
    assert pool.bytes_pinned() == 64 * 1024
    assert pool.max_memory() == 64 * 1024
    # Verify through the parent pool that it is pinned
    assert pool.is_pinned(allocation1)

    # With threshold enforcement enabled:
    op_pool.enable_threshold_enforcement()
    assert op_pool.threshold_enforcement_enabled
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        # Total of 260KiB would be above the 256KiB threshold
        op_pool.allocate(210 * 1024)
    # Verify stats after allocation attempt
    assert op_pool.bytes_allocated() == 50 * 1024
    assert op_pool.bytes_pinned() == 50 * 1024
    assert op_pool.max_memory() == 50 * 1024
    assert pool.bytes_allocated() == 64 * 1024
    assert pool.bytes_pinned() == 64 * 1024
    assert pool.max_memory() == 64 * 1024
    # Verify through the parent pool that the first allocation is still pinned
    assert pool.is_pinned(allocation1)

    # Without threshold enforcement enabled:
    op_pool.disable_threshold_enforcement()
    assert not op_pool.threshold_enforcement_enabled

    allocation2: BufferPoolAllocation = op_pool.allocate(210 * 1024)
    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 260 * 1024
    assert op_pool.bytes_pinned() == 260 * 1024
    assert op_pool.max_memory() == 260 * 1024
    assert pool.bytes_allocated() == 320 * 1024
    assert pool.bytes_pinned() == 320 * 1024
    assert pool.max_memory() == 320 * 1024
    # Verify through the parent pool that both allocations are pinned
    assert pool.is_pinned(allocation1)
    assert pool.is_pinned(allocation2)

    # Free both
    op_pool.free(allocation1)
    op_pool.free(allocation2)

    # Verify stats
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.max_memory() == 260 * 1024
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    assert pool.max_memory() == 320 * 1024

    ##
    ## ----- 5: Allocate small, then try to allocate amount that ----- ##
    ## -----    would cross max limit (w/ and w/o threshold      ----- ##
    ## -----    enforcement)                                     ----- ##
    ##

    allocation1: BufferPoolAllocation = op_pool.allocate(220 * 1024)
    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 220 * 1024
    assert op_pool.bytes_pinned() == 220 * 1024
    assert op_pool.max_memory() == 260 * 1024
    assert pool.bytes_allocated() == 256 * 1024
    assert pool.bytes_pinned() == 256 * 1024
    assert pool.max_memory() == 320 * 1024
    # Verify through the parent pool that it is pinned
    assert pool.is_pinned(allocation1)

    # With threshold enforcement enabled:
    op_pool.enable_threshold_enforcement()
    assert op_pool.threshold_enforcement_enabled

    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        # This would take the total to 520KiB, which is higher than
        # 256KiB threshold
        op_pool.allocate(300 * 1024)

    # Verify stats after allocation attempt
    assert op_pool.bytes_allocated() == 220 * 1024
    assert op_pool.bytes_pinned() == 220 * 1024
    assert op_pool.max_memory() == 260 * 1024
    assert pool.bytes_allocated() == 256 * 1024
    assert pool.bytes_pinned() == 256 * 1024
    assert pool.max_memory() == 320 * 1024
    # Verify through the parent pool that the first allocation is still pinned
    assert pool.is_pinned(allocation1)

    # Without threshold enforcement enabled:
    op_pool.disable_threshold_enforcement()
    assert not op_pool.threshold_enforcement_enabled

    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match=re.escape("Not enough space in the buffer pool to allocate"),
    ):
        # This would take the total to over 4MiB, which is higher than
        # the BufferPool limit.
        op_pool.allocate(3900 * 1024)

    # Verify stats after allocation attempt
    assert op_pool.bytes_allocated() == 220 * 1024
    assert op_pool.bytes_pinned() == 220 * 1024
    assert op_pool.max_memory() == 260 * 1024
    assert pool.bytes_allocated() == 256 * 1024
    assert pool.bytes_pinned() == 256 * 1024
    assert pool.max_memory() == 320 * 1024
    # Verify through the parent pool that the first allocation is still pinned
    assert pool.is_pinned(allocation1)

    # Test the same through the allocate_scratch API
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match=re.escape(
            "Allocation canceled beforehand. Not enough space in the buffer pool"
        ),
    ):
        # This would take the total to over 4MiB, which is higher than
        # the BufferPool limit.
        op_pool.allocate_scratch(3900 * 1024)

    # Verify stats after allocation attempt
    assert op_pool.bytes_allocated() == 220 * 1024
    assert op_pool.bytes_pinned() == 220 * 1024
    assert op_pool.max_memory() == 260 * 1024
    assert pool.bytes_allocated() == 256 * 1024
    assert pool.bytes_pinned() == 256 * 1024
    assert pool.max_memory() == 320 * 1024
    # Verify through the parent pool that the first allocation is still pinned
    assert pool.is_pinned(allocation1)

    # Free and verify stats
    op_pool.free(allocation1)
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.max_memory() == 260 * 1024
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    assert pool.max_memory() == 320 * 1024

    del op_pool
    del pool


def test_operator_pool_pin_unpin():
    """
    Test multiple pin/unpin scenarios, including
    ensuring that threshold enforcement and limit
    enforcement work as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=2,
        min_size_class=8,
        enforce_max_limit_during_allocation=True,
        debug_mode=True,
    )  # 2MiB total, 8KiB min size class
    pool: BufferPool = BufferPool.from_options(options)

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool)

    ##
    ## ------------- 1: Allocate, then pin, then pin again ----------- ##
    ##

    # 20KiB is above malloc threshold (6KiB), so pin and unpin can be
    # tested properly.
    allocation: BufferPoolAllocation = op_pool.allocate(20 * 1024)

    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 20 * 1024
    assert op_pool.bytes_pinned() == 20 * 1024
    assert op_pool.max_memory() == 20 * 1024
    assert pool.bytes_allocated() == 32 * 1024
    assert pool.bytes_pinned() == 32 * 1024
    assert pool.max_memory() == 32 * 1024

    # Pin it
    op_pool.pin(allocation)

    # Verify stats after pinning
    assert op_pool.bytes_allocated() == 20 * 1024
    assert op_pool.bytes_pinned() == 20 * 1024
    assert op_pool.max_memory() == 20 * 1024
    assert pool.bytes_allocated() == 32 * 1024
    assert pool.bytes_pinned() == 32 * 1024
    assert pool.max_memory() == 32 * 1024

    # Now pin again
    op_pool.pin(allocation)

    # Verify stats after pinning
    assert op_pool.bytes_allocated() == 20 * 1024
    assert op_pool.bytes_pinned() == 20 * 1024
    assert op_pool.max_memory() == 20 * 1024
    assert pool.bytes_allocated() == 32 * 1024
    assert pool.bytes_pinned() == 32 * 1024
    assert pool.max_memory() == 32 * 1024

    # Free and verify stats
    op_pool.free(allocation)
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.max_memory() == 20 * 1024
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    assert pool.max_memory() == 32 * 1024

    ##
    ## -------------- 2: Allocate, then unpin, then pin -------------- ##
    ##

    allocation: BufferPoolAllocation = op_pool.allocate(20 * 1024)

    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 20 * 1024
    assert op_pool.bytes_pinned() == 20 * 1024
    assert op_pool.max_memory() == 20 * 1024
    assert pool.bytes_allocated() == 32 * 1024
    assert pool.bytes_pinned() == 32 * 1024
    assert pool.max_memory() == 32 * 1024

    # Unpin
    op_pool.unpin(allocation)
    # Verify stats after unpinning
    assert op_pool.bytes_allocated() == 20 * 1024
    assert op_pool.bytes_pinned() == 0
    assert op_pool.max_memory() == 20 * 1024
    assert pool.bytes_allocated() == 32 * 1024
    assert pool.bytes_pinned() == 0
    assert pool.max_memory() == 32 * 1024

    # Unpin again
    op_pool.unpin(allocation)
    # Verify stats after unpinning
    assert op_pool.bytes_allocated() == 20 * 1024
    assert op_pool.bytes_pinned() == 0
    assert op_pool.max_memory() == 20 * 1024
    assert pool.bytes_allocated() == 32 * 1024
    assert pool.bytes_pinned() == 0
    assert pool.max_memory() == 32 * 1024

    # Pin it back
    op_pool.pin(allocation)
    # Verify stats after pinning
    assert op_pool.bytes_allocated() == 20 * 1024
    assert op_pool.bytes_pinned() == 20 * 1024
    assert op_pool.max_memory() == 20 * 1024
    assert pool.bytes_allocated() == 32 * 1024
    assert pool.bytes_pinned() == 32 * 1024
    assert pool.max_memory() == 32 * 1024

    # Unpin it again
    op_pool.unpin(allocation)
    # Verify stats after unpinning
    assert op_pool.bytes_allocated() == 20 * 1024
    assert op_pool.bytes_pinned() == 0
    assert op_pool.max_memory() == 20 * 1024
    assert pool.bytes_allocated() == 32 * 1024
    assert pool.bytes_pinned() == 0
    assert pool.max_memory() == 32 * 1024

    # Free and verify stats
    op_pool.free(allocation)
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.max_memory() == 20 * 1024
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    assert pool.max_memory() == 32 * 1024

    ##
    ## ---- 3: Allocate moderate size, then try to allocate amount --- ##
    ## ----    such that both together are not below threshold     --- ##
    ## ----    but after unpinning 1st allocation, the second is.  --- ##
    ## ----    Try pinning the 1st allocation and verify behavior. --- ##
    ##

    # 110 + 160 is above the threshold (256KiB), but they individually are not.

    allocation1: BufferPoolAllocation = op_pool.allocate(110 * 1024)
    op_pool.enable_threshold_enforcement()
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        op_pool.allocate(160 * 1024)

    # Verify stats after both allocation attempts
    assert op_pool.bytes_allocated() == 110 * 1024
    assert op_pool.bytes_pinned() == 110 * 1024
    assert op_pool.max_memory() == 110 * 1024
    assert pool.bytes_allocated() == 128 * 1024
    assert pool.bytes_pinned() == 128 * 1024
    assert pool.max_memory() == 128 * 1024

    # Unpin allocation and try again:
    op_pool.unpin(allocation1)
    allocation2: BufferPoolAllocation = op_pool.allocate(160 * 1024)
    assert op_pool.bytes_allocated() == 270 * 1024
    assert op_pool.bytes_pinned() == 160 * 1024
    assert op_pool.max_memory() == 270 * 1024
    assert pool.bytes_allocated() == 384 * 1024
    assert pool.bytes_pinned() == 256 * 1024
    assert pool.max_memory() == 384 * 1024

    # Trying to pin allocation1 back should error out
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        op_pool.pin(allocation1)

    # Verify stats after pin attempt
    assert op_pool.bytes_allocated() == 270 * 1024
    assert op_pool.bytes_pinned() == 160 * 1024
    assert op_pool.max_memory() == 270 * 1024
    assert pool.bytes_allocated() == 384 * 1024
    assert pool.bytes_pinned() == 256 * 1024
    assert pool.max_memory() == 384 * 1024

    # Try allocating another within threshold
    allocation3: BufferPoolAllocation = op_pool.allocate(90 * 1024)

    assert op_pool.bytes_allocated() == 360 * 1024
    assert op_pool.bytes_pinned() == 250 * 1024
    assert op_pool.max_memory() == 360 * 1024
    assert pool.bytes_allocated() == 512 * 1024
    assert pool.bytes_pinned() == 384 * 1024
    assert pool.max_memory() == 512 * 1024

    ##
    ## ---- 4: Verify that we can allocate as much as parent pool ---- ##
    ## ----    allows, as long as pinned amount is below limit.   ---- ##
    ##

    op_pool.enable_threshold_enforcement()

    # Continue allocating and unpinning (to keep pinned amount below 256KiB)
    # and verify that we can go up to 2MiB (parent pool size -- we can technically
    # go higher, but we haven't set up spilling)

    # Unpin allocation2
    op_pool.unpin(allocation2)
    assert op_pool.bytes_allocated() == 360 * 1024
    assert op_pool.bytes_pinned() == 90 * 1024
    assert op_pool.max_memory() == 360 * 1024
    assert pool.bytes_allocated() == 512 * 1024
    assert pool.bytes_pinned() == 128 * 1024
    assert pool.max_memory() == 512 * 1024

    # Allocate another
    allocation4: BufferPoolAllocation = op_pool.allocate(110 * 1024)
    assert op_pool.bytes_allocated() == 470 * 1024
    assert op_pool.bytes_pinned() == 200 * 1024
    assert pool.bytes_allocated() == 640 * 1024
    assert pool.bytes_pinned() == 256 * 1024

    # Go up to 2MiB
    op_pool.unpin(allocation3)
    op_pool.unpin(allocation4)
    allocation5: BufferPoolAllocation = op_pool.allocate(160 * 1024)
    op_pool.unpin(allocation5)
    assert op_pool.bytes_allocated() == 630 * 1024
    assert op_pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 896 * 1024
    assert pool.bytes_pinned() == 0
    allocation6: BufferPoolAllocation = op_pool.allocate(240 * 1024)
    op_pool.unpin(allocation6)
    allocation7: BufferPoolAllocation = op_pool.allocate(240 * 1024)
    op_pool.unpin(allocation7)
    allocation8: BufferPoolAllocation = op_pool.allocate(240 * 1024)
    op_pool.unpin(allocation8)
    allocation9: BufferPoolAllocation = op_pool.allocate(240 * 1024)
    op_pool.unpin(allocation9)
    assert op_pool.bytes_allocated() == 1590 * 1024
    assert op_pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 1920 * 1024
    assert pool.bytes_pinned() == 0

    # Check that error is raised when trying over 2MiB total
    # (at the BufferPool level since it uses frame sizes and not
    # just the actual allocation amounts)
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match="Spilling is not available to free up sufficient space in memory",
    ):
        op_pool.allocate(240 * 1024)

    # Try pinning existing allocations and it should raise threshold error
    op_pool.pin(allocation1)
    assert op_pool.bytes_allocated() == 1590 * 1024
    assert op_pool.bytes_pinned() == 110 * 1024
    assert pool.bytes_allocated() == 1920 * 1024
    assert pool.bytes_pinned() == 128 * 1024

    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        op_pool.pin(allocation2)

    # Verify stats after pin attempt
    assert op_pool.bytes_allocated() == 1590 * 1024
    assert op_pool.bytes_pinned() == 110 * 1024
    assert pool.bytes_allocated() == 1920 * 1024
    assert pool.bytes_pinned() == 128 * 1024

    # Disable threshold enforcement and try pinning again and it should succeed.
    op_pool.disable_threshold_enforcement()
    op_pool.pin(allocation2)
    assert op_pool.bytes_allocated() == 1590 * 1024
    assert op_pool.bytes_pinned() == 270 * 1024
    assert pool.bytes_allocated() == 1920 * 1024
    assert pool.bytes_pinned() == 384 * 1024

    # Pin all this way (up to 512KiB)
    op_pool.pin(allocation6)  # This takes it to 510KiB
    assert op_pool.bytes_allocated() == 1590 * 1024
    assert op_pool.bytes_pinned() == 510 * 1024
    assert pool.bytes_allocated() == 1920 * 1024
    assert pool.bytes_pinned() == 640 * 1024

    # This will succeed since threshold enforcement is off
    # and total would still be below total buffer pool size
    op_pool.pin(allocation3)

    assert op_pool.bytes_allocated() == 1590 * 1024
    assert op_pool.bytes_pinned() == 600 * 1024
    assert pool.bytes_allocated() == 1920 * 1024
    assert pool.bytes_pinned() == 768 * 1024

    # Then free all.
    op_pool.free(allocation1)
    op_pool.free(allocation2)
    op_pool.free(allocation3)
    op_pool.free(allocation4)
    op_pool.free(allocation5)
    op_pool.free(allocation6)
    op_pool.free(allocation7)
    op_pool.free(allocation8)
    op_pool.free(allocation9)
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0

    del op_pool
    del pool


def test_operator_pool_reallocate_basic():
    """
    Test basic reallocate functionality of the
    OperatorBufferPool.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=8,
        enforce_max_limit_during_allocation=True,
    )  # 4MiB, 8KiB
    pool: BufferPool = BufferPool.from_options(options)

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool, 0.5)

    ## Simple reallocate

    allocation: BufferPoolAllocation = op_pool.allocate(20 * 1024)

    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 20 * 1024
    assert op_pool.bytes_pinned() == 20 * 1024
    assert op_pool.max_memory() == 20 * 1024
    assert pool.bytes_allocated() == 32 * 1024
    assert pool.bytes_pinned() == 32 * 1024
    assert pool.max_memory() == 32 * 1024
    assert pool.is_pinned(allocation)

    op_pool.reallocate(40 * 1024, allocation)

    # Verify stats after re-allocation
    assert op_pool.bytes_allocated() == 40 * 1024
    assert op_pool.bytes_pinned() == 40 * 1024
    assert op_pool.max_memory() == 40 * 1024
    assert pool.bytes_allocated() == 64 * 1024
    assert pool.bytes_pinned() == 64 * 1024
    assert (
        pool.max_memory() == 96 * 1024
    )  # 64 + 32 since we needed both in memory at the same time
    assert pool.is_pinned(allocation)

    # Unpin and reallocate again
    op_pool.unpin(allocation)
    op_pool.reallocate(60 * 1024, allocation)
    assert op_pool.bytes_allocated() == 60 * 1024
    assert op_pool.bytes_pinned() == 60 * 1024
    assert op_pool.max_memory() == 60 * 1024
    assert pool.bytes_allocated() == 64 * 1024
    assert pool.bytes_pinned() == 64 * 1024
    assert pool.max_memory() == 96 * 1024
    assert pool.is_pinned(allocation)  # reallocate pins by default

    # Free and verify stats
    op_pool.free(allocation)
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.max_memory() == 60 * 1024
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    assert pool.max_memory() == 96 * 1024

    del op_pool
    del pool


def test_operator_pool_reallocate_edge_cases():
    """
    Test certain edge case scenarios of the reallocate
    functionality in OperatorBufferPool.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=8,
        enforce_max_limit_during_allocation=True,
        debug_mode=True,
    )  # 4MiB, 8KiB
    pool: BufferPool = BufferPool.from_options(options)

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool, 0.5)

    ## 1: Allocate within threshold, then unpin it. Then allocate another such
    ##    both together cannot be pinned. Trying to reallocate first allocation
    ##    (even to a lesser amount) should raise an error

    allocation1: BufferPoolAllocation = op_pool.allocate(200 * 1024)
    op_pool.unpin(allocation1)

    allocation2: BufferPoolAllocation = op_pool.allocate(200 * 1024)

    # Fails since 400KiB is above the threshold.
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        # 200 (from allocation2) + 30 is lower than threshold, but won't be allowed
        # since it still requires pinning the original 200KiB.
        op_pool.reallocate(30 * 1024, allocation1)

    assert op_pool.bytes_allocated() == 400 * 1024
    assert op_pool.bytes_pinned() == 200 * 1024
    assert pool.bytes_allocated() == 512 * 1024
    assert pool.bytes_pinned() == 256 * 1024
    assert pool.is_pinned(allocation2)
    assert not pool.is_pinned(allocation1)

    # Should pass when threshold enforcement is disabled:
    op_pool.disable_threshold_enforcement()
    op_pool.reallocate(30 * 1024, allocation1)
    assert op_pool.bytes_allocated() == 230 * 1024
    assert op_pool.bytes_pinned() == 230 * 1024
    assert pool.bytes_allocated() == 288 * 1024
    assert pool.bytes_pinned() == 288 * 1024
    assert pool.is_pinned(allocation2)
    assert pool.is_pinned(allocation1)

    op_pool.free(allocation1)
    op_pool.free(allocation2)
    op_pool.enable_threshold_enforcement()  # Reset for future tests
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0

    ## 2: Similar to (1), but check if old_size + new_size together would take it
    ##    above the threshold.
    allocation: BufferPoolAllocation = op_pool.allocate(120 * 1024)
    op_pool.enable_threshold_enforcement()
    # Fails since 150 + 120 is above the threshold.
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        op_pool.reallocate(150 * 1024, allocation)
    assert op_pool.bytes_allocated() == 120 * 1024
    assert op_pool.bytes_pinned() == 120 * 1024
    assert pool.bytes_allocated() == 128 * 1024
    assert pool.bytes_pinned() == 128 * 1024
    assert pool.is_pinned(allocation)

    op_pool.disable_threshold_enforcement()
    # Should go through now
    op_pool.reallocate(150 * 1024, allocation)
    assert op_pool.bytes_allocated() == 150 * 1024
    assert op_pool.bytes_pinned() == 150 * 1024
    assert pool.bytes_allocated() == 256 * 1024
    assert pool.bytes_pinned() == 256 * 1024
    assert pool.is_pinned(allocation)

    op_pool.free(allocation)
    op_pool.enable_threshold_enforcement()  # Reset for future tests
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0

    ## 3: Similar to (2), but check if old_size + new_size together would take it
    ##    above buffer pool size.

    # Need to disable for testing this code path:
    op_pool.disable_threshold_enforcement()
    allocation = op_pool.allocate(250 * 1024)
    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match=re.escape(
            "Allocation canceled beforehand. Not enough space in the buffer pool"
        ),
    ):
        op_pool.reallocate(int(3.8 * 1024 * 1024), allocation)
    assert op_pool.bytes_allocated() == 250 * 1024
    assert op_pool.bytes_pinned() == 250 * 1024
    assert pool.bytes_allocated() == 256 * 1024
    assert pool.bytes_pinned() == 256 * 1024
    assert pool.is_pinned(allocation)

    # Try with unpinned:
    op_pool.unpin(allocation)
    assert op_pool.bytes_allocated() == 250 * 1024
    assert op_pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 256 * 1024
    assert pool.bytes_pinned() == 0
    assert not pool.is_pinned(allocation)

    with pytest.raises(
        pyarrow.lib.ArrowMemoryError,
        match=re.escape(
            "Allocation canceled beforehand. Not enough space in the buffer pool"
        ),
    ):
        op_pool.reallocate(int(3.8 * 1024 * 1024), allocation)
    assert op_pool.bytes_allocated() == 250 * 1024
    assert op_pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 256 * 1024
    assert pool.bytes_pinned() == 0
    assert not pool.is_pinned(allocation)

    # Free and verify stats
    op_pool.free(allocation)
    op_pool.enable_threshold_enforcement()  # Reset for future tests
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0

    del op_pool
    del pool


def test_operator_pool_scratch_allocation():
    """
    Test that allocating memory from the scratch mem
    portion of the OperatorBufferPool works as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=4, min_size_class=8, enforce_max_limit_during_allocation=True
    )  # 4MiB, 8KiB
    pool: BufferPool = BufferPool.from_options(options)

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool, 0.5)

    ## 1. Basic test

    allocation: BufferPoolAllocation = op_pool.allocate_scratch(5 * 1024)

    # Verify that default allocation is 64B aligned
    assert allocation.alignment == 64
    assert allocation.get_ptr_as_int() % 64 == 0
    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 5 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 5 * 1024
    assert op_pool.main_mem_bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 5 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 5 * 1024
    assert op_pool.main_mem_bytes_pinned() == 0
    assert op_pool.max_memory() == 5 * 1024
    assert op_pool.main_mem_max_memory() == 0
    assert pool.bytes_allocated() == 5 * 1024
    assert pool.bytes_pinned() == 5 * 1024
    assert pool.max_memory() == 5 * 1024

    # Free and verify stats
    op_pool.free_scratch(allocation)
    assert op_pool.bytes_allocated() == 0
    assert op_pool.main_mem_bytes_allocated() == 0
    assert op_pool.scratch_mem_bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.main_mem_bytes_pinned() == 0
    assert op_pool.scratch_mem_bytes_pinned() == 0
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    # Max memory should still be the same
    assert op_pool.max_memory() == 5 * 1024
    assert op_pool.main_mem_max_memory() == 0
    assert pool.max_memory() == 5 * 1024

    ## 2. Allocate both main and scratch
    alloc_main: BufferPoolAllocation = op_pool.allocate(250 * 1024)
    alloc_scratch: BufferPoolAllocation = op_pool.allocate_scratch(250 * 1024)
    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 500 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 250 * 1024
    assert op_pool.main_mem_bytes_allocated() == 250 * 1024
    assert op_pool.bytes_pinned() == 500 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 250 * 1024
    assert op_pool.main_mem_bytes_pinned() == 250 * 1024
    assert op_pool.max_memory() == 500 * 1024
    assert op_pool.main_mem_max_memory() == 250 * 1024
    assert pool.bytes_allocated() == 512 * 1024
    assert pool.bytes_pinned() == 512 * 1024
    assert pool.max_memory() == 512 * 1024

    # Free and verify stats
    op_pool.free(alloc_main)
    op_pool.free_scratch(alloc_scratch)
    assert op_pool.bytes_allocated() == 0
    assert op_pool.scratch_mem_bytes_allocated() == 0
    assert op_pool.main_mem_bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.scratch_mem_bytes_pinned() == 0
    assert op_pool.main_mem_bytes_pinned() == 0
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    # Max memory should still be the same
    assert op_pool.max_memory() == 500 * 1024
    assert op_pool.main_mem_max_memory() == 250 * 1024
    assert pool.max_memory() == 512 * 1024

    ## 3. Verify that error is raised when trying to allocate more than
    ## total even when main mem is under threshold
    assert op_pool.threshold_enforcement_enabled
    alloc_main = op_pool.allocate(10 * 1024)
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        op_pool.allocate_scratch(510 * 1024)
    # Verify stats
    assert op_pool.bytes_allocated() == 10 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 0
    assert op_pool.main_mem_bytes_allocated() == 10 * 1024
    assert op_pool.bytes_pinned() == 10 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 0
    assert op_pool.main_mem_bytes_pinned() == 10 * 1024
    op_pool.free(alloc_main)

    ## 4. Same as (3), but the other way around
    alloc_scratch = op_pool.allocate_scratch(510 * 1024)
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        op_pool.allocate(10 * 1024)
    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 510 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 510 * 1024
    assert op_pool.main_mem_bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 510 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 510 * 1024
    assert op_pool.main_mem_bytes_pinned() == 0
    op_pool.free_scratch(alloc_scratch)

    # Cleanup
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    del op_pool
    del pool


def test_operator_pool_scratch_pin_unpin():
    """
    Test than pinning/unpinning from the scratch mem
    portion of the OperatorBufferPool works as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=2,
        min_size_class=8,
        enforce_max_limit_during_allocation=True,
    )  # 2MiB total, 8KiB min size class
    pool: BufferPool = BufferPool.from_options(options)

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool)

    # Allocate a moderate size, then try to allocate amount
    # such that both together are not below budget. But after
    # unpinning the 1st allocation, the second is. Try pinning
    # the 1st back and verify behavior. Then free the second
    # and pin 1st back and verify behavior.

    allocation1: BufferPoolAllocation = op_pool.allocate_scratch(220 * 1024)
    op_pool.enable_threshold_enforcement()
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        op_pool.allocate_scratch(320 * 1024)

    # Verify stats after both allocation attempts
    assert op_pool.bytes_allocated() == 220 * 1024
    assert op_pool.bytes_pinned() == 220 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 220 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 220 * 1024
    assert op_pool.max_memory() == 220 * 1024

    # Unpin allocation and try again:
    op_pool.unpin_scratch(allocation1)
    allocation2: BufferPoolAllocation = op_pool.allocate_scratch(320 * 1024)
    assert op_pool.bytes_allocated() == 540 * 1024
    assert op_pool.bytes_pinned() == 320 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 540 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 320 * 1024
    assert op_pool.max_memory() == 540 * 1024

    # Trying to pin allocation1 back should error out
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        op_pool.pin_scratch(allocation1)

    # Verify stats after pin attempt
    assert op_pool.bytes_allocated() == 540 * 1024
    assert op_pool.bytes_pinned() == 320 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 540 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 320 * 1024
    assert op_pool.max_memory() == 540 * 1024

    # Unpin the 2nd, and then pin the 1st. This should go through.
    op_pool.unpin_scratch(allocation2)
    op_pool.pin_scratch(allocation1)
    assert op_pool.bytes_allocated() == 540 * 1024
    assert op_pool.bytes_pinned() == 220 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 540 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 220 * 1024
    assert op_pool.max_memory() == 540 * 1024

    # Cleanup
    op_pool.free_scratch(allocation1)
    op_pool.free_scratch(allocation2)
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.scratch_mem_bytes_allocated() == 0
    assert op_pool.scratch_mem_bytes_pinned() == 0
    assert op_pool.max_memory() == 540 * 1024

    del op_pool
    del pool


def test_operator_pool_scratch_reallocate():
    """
    Test than reallocating from the scratch mem
    portion of the OperatorBufferPool works as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=8,
        enforce_max_limit_during_allocation=True,
    )  # 4MiB, 8KiB
    pool: BufferPool = BufferPool.from_options(options)

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool, 0.5)

    ## Simple reallocate

    allocation: BufferPoolAllocation = op_pool.allocate_scratch(40 * 1024)

    # Verify stats after allocation
    assert op_pool.bytes_allocated() == 40 * 1024
    assert op_pool.bytes_pinned() == 40 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 40 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 40 * 1024
    assert op_pool.max_memory() == 40 * 1024
    assert pool.is_pinned(allocation)

    op_pool.reallocate_scratch(80 * 1024, allocation)

    # Verify stats after re-allocation
    assert op_pool.bytes_allocated() == 80 * 1024
    assert op_pool.bytes_pinned() == 80 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 80 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 80 * 1024
    assert op_pool.max_memory() == 80 * 1024
    assert pool.is_pinned(allocation)

    # Unpin and reallocate again
    op_pool.unpin_scratch(allocation)
    op_pool.reallocate_scratch(120 * 1024, allocation)
    assert op_pool.bytes_allocated() == 120 * 1024
    assert op_pool.bytes_pinned() == 120 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 120 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 120 * 1024
    assert op_pool.max_memory() == 120 * 1024
    assert pool.is_pinned(allocation)  # reallocate pins by default

    # Free and verify stats
    op_pool.free_scratch(allocation)
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.scratch_mem_bytes_allocated() == 0
    assert op_pool.scratch_mem_bytes_pinned() == 0
    assert op_pool.max_memory() == 120 * 1024

    del op_pool
    del pool


## OperatorScratchPool tests


def test_operator_scratch_pool_attributes():
    """
    Verify that the operator scratch pool attributes are
    initialized correctly.
    """

    # Create one with default BufferPool as its parent
    pool = BufferPool.default()
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool, 0.4)
    op_scratch_pool: OperatorScratchPool = OperatorScratchPool(op_pool)
    assert op_scratch_pool.backend_name == op_pool.backend_name == pool.backend_name
    del op_scratch_pool
    del op_pool
    del pool


def test_operator_scratch_pool_allocation():
    """
    Test that allocating from the OperatorScratchPool
    works as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=4, min_size_class=8, enforce_max_limit_during_allocation=True
    )  # 4MiB, 8KiB
    pool: BufferPool = BufferPool.from_options(options)

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool, 0.5)
    op_scratch_pool: OperatorScratchPool = OperatorScratchPool(op_pool)

    ## 1. Basic allocation
    allocation: BufferPoolAllocation = op_scratch_pool.allocate(5 * 1024)
    # Verify that default allocation is 64B aligned
    assert allocation.alignment == 64
    assert allocation.get_ptr_as_int() % 64 == 0
    # Verify stats after allocation
    assert op_scratch_pool.bytes_allocated() == 5 * 1024
    assert op_scratch_pool.bytes_pinned() == 5 * 1024
    assert op_pool.bytes_allocated() == 5 * 1024
    assert op_pool.bytes_pinned() == 5 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 5 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 5 * 1024
    assert op_pool.main_mem_bytes_allocated() == 0
    assert op_pool.main_mem_bytes_pinned() == 0
    assert pool.bytes_allocated() == 5 * 1024
    assert pool.bytes_pinned() == 5 * 1024
    assert op_pool.max_memory() == 5 * 1024
    assert pool.max_memory() == 5 * 1024
    # We don't track max-memory for scratch portion,
    # so it'll be 0
    assert op_scratch_pool.max_memory() == 0

    # Free and verify stats
    op_scratch_pool.free(allocation)
    assert op_scratch_pool.bytes_allocated() == 0
    assert op_scratch_pool.bytes_pinned() == 0
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.scratch_mem_bytes_allocated() == 0
    assert op_pool.scratch_mem_bytes_pinned() == 0
    assert op_pool.main_mem_bytes_allocated() == 0
    assert op_pool.main_mem_bytes_pinned() == 0
    assert pool.bytes_allocated() == 0
    assert pool.bytes_pinned() == 0
    # Max memory should still be the same
    assert op_pool.max_memory() == 5 * 1024
    assert pool.max_memory() == 5 * 1024
    assert op_scratch_pool.max_memory() == 0

    ## 2. Test that we can go above the threshold safely.
    allocation = op_scratch_pool.allocate(400 * 1024)
    assert op_scratch_pool.bytes_allocated() == 400 * 1024
    assert op_scratch_pool.bytes_pinned() == 400 * 1024
    assert op_scratch_pool.bytes_allocated() == 400 * 1024
    assert op_scratch_pool.bytes_pinned() == 400 * 1024
    assert op_pool.bytes_allocated() == 400 * 1024
    assert op_pool.bytes_pinned() == 400 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 400 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 400 * 1024

    op_scratch_pool.free(allocation)
    assert op_scratch_pool.bytes_allocated() == 0
    assert op_scratch_pool.bytes_pinned() == 0
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.scratch_mem_bytes_allocated() == 0
    assert op_pool.scratch_mem_bytes_pinned() == 0

    ## 3. Test that we can't go over the budget.
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        op_scratch_pool.allocate(600 * 1024)
    assert op_scratch_pool.bytes_allocated() == 0
    assert op_scratch_pool.bytes_pinned() == 0
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.scratch_mem_bytes_allocated() == 0
    assert op_pool.scratch_mem_bytes_pinned() == 0

    ## 4. Test that we can't go over the budget if we
    ## have a main mem allocation and then some from
    ## the scratch portion.
    allocation_main_mem: BufferPoolAllocation = op_pool.allocate(250 * 1024)
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        op_scratch_pool.allocate(275 * 1024)
    assert op_scratch_pool.bytes_allocated() == 0
    assert op_scratch_pool.bytes_pinned() == 0
    assert op_pool.scratch_mem_bytes_allocated() == 0
    assert op_pool.scratch_mem_bytes_pinned() == 0
    assert op_pool.main_mem_bytes_allocated() == 250 * 1024
    assert op_pool.main_mem_bytes_pinned() == 250 * 1024

    op_pool.free(allocation_main_mem)
    assert op_scratch_pool.bytes_allocated() == 0
    assert op_scratch_pool.bytes_pinned() == 0
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.scratch_mem_bytes_allocated() == 0
    assert op_pool.scratch_mem_bytes_pinned() == 0
    assert op_pool.main_mem_bytes_allocated() == 0
    assert op_pool.main_mem_bytes_pinned() == 0

    del op_scratch_pool
    del op_pool
    del pool


def test_operator_scratch_pool_pin_unpin():
    """
    Test that pinning/unpinning from the OperatorScratchPool
    works as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=2,
        min_size_class=8,
        enforce_max_limit_during_allocation=True,
    )  # 2MiB total, 8KiB min size class
    pool: BufferPool = BufferPool.from_options(options)

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool)
    op_scratch_pool: OperatorScratchPool = OperatorScratchPool(op_pool)

    # Allocate a moderate size, then try to allocate amount
    # such that both together are not below budget. But after
    # unpinning the 1st allocation, the second is. Try pinning
    # the 1st back and verify behavior. Then free the second
    # and pin 1st back and verify behavior.

    allocation1: BufferPoolAllocation = op_scratch_pool.allocate(220 * 1024)
    op_pool.enable_threshold_enforcement()
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        op_scratch_pool.allocate(320 * 1024)

    # Verify stats after both allocation attempts
    assert op_scratch_pool.bytes_allocated() == 220 * 1024
    assert op_scratch_pool.bytes_pinned() == 220 * 1024
    assert op_pool.bytes_allocated() == 220 * 1024
    assert op_pool.bytes_pinned() == 220 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 220 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 220 * 1024

    # Unpin allocation and try again:
    op_scratch_pool.unpin(allocation1)
    allocation2: BufferPoolAllocation = op_scratch_pool.allocate(320 * 1024)
    assert op_scratch_pool.bytes_allocated() == 540 * 1024
    assert op_scratch_pool.bytes_pinned() == 320 * 1024
    assert op_pool.bytes_allocated() == 540 * 1024
    assert op_pool.bytes_pinned() == 320 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 540 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 320 * 1024

    # Trying to pin allocation1 back should error out
    with pytest.raises(RuntimeError, match="OperatorPoolThresholdExceededError"):
        op_scratch_pool.pin(allocation1)

    # Verify stats after pin attempt
    assert op_scratch_pool.bytes_allocated() == 540 * 1024
    assert op_scratch_pool.bytes_pinned() == 320 * 1024
    assert op_pool.bytes_allocated() == 540 * 1024
    assert op_pool.bytes_pinned() == 320 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 540 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 320 * 1024

    # Unpin the 2nd, and then pin the 1st. This should go through.
    op_scratch_pool.unpin(allocation2)
    op_scratch_pool.pin(allocation1)
    assert op_scratch_pool.bytes_allocated() == 540 * 1024
    assert op_scratch_pool.bytes_pinned() == 220 * 1024
    assert op_pool.bytes_allocated() == 540 * 1024
    assert op_pool.bytes_pinned() == 220 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 540 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 220 * 1024

    # Cleanup
    op_scratch_pool.free(allocation1)
    op_scratch_pool.free(allocation2)
    assert op_scratch_pool.bytes_allocated() == 0
    assert op_scratch_pool.bytes_pinned() == 0
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.scratch_mem_bytes_allocated() == 0
    assert op_pool.scratch_mem_bytes_pinned() == 0

    del op_scratch_pool
    del op_pool
    del pool


def test_operator_scratch_pool_reallocate():
    """
    Test that re-allocating from the OperatorScratchPool
    works as expected.
    """

    # Allocate a very small pool for testing
    options = BufferPoolOptions(
        memory_size=4,
        min_size_class=8,
        enforce_max_limit_during_allocation=True,
    )  # 4MiB, 8KiB
    pool: BufferPool = BufferPool.from_options(options)

    # Create a small operator pool (512KiB)
    op_pool: OperatorBufferPool = OperatorBufferPool(512 * 1024, pool, 0.5)
    op_scratch_pool: OperatorScratchPool = OperatorScratchPool(op_pool)

    ## Simple reallocate

    allocation: BufferPoolAllocation = op_scratch_pool.allocate(40 * 1024)

    # Verify stats after allocation
    assert op_scratch_pool.bytes_allocated() == 40 * 1024
    assert op_scratch_pool.bytes_pinned() == 40 * 1024
    assert op_pool.bytes_allocated() == 40 * 1024
    assert op_pool.bytes_pinned() == 40 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 40 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 40 * 1024
    assert pool.is_pinned(allocation)

    op_scratch_pool.reallocate(80 * 1024, allocation)
    # Verify stats after re-allocation
    assert op_scratch_pool.bytes_allocated() == 80 * 1024
    assert op_scratch_pool.bytes_pinned() == 80 * 1024
    assert op_pool.bytes_allocated() == 80 * 1024
    assert op_pool.bytes_pinned() == 80 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 80 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 80 * 1024
    assert pool.is_pinned(allocation)

    # Unpin and reallocate again
    op_scratch_pool.unpin(allocation)
    op_scratch_pool.reallocate(120 * 1024, allocation)
    assert op_scratch_pool.bytes_allocated() == 120 * 1024
    assert op_scratch_pool.bytes_pinned() == 120 * 1024
    assert op_pool.bytes_allocated() == 120 * 1024
    assert op_pool.bytes_pinned() == 120 * 1024
    assert op_pool.scratch_mem_bytes_allocated() == 120 * 1024
    assert op_pool.scratch_mem_bytes_pinned() == 120 * 1024
    assert pool.is_pinned(allocation)  # reallocate pins by default

    # Free and verify stats
    op_scratch_pool.free(allocation)
    assert op_scratch_pool.bytes_allocated() == 0
    assert op_scratch_pool.bytes_pinned() == 0
    assert op_pool.bytes_allocated() == 0
    assert op_pool.bytes_pinned() == 0
    assert op_pool.scratch_mem_bytes_allocated() == 0
    assert op_pool.scratch_mem_bytes_pinned() == 0

    del op_scratch_pool
    del op_pool
    del pool


def test_array_unpinned():
    """
    Test that arrays can be unpinned and are indicated so in
    the BufferPool
    """
    from numba.core import types

    from bodo.libs.array import array_info_type

    _array_info_unpin = types.ExternalFunction(
        "array_info_unpin", types.void(array_info_type)
    )

    @bodo.jit(returns_maybe_distributed=False)
    def impl():
        # Get initial statistics
        initial_allocated = 0
        initial_pinned = 0
        with bodo.objmode(initial_allocated="int64", initial_pinned="int64"):
            initial_allocated = default_buffer_pool_bytes_allocated()
            initial_pinned = default_buffer_pool_bytes_pinned()

        # Perform a Dataframe / Table Allocation
        # Seems to use int64 automatically
        df = pd.DataFrame({"A": np.arange(8000)})

        # Check Metrics before Unpinning
        passed_checks = True
        with bodo.objmode(passed_checks="boolean"):
            bytes_allocated = default_buffer_pool_bytes_allocated()
            bytes_pinned = default_buffer_pool_bytes_pinned()
            passed_checks = (bytes_allocated - initial_allocated) == 64 * 1024 and (
                bytes_pinned - initial_pinned
            ) == 64 * 1024
        if not passed_checks:
            return False

        arr_info = bodo.libs.array.array_to_info(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, 0)
        )
        _array_info_unpin(arr_info)

        # Check Metrics after Unpinning
        with bodo.objmode(passed_checks="boolean"):
            bytes_allocated = default_buffer_pool_bytes_allocated()
            bytes_pinned = default_buffer_pool_bytes_pinned()
            passed_checks = (bytes_allocated - initial_allocated) == 64 * 1024 and (
                bytes_pinned - initial_pinned
            ) == 0
        return passed_checks

    # TODO: Consider disabling test when bodo.get_size() != 1
    if bodo.get_rank() == 0:
        with pytest.warns(
            BodoWarning, match="No parallelism found for function 'impl'"
        ):
            assert impl()
    else:
        assert impl()
