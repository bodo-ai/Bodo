"""
Non-JIT Python helpers for theta sketch lifecycle in the pandas/df_lib path.
This module does NOT import numba or any JIT-dependent modules.
"""

import ctypes as _ctypes
import sys as _sys
from uuid import uuid4

import pandas as pd

from bodo.io.iceberg.common import _format_data_loc, _fs_from_file_path
from bodo.spawn.utils import run_rank0


def _load_theta_utils_lib():
    ext_module = _sys.modules.get("bodo.ext")
    if ext_module is None:
        raise RuntimeError("bodo.ext module not loaded")
    return _ctypes.CDLL(ext_module.__file__)


_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = _load_theta_utils_lib()
    return _lib


def delete_sketches(ptr):
    if ptr == 0:
        return
    lib = _get_lib()
    lib.bodo_theta_utils_delete_sketches.argtypes = [_ctypes.c_void_p]
    lib.bodo_theta_utils_delete_sketches(ptr)


def write_puffin_file_from_sketches(
    sketch_ptr,
    puffin_file_loc,
    bucket_region,
    snapshot_id,
    sequence_number,
    iceberg_schema,
    arrow_fs,
    existing_puffin_loc="",
):
    lib = _get_lib()
    lib.bodo_theta_utils_write_puffin.restype = _ctypes.py_object
    lib.bodo_theta_utils_write_puffin.argtypes = [
        _ctypes.c_void_p,
        _ctypes.c_char_p,
        _ctypes.c_char_p,
        _ctypes.c_int64,
        _ctypes.c_int64,
        _ctypes.py_object,
        _ctypes.py_object,
        _ctypes.c_char_p,
    ]
    return lib.bodo_theta_utils_write_puffin(
        _ctypes.c_void_p(sketch_ptr),
        puffin_file_loc.encode(),
        bucket_region.encode(),
        _ctypes.c_int64(snapshot_id),
        _ctypes.c_int64(sequence_number),
        iceberg_schema,
        arrow_fs,
        existing_puffin_loc.encode(),
    )


def _type_supports_theta_sketch(dtype):
    """Check if a PyArrow type supports theta sketches.

    Mirrors the C++ type_supports_theta_sketch() in _theta_sketches.h.
    """
    import pyarrow as pa

    return (
        pa.types.is_int32(dtype)
        or pa.types.is_int64(dtype)
        or pa.types.is_date32(dtype)
        or pa.types.is_time64(dtype)
        or pa.types.is_timestamp(dtype)
        or pa.types.is_large_string(dtype)
        or pa.types.is_large_binary(dtype)
        or pa.types.is_dictionary(dtype)
        or pa.types.is_decimal(dtype)
        or pa.types.is_float32(dtype)
        or pa.types.is_float64(dtype)
    )


def _is_default_theta_sketch_type(dtype):
    """Check if a PyArrow type enables theta sketches by default.

    Mirrors the C++ is_default_theta_sketch_type() in _theta_sketches.h.
    Same as supported, but excludes float32/float64.
    """
    import pyarrow as pa

    if not _type_supports_theta_sketch(dtype):
        return False
    return not (pa.types.is_float32(dtype) or pa.types.is_float64(dtype))


def get_default_theta_sketch_columns_py(iceberg_pyarrow_schema):
    return [
        _is_default_theta_sketch_type(iceberg_pyarrow_schema.field(i).type)
        for i in range(len(iceberg_pyarrow_schema))
    ]


def get_supported_theta_sketch_columns_py(iceberg_pyarrow_schema):
    return [
        _type_supports_theta_sketch(iceberg_pyarrow_schema.field(i).type)
        for i in range(len(iceberg_pyarrow_schema))
    ]


def compact_serialize_sketches(ptr):
    """Compact an UpdateSketchCollection and serialize to bytes.

    This is a local (non-MPI) operation. The returned bytes can be gathered
    across ranks via Python/MPI and later merged on rank 0.
    """
    if ptr == 0:
        return None
    lib = _get_lib()
    lib.bodo_theta_utils_compact_serialize.restype = _ctypes.py_object
    lib.bodo_theta_utils_compact_serialize.argtypes = [_ctypes.c_void_p]
    return lib.bodo_theta_utils_compact_serialize(_ctypes.c_void_p(ptr))


def merge_and_write_puffin(
    serialized_list,
    puffin_file_loc,
    bucket_region,
    snapshot_id,
    sequence_number,
    iceberg_schema,
    arrow_fs,
    existing_puffin_loc="",
):
    """Merge pre-serialized CompactSketchCollections and write puffin file.

    This is a rank-0-only, non-MPI operation. Each element of
    serialized_list is a bytes object from compact_serialize_sketches().
    """
    lib = _get_lib()
    lib.bodo_theta_utils_merge_and_write_puffin.restype = _ctypes.py_object
    lib.bodo_theta_utils_merge_and_write_puffin.argtypes = [
        _ctypes.py_object,
        _ctypes.c_char_p,
        _ctypes.c_char_p,
        _ctypes.c_int64,
        _ctypes.c_int64,
        _ctypes.py_object,
        _ctypes.py_object,
        _ctypes.c_char_p,
    ]
    return lib.bodo_theta_utils_merge_and_write_puffin(
        serialized_list,
        puffin_file_loc.encode(),
        bucket_region.encode(),
        _ctypes.c_int64(snapshot_id),
        _ctypes.c_int64(sequence_number),
        iceberg_schema,
        arrow_fs,
        existing_puffin_loc.encode(),
    )


@run_rank0
def fetch_puffin_metadata(txn):
    metadata = txn.table_metadata
    snapshot_id = metadata.current_snapshot_id
    assert snapshot_id is not None
    snapshot = metadata.current_snapshot()
    assert snapshot is not None
    sequence_number = snapshot.sequence_number
    assert sequence_number is not None
    location = _format_data_loc(
        f"{metadata.location}/metadata/{snapshot_id}-{uuid4()}.stats",
        _fs_from_file_path(metadata.location, txn._table.io),
    )
    return snapshot_id, sequence_number, location


@run_rank0
def table_columns_have_theta_sketches(table_metadata):
    cols = table_metadata.schema().columns
    snap_id = table_metadata.current_snapshot_id
    have_theta_sketches = [False] * len(cols)
    if snap_id is None:
        return pd.array(have_theta_sketches)
    field_id_to_idx = {col.field_id: i for i, col in enumerate(cols)}
    for stat_file in table_metadata.statistics:
        if stat_file.snapshot_id == snap_id:
            for blob_metadata in stat_file.blob_metadata:
                if blob_metadata.type != "apache-datasketches-theta-v1":
                    continue
                if len(blob_metadata.fields) != 1:
                    continue
                field = blob_metadata.fields[0]
                if field in field_id_to_idx:
                    have_theta_sketches[field_id_to_idx[field]] = True
            break
    return pd.array(have_theta_sketches, dtype="boolean")


@run_rank0
def table_columns_enabled_theta_sketches(txn):
    cols = txn.table_metadata.schema().columns
    props = txn.table_metadata.properties
    enabled = [
        props.get(f"bodo.write.theta_sketch_enabled.{col.name}", "true") == "true"
        for col in cols
    ]
    return pd.array(enabled, dtype="boolean")


@run_rank0
def get_old_statistics_file_path(txn):
    snap_id = txn.table_metadata.current_snapshot_id
    if snap_id is None:
        raise RuntimeError("Table does not have a snapshot.")
    for stat_file in txn.table_metadata.statistics:
        if stat_file.snapshot_id == snap_id:
            return stat_file.statistics_path
    raise RuntimeError("Table does not have a valid statistics file.")
