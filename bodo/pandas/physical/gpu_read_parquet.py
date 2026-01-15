import glob
import os
from collections.abc import Iterator

import cudf
import pyarrow.parquet as pq

from bodo.mpi4py import MPI

# ---------- Helpers to inspect dataset ----------


def is_single_file(path: str) -> bool:
    """Return True if path is a single parquet file (not a directory)."""
    return os.path.isfile(path)


def list_parquet_files(path: str) -> list[str]:
    """Return sorted list of parquet files for a path (file or directory)."""
    if os.path.isfile(path):
        return [path]
    # directory or glob pattern
    # include .parquet files in directory (non-recursive)
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    return files


# ---------- Partitioning logic ----------


def partition_by_row_groups(
    single_file: str, rank: int, size: int
) -> list[tuple[str, int, int]]:
    """
    Partition a single parquet file by row groups into `size` parts.
    Returns a list of (file_path, start_rg, end_rg_exclusive) for this rank.
    Each rank receives contiguous row-group ranges.
    """
    pf = pq.ParquetFile(single_file)
    total_rg = pf.num_row_groups
    # compute roughly equal contiguous chunks of row groups
    base = total_rg // size
    rem = total_rg % size
    # compute start/end for this rank
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    if start >= end:
        return []  # this rank has no row groups
    return [(single_file, start, end)]


def partition_by_files(
    files: list[str], rank: int, size: int
) -> list[tuple[str, int, int]]:
    """
    Partition a list of files across ranks by file. For each assigned file,
    return (file_path, start_rg=0, end_rg_exclusive=num_row_groups).
    """
    # simple round-robin or block partitioning; use block partitioning for contiguous files
    total = len(files)
    base = total // size
    rem = total % size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    assigned = []
    for i in range(start, end):
        f = files[i]
        pf = pq.ParquetFile(f)
        assigned.append((f, 0, pf.num_row_groups))
    return assigned


def compute_rank_parts(path: str, rank: int, size: int) -> list[tuple[str, int, int]]:
    """
    Determine which (file, start_rg, end_rg) ranges this rank owns.
    If path is a single file -> partition by row groups.
    If path is directory or multiple files -> partition by file.
    """
    files = list_parquet_files(path)
    if len(files) == 0:
        return []
    if len(files) == 1:
        # single file -> partition by row groups
        return partition_by_row_groups(files[0], rank, size)
    else:
        # multi-file dataset -> partition by file
        return partition_by_files(files, rank, size)


# ---------- Batch reader for a rank ----------


def read_batches_for_parts(
    parts: list[tuple[str, int, int]], target_rows: int
) -> Iterator[tuple[cudf.DataFrame, bool]]:
    """
    Given a list of (file_path, start_rg, end_rg_exclusive) for this rank,
    yield batches (cudf.DataFrame, eof_flag). Each batch contains whole row groups
    and accumulates row groups until rows >= target_rows or parts exhausted.
    """
    if not parts:
        # nothing assigned to this rank
        yield cudf.DataFrame(), True
        return

    # iterate through assigned file ranges
    for file_path, start_rg, end_rg in parts:
        rg = start_rg
        # if there are no row groups in this file, skip
        if rg >= end_rg:
            continue

        while rg < end_rg:
            rows_accum = 0
            gpu_frames = []
            # accumulate whole row groups until target reached or file range exhausted
            while rg < end_rg and rows_accum < target_rows:
                # read whole row group into GPU
                try:
                    df_rg = cudf.read_parquet(file_path, row_group=rg)
                except TypeError:
                    # some cudf versions expect row_groups=[rg]
                    df_rg = cudf.read_parquet(file_path, row_groups=[rg])
                n = len(df_rg)
                gpu_frames.append(df_rg)
                rows_accum += n
                rg += 1

            # determine eof for this file: if rg >= end_rg and this is last part overall, we may be at EOF
            # but we only know global EOF when all parts are consumed by this rank; the caller can treat
            # EOF True when this was the last row group of the last assigned file.
            # For simplicity, set eof=True only when we've exhausted all parts after yielding this batch.
            # We'll check after outer loop.
            batch_df = (
                cudf.concat(gpu_frames, ignore_index=True)
                if len(gpu_frames) > 1
                else gpu_frames[0]
            )
            # yield batch; caller will continue iteration until generator is exhausted
            yield batch_df, False

    # after all parts consumed, signal EOF
    yield cudf.DataFrame(), True


# ---------- Convenience wrapper to get generator for current MPI rank ----------


def get_rank_batch_generator(
    path: str, target_rows: int
) -> Iterator[tuple[cudf.DataFrame, bool]]:
    """
    For the current MPI rank, compute assigned parts and return a generator that yields
    (cudf.DataFrame, eof_flag) batches of whole row groups.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parts = compute_rank_parts(path, rank, size)
    return read_batches_for_parts(parts, target_rows)


def get_next_batch(gen):
    return next(gen)
