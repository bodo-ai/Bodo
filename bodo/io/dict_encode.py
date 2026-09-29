"""Shared utilities for dictionary-encoded string reads.

Used by both the Iceberg and Parquet read paths in the DataFrame library to
select string columns that should be read with dictionary encoding and to
interpret the BODO_TEST_ICEBERG_DICT_ENCODE knob consistently.
"""

from __future__ import annotations

import os
import typing as pt

# Environment variable controlling dictionary encoding of string columns in
# the DataFrame library read paths (Iceberg and Parquet):
#   - "auto" (default): use the data statistics heuristic.
#   - "1": dict-encode all string columns (used for testing/A-B experiments).
#   - "0": disable dictionary encoding.
DICT_ENCODE_ENV_VAR = "BODO_TEST_ICEBERG_DICT_ENCODE"


def get_dict_encode_mode() -> str:
    """Get the dictionary encoding mode from the environment variable.

    Returns:
        str: One of "auto", "1" or "0".
    """
    mode = os.environ.get(DICT_ENCODE_ENV_VAR, "auto")
    if mode not in ("auto", "0", "1"):
        raise ValueError(
            f"Invalid value for {DICT_ENCODE_ENV_VAR}: {mode}. "
            "Expected one of 'auto', '0', '1'."
        )
    return mode


def select_str_as_dict_cols(
    str_col_names: list[str],
    determine_str_as_dict: pt.Callable[[list[str]], pt.Iterable[str]],
) -> list[str]:
    """Select the columns that should be read with dictionary encoding based
    on the BODO_TEST_ICEBERG_DICT_ENCODE knob.

    Args:
        str_col_names (list[str]): Names of the string columns to consider.
        determine_str_as_dict (Callable): Heuristic that returns the subset of
            the given column names that should be dict-encoded. Only called
            when the knob is set to "auto".

    Returns:
        list[str]: Names of the selected columns (subset of str_col_names).
    """
    if len(str_col_names) == 0:
        return []

    mode = get_dict_encode_mode()
    if mode == "0":
        str_as_dict_names: set[str] = set()
    elif mode == "1":
        str_as_dict_names = set(str_col_names)
    else:
        str_as_dict_names = set(determine_str_as_dict(str_col_names))

    return [name for name in str_col_names if name in str_as_dict_names]


def str_as_dict_from_col_sizes(
    str_col_names: list[str],
    total_uncompressed_sizes: pt.Sequence[int],
    total_rows: int,
) -> set[str]:
    """Apply the dict-encoding threshold to accumulated column sizes.

    A column is dict-encoded if its average uncompressed size per row is
    below READ_STR_AS_DICT_THRESHOLD (looked up through the module to allow
    test monkeypatching).

    Args:
        str_col_names (list[str]): Names of the string columns.
        total_uncompressed_sizes (Sequence[int]): Total uncompressed size of
            each column across the sampled files.
        total_rows (int): Total number of rows across the sampled files.

    Returns:
        set[str]: Names of the columns that should be dict-encoded.
    """
    from bodo.io.parquet_pio import READ_STR_AS_DICT_THRESHOLD

    if total_rows == 0:
        return set()

    str_as_dict = set()
    for i, col_name in enumerate(str_col_names):
        metric = total_uncompressed_sizes[i] / total_rows
        if metric < READ_STR_AS_DICT_THRESHOLD:
            str_as_dict.add(col_name)
    return str_as_dict
