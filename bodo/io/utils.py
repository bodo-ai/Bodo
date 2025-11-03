"""
Utility functions for I/O operations.
This file should import JIT lazily to avoid slowing down non-JIT code paths.
"""

from __future__ import annotations

from urllib.parse import urlparse


def parse_dbtype(con_str) -> tuple[str, str]:
    """
    Converts a constant string used for db_type to a standard representation
    for each database.
    """
    parseresult = urlparse(con_str)
    db_type = parseresult.scheme
    con_paswd = parseresult.password
    # urlparse skips oracle since its handle has _
    # which is not in `scheme_chars`
    # oracle+cx_oracle
    if con_str.startswith("oracle+cx_oracle://"):
        return "oracle", con_paswd
    if db_type == "mysql+pymysql":
        # Standardize mysql to always use "mysql"
        return "mysql", con_paswd

    # NOTE: if you're updating supported schemes here, don't forget
    # to update the associated error message in _run_call_read_sql_table

    if con_str.startswith("iceberg+glue") or parseresult.scheme in (
        "iceberg",
        "iceberg+file",
        "iceberg+s3",
        "iceberg+thrift",
        "iceberg+http",
        "iceberg+https",
    ):
        # Standardize iceberg to always use "iceberg"
        return "iceberg", con_paswd
    return db_type, con_paswd
