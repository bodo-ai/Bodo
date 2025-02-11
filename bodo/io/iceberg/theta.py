from __future__ import annotations

import sys
import typing as pt

import pandas as pd
from numba.core import types
from numba.extending import models, register_model

from bodo.utils.py_objs import install_py_obj_class
from bodo.utils.utils import run_rank0

if pt.TYPE_CHECKING:  # pragma: no cover
    from pyiceberg.table import Transaction


# Create a type for the Iceberg StatisticsFile object
# if we have the connector.
statistics_file_type = None
try:
    import bodo_iceberg_connector

    statistics_file_type = bodo_iceberg_connector.StatisticsFile
except ImportError:
    pass

this_module = sys.modules[__name__]
install_py_obj_class(
    types_name="statistics_file_type",
    python_type=statistics_file_type,
    module=this_module,
    class_name="StatisticsFileType",
    model_name="StatisticsFileModel",
)


class ThetaSketchCollectionType(types.Type):
    """Type for C++ pointer to a collection of theta sketches"""

    def __init__(self):  # pragma: no cover
        super().__init__(name="ThetaSketchCollectionType(r)")


register_model(ThetaSketchCollectionType)(models.OpaqueModel)

theta_sketch_collection_type = ThetaSketchCollectionType()


@run_rank0
def fetch_puffin_metadata(
    txn: Transaction,
) -> tuple[int, int, str]:
    """Fetch the puffin file metadata that we need from the committed
    transaction to write the puffin file. These are the:
        1. Snapshot ID for the committed data
        2. Sequence Number for the committed data
        3. The Location at which to write the puffin file.

    Args:
        transaction_id (int): Transaction ID to remove.
        conn_str (str): Connection string for indexing into our object list.
        db_name (str): Name of the database for indexing into our object list.
        table_name (str): Name of the table for indexing into our object list.

    Returns:
        tuple[int, int, str]: Tuple of the snapshot ID, sequence number, and
        location at which to write the puffin file.
    """
    raise NotImplementedError


@run_rank0
def commit_statistics_file(
    txn: Transaction,
    statistic_file_info,
) -> None:
    """
    Commit the statistics file to the iceberg table. This occurs after
    the puffin file has already been written and records the statistic_file_info
    in the metadata.

    Args:
        conn_str (str): The Iceberg connector string.
        db_name (str): The iceberg database name.
        table_name (str): The iceberg table.
        statistic_file_info (bodo_iceberg_connector.StatisticsFile):
            The Python object containing the statistics file information.
    """
    raise NotImplementedError


@run_rank0
def table_columns_have_theta_sketches(txn: Transaction) -> pd.arrays.BooleanArray:
    raise NotImplementedError


@run_rank0
def table_columns_enabled_theta_sketches(txn: Transaction) -> pd.arrays.BooleanArray:
    """
    Get an array of booleans indicating whether each column in the table
    has theta sketches enabled, as per the table property of
    'bodo.write.theta_sketch_enabled.<column_name>'.

    Args:
        conn_str (str): The Iceberg connector string.
        db_name (str): The iceberg database name.
        table_name (str): The iceberg table.
    """
    raise NotImplementedError


@run_rank0
def get_old_statistics_file_path(txn: Transaction) -> str:
    """
    Get the old puffin file path from the connector. We know that the puffin file
    must exist because of previous checks.
    """
    raise NotImplementedError
