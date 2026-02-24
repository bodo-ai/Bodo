import pandas as pd
import pyarrow as pa
from pyiceberg.catalog import WAREHOUSE_LOCATION


def read_iceberg_table(table_name: str, database_name: str) -> pd.DataFrame:
    """Read an iceberg table from a given database path and return a Pandas DataFrame.
    Converts datetime us unit to ns to match Bodo output.
    """
    from bodo.io.iceberg.catalog.dir import DirCatalog

    # TODO: support other catalog types and properties
    properties = {
        WAREHOUSE_LOCATION: database_name,
    }
    catalog_name = database_name
    catalog = DirCatalog(catalog_name, **properties)
    table = catalog.load_table(table_name)
    pa_table = table.scan().to_arrow()

    # Convert datetime us unit to ns to match Bodo output
    new_schema = pa_table.schema
    for i, f in enumerate(pa_table.schema):
        if isinstance(f.type, pa.TimestampType) and f.type.unit == "us":
            new_schema = new_schema.remove(i)
            new_schema = new_schema.insert(
                i, pa.field(f.name, pa.timestamp("ns", tz=f.type.tz))
            )

    if new_schema != pa_table.schema:
        pa_table = pa_table.cast(new_schema)

    return pa_table.to_pandas()


def read_iceberg_table_single_rank(table_name, database_name):
    """Same as above but runs on a single rank and broadcasts the result."""
    from mpi4py import MPI

    import bodo

    if bodo.get_rank() == 0:
        py_out = read_iceberg_table(table_name, database_name)
    else:
        py_out = None

    comm = MPI.COMM_WORLD
    py_out = comm.bcast(py_out, root=0)
    return py_out
