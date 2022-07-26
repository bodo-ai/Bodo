"""
API used to translate a Java Schema object into various Pythonic
representations (Arrow and Bodo)
"""
from collections import namedtuple
from typing import Optional

from bodo_iceberg_connector.catalog_conn import (
    gen_table_loc,
    normalize_loc,
    parse_conn_str,
)
from bodo_iceberg_connector.config import DEFAULT_PORT
from bodo_iceberg_connector.errors import IcebergError, IcebergJavaError
from bodo_iceberg_connector.py4j_support import get_java_table_handler
from bodo_iceberg_connector.schema_helper import arrow_schema_j2py
from py4j.protocol import Py4JJavaError

# Types I didn't figure out how to test with Spark:
#   FixedType
#   TimeType
#   UUIDType
# These should be possible to support based on this information
# https://iceberg.apache.org/spec/#parquet

# Create a named tuple for schema components
BodoIcebergSchema = namedtuple(
    "BodoIcebergSchema", "colnames coltypes field_ids is_required"
)


def get_typing_info(conn_str: str, schema: str, table: str):
    """
    Return information about an Iceberg Table needed at compile-time
    Primarily used for writing to Iceberg
    """
    schema_id, table_loc, _, pyarrow_schema, iceberg_schema_str = get_iceberg_info(
        DEFAULT_PORT, conn_str, schema, table, False
    )

    return (table_loc, schema_id, pyarrow_schema, iceberg_schema_str, "", "")


def get_iceberg_typing_schema(conn_str: str, schema: str, table: str):
    """
    Returns the table schema information for a given iceberg table
    used at typing. Also returns the pyarrow schema object.
    """
    # TODO: Combine with get_typing_info?
    _, _, schemas, pyarrow_schema, _ = get_iceberg_info(
        DEFAULT_PORT, conn_str, schema, table
    )
    assert schemas is not None
    return (schemas.colnames, schemas.coltypes, pyarrow_schema)


def get_iceberg_runtime_schema(conn_str: str, schema: str, table: str):
    """
    Returns the table schema information for a given iceberg table
    used at runtime.
    """
    _, _, schemas, _, _ = get_iceberg_info(DEFAULT_PORT, conn_str, schema, table)
    assert schemas is not None
    return (schemas.field_ids, schemas.coltypes)


def get_iceberg_info(port: int, conn_str: str, schema: str, table: str, error=True):
    """
    Returns all of the necessary Bodo schemas for an iceberg table,
    both using field_id and names.

    Port is unused and kept in case we opt to switch back to py4j
    """
    try:
        # Parse conn_str to determine catalog type and warehouse location
        catalog_type, warehouse = parse_conn_str(conn_str)

        # Construct Table Reader
        bodo_iceberg_table_reader = get_java_table_handler(
            conn_str,
            catalog_type,
            schema,
            table,
        )

        # Get Iceberg Schema Info
        java_table_info = bodo_iceberg_table_reader.getTableInfo(error)
        if java_table_info is None:
            schema_id = None
            iceberg_schema_str = ""
            java_schema = None
            py_schema = None
            pyarrow_schema = None
            pyarrow_types = []
            iceberg_schema = None

            if warehouse is None:
                raise IcebergError(
                    "`warehouse` parameter required in connection string"
                )
            table_loc = gen_table_loc(catalog_type, warehouse, schema, table)

        else:
            schema_id: Optional[int] = java_table_info.getSchemaID()
            iceberg_schema_str = str(java_table_info.getIcebergSchemaEncoding())
            java_schema = java_table_info.getIcebergSchema()
            py_schema = iceberg_schema_java_to_py(java_schema)
            pyarrow_schema = arrow_schema_j2py(java_table_info.getArrowSchema())
            pyarrow_types = [
                pyarrow_schema.field(name) for name in pyarrow_schema.names
            ]
            iceberg_schema = BodoIcebergSchema(
                py_schema.colnames,
                pyarrow_types,
                py_schema.field_ids,
                py_schema.is_required,
            )

            # TODO: Override when warehouse is passed in?
            # Or move the table? Java API has ability to do so
            table_loc = java_table_info.getLoc()

            assert (
                py_schema.colnames == pyarrow_schema.names
            ), "Iceberg Schema Field Names Should be Equal in PyArrow Schema"

    except Py4JJavaError as e:
        raise IcebergJavaError.from_java_error(e)

    # TODO: Remove when sure that Iceberg's Schema and PyArrow's Schema always match
    # field_ids are necessary
    # pyarrow_types are directly from pyarrow schema so not necessary
    # Unsure about colnames
    # is_required == nullable from pyarrow
    return (
        schema_id,
        normalize_loc(table_loc),
        iceberg_schema,
        pyarrow_schema,
        iceberg_schema_str,
    )


def iceberg_schema_java_to_py(java_schema):
    """
    Converts an Iceberg Java schema object to a Python equivalent.
    """
    # List of column names
    colnames = []
    # List of column types
    coltypes = []
    # List of field ids
    field_ids = []
    # List of if each column is required
    is_required_lst = []
    # Get a list of Java objects for the columns
    field_objects = java_schema.columns()
    for field in field_objects:
        # This should be a Python string.
        name = str(field.name())
        # This should be a Python Integer
        field_id = int(field.fieldId())
        # This should be a Python Boolean
        is_required = bool(field.isRequired())
        # This should be a Java object
        iceberg_type = field.type()
        type_val = str(iceberg_type.toString())
        colnames.append(name)
        coltypes.append(type_val)
        field_ids.append(field_id)
        is_required_lst.append(is_required)

    return BodoIcebergSchema(colnames, coltypes, field_ids, is_required_lst)
