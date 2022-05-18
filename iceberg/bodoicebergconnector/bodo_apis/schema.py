"""
API used to translate a Java Schema object into various Pythonic
representations (Arrow and Bodo)
"""
from collections import namedtuple

import bodoicebergconnector.bodo_apis.jpype_support
from bodoicebergconnector.bodo_apis.config import DEFAULT_PORT

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


def get_bodo_connector_typing_schema(warehouse, schema, table):
    """
    Returns the table schema information for a given iceberg table
    used at typing. Also returns the pyarrow schema object.
    """
    schemas, pyarrow_schema = get_bodo_schemas(DEFAULT_PORT, warehouse, schema, table)
    return (schemas.colnames, schemas.coltypes, pyarrow_schema)


def get_bodo_connector_runtime_schema(warehouse, schema, table):
    """
    Returns the table schema information for a given iceberg table
    used at runtime.
    """
    schemas, _ = get_bodo_schemas(DEFAULT_PORT, warehouse, schema, table)
    return (schemas.field_ids, schemas.coltypes)


def get_pyarrow_schema(port, warehouse, schema, table, bodo_iceberg_table_reader=None):
    """
    Returns the pyarrow schemas for an iceberg table.
    Return type is `pyarrow.lib.Schema`.

    Port is unused and kept in case we opt to switch back to py4j
    """
    import pyarrow.jvm

    if bodo_iceberg_table_reader is None:
        bodo_iceberg_table_reader = (
            bodoicebergconnector.bodo_apis.jpype_support.get_iceberg_java_table_reader(
                warehouse,
                schema,
                table,
            )
        )

    # replace types with Bodo types using Arrow schema
    java_arrow_schema = bodo_iceberg_table_reader.getArrowSchema()
    pyarrow_schema = pyarrow.jvm.schema(java_arrow_schema)

    return pyarrow_schema


def get_bodo_schemas(port, warehouse, schema, table):
    """
    Returns all of the necessary Bodo schemas for an iceberg table,
    both using field_id and names.

    Port is unused and kept in case we opt to switch back to py4j
    """
    from bodo.io.parquet_pio import _get_numba_typ_from_pa_typ

    bodo_iceberg_table_reader = (
        bodoicebergconnector.bodo_apis.jpype_support.get_iceberg_java_table_reader(
            warehouse,
            schema,
            table,
        )
    )

    # get Iceberg schema
    java_schema = bodo_iceberg_table_reader.getIcebergSchema()
    py_schema = java_schema_to_python(java_schema)

    pyarrow_schema = get_pyarrow_schema(
        port, warehouse, schema, table, bodo_iceberg_table_reader
    )
    bodo_types = [
        _get_numba_typ_from_pa_typ(pyarrow_schema.field(name), False, True, None)[0]
        for name in pyarrow_schema.names
    ]

    return (
        BodoIcebergSchema(
            py_schema.colnames, bodo_types, py_schema.field_ids, py_schema.is_required
        ),
        pyarrow_schema,
    )


def java_schema_to_python(java_schema):
    """
    Converts a Java schema object to a Python equivalent
    specified by format_type.

    The valid options are:
        - "bodo": Convert to bodo types
        - None: Convert to a generic Python string.

        In the future we may add more (e.g. Arrow) if required.
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
