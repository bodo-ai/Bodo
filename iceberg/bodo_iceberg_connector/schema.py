"""
API used to translate a Java Schema object into various Pythonic
representations (Arrow and Bodo)
"""
from collections import namedtuple

import pyarrow.jvm
from bodo_iceberg_connector.config import DEFAULT_PORT
from bodo_iceberg_connector.errors import IcebergError, IcebergJavaError
from bodo_iceberg_connector.py4j_support import get_iceberg_java_table_reader
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


def get_iceberg_typing_schema(warehouse: str, schema: str, table: str):
    """
    Returns the table schema information for a given iceberg table
    used at typing. Also returns the pyarrow schema object.
    """
    schemas, pyarrow_schema = get_iceberg_schemas(
        DEFAULT_PORT, warehouse, schema, table
    )
    return (schemas.colnames, schemas.coltypes, pyarrow_schema)


def get_iceberg_runtime_schema(warehouse: str, schema: str, table: str):
    """
    Returns the table schema information for a given iceberg table
    used at runtime.
    """
    schemas, _ = get_iceberg_schemas(DEFAULT_PORT, warehouse, schema, table)
    return (schemas.field_ids, schemas.coltypes)


def get_pyarrow_schema(
    port: int, warehouse: str, schema: str, table: str, bodo_iceberg_table_reader=None
) -> "pyarrow.Schema":
    """
    Returns the pyarrow schemas for an iceberg table.
    Return type is `pyarrow.lib.Schema`.

    Port is unused and kept in case we opt to switch back to py4j
    """
    if bodo_iceberg_table_reader is None:
        bodo_iceberg_table_reader = get_iceberg_java_table_reader(
            warehouse,
            schema,
            table,
        )

    # replace types with Bodo types using Arrow schema
    java_arrow_schema = bodo_iceberg_table_reader.getArrowSchema()

    # First attempt to use PyArrow's built-in implementation
    # (which currently only supports primitive types)
    # If it fails, use the extended custom implementation
    try:
        pyarrow_schema = pyarrow.jvm.schema(java_arrow_schema)
    except NotImplementedError:
        pyarrow_schema = arrow_schema_java_to_py(java_arrow_schema)

    return pyarrow_schema


def get_iceberg_schemas(port: int, warehouse: str, schema: str, table: str):
    """
    Returns all of the necessary Bodo schemas for an iceberg table,
    both using field_id and names.

    Port is unused and kept in case we opt to switch back to py4j
    """
    try:
        bodo_iceberg_table_reader = get_iceberg_java_table_reader(
            warehouse,
            schema,
            table,
        )

        # get Iceberg schema
        java_schema = bodo_iceberg_table_reader.getIcebergSchema()
        py_schema = iceberg_schema_java_to_py(java_schema)

        pyarrow_schema = get_pyarrow_schema(
            port, warehouse, schema, table, bodo_iceberg_table_reader
        )
        bodo_types = [pyarrow_schema.field(name) for name in pyarrow_schema.names]

    except Py4JJavaError as e:
        raise IcebergJavaError.from_java_error(e)

    return (
        BodoIcebergSchema(
            py_schema.colnames, bodo_types, py_schema.field_ids, py_schema.is_required
        ),
        pyarrow_schema,
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


def arrow_schema_java_to_py(jvm_schema) -> "pyarrow.Schema":
    """
    Construct a Schema from a org.apache.arrow.vector.types.pojo.Schema
    instance.

    Parameters:
        jvm_schema: org.apache.arrow.vector.types.pojo.Schema

    Returns: pyarrow.Schema
    """
    # Implementation from PyArrow's source:
    # https://github.com/apache/arrow/blob/9719b374408cfd37087f481c8e3f3a98fc89a3a8/python/pyarrow/jvm.py#L259

    import pyarrow

    fields = jvm_schema.getFields()
    # BODO CHANGE: New name for function arrow_field_j2py(...) from field(...)
    fields = [arrow_field_j2py(f) for f in fields]
    jvm_metadata = jvm_schema.getCustomMetadata()
    if jvm_metadata.isEmpty():
        metadata = None
    else:
        metadata = {
            str(entry.getKey()): str(entry.getValue())
            for entry in jvm_metadata.entrySet()
        }
    return pyarrow.schema(fields, metadata)


def arrow_field_j2py(jvm_field) -> "pyarrow.Field":
    """
    Construct a PyArrow Field from a Java Arrow Field instance.

    Parameters:
        jvm_field: org.apache.arrow.vector.types.pojo.Field

    Returns: pyarrow.Field
    """
    import pyarrow

    name = str(jvm_field.getName())
    # BODO CHANGE: arrow_type_j2py was inlined in the official implementation
    # A separate function is required for recursively typing the keys and values in Map
    typ = arrow_type_j2py(jvm_field)

    nullable = jvm_field.isNullable()
    jvm_metadata = jvm_field.getMetadata()
    if jvm_metadata.isEmpty():
        metadata = None
    else:
        metadata = {
            str(entry.getKey()): str(entry.getValue())
            for entry in jvm_metadata.entrySet()
        }
    return pyarrow.field(name, typ, nullable, metadata)


def arrow_type_j2py(jvm_field):
    """
    Constructs the PyArrow Type of a Java Arrow Field instance.

    Parameters:
        jvm_field: org.apache.arrow.vector.types.pojo.ArrowType

    Returns: pyarrow.Type
    """
    # BODO CHANGE: Extracted this code into a separate function for only getting PyArrow type
    # A separate function is required for recursively typing the keys and values in Map
    import pyarrow

    jvm_type = jvm_field.getType()
    type_str: str = jvm_type.getTypeID().toString()

    # Primitive Type Conversion
    if type_str == "Null":
        return pyarrow.null()
    elif type_str == "Int":
        return pyarrow.jvm._from_jvm_int_type(jvm_type)
    elif type_str == "FloatingPoint":
        return pyarrow.jvm._from_jvm_float_type(jvm_type)
    elif type_str == "Utf8":
        return pyarrow.string()
    elif type_str == "Binary":
        return pyarrow.binary()
    elif type_str == "FixedSizeBinary":
        return pyarrow.binary(jvm_type.getByteWidth())
    elif type_str == "Bool":
        return pyarrow.bool_()
    elif type_str == "Time":
        return pyarrow.jvm._from_jvm_time_type(jvm_type)
    elif type_str == "Timestamp":
        return pyarrow.jvm._from_jvm_timestamp_type(jvm_type)
    elif type_str == "Date":
        return pyarrow.jvm._from_jvm_date_type(jvm_type)
    elif type_str == "Decimal":
        return pyarrow.decimal128(jvm_type.getPrecision(), jvm_type.getScale())

    # Complex Type Conversion
    # BODO CHANGE: Implemented Typing for List, Struct, and Map
    elif type_str == "List":
        elem_field = arrow_field_j2py(jvm_field.getChildren()[0])
        return pyarrow.list_(elem_field)

    elif type_str == "Struct":
        fields = [arrow_field_j2py(elem) for elem in jvm_field.getChildren()]
        return pyarrow.struct(fields)

    elif type_str == "Map":
        # For some reason, the first child field of a Map is another Map
        # The second Map contains the key and value types
        # TODO: Check if we can pass in fields to pyarrow.map_ or pyarrow.MapType for key and value
        # Would no longer need arrow_type_j2py and could only have arrow_field_j2py
        # implement conversion for complex types
        key_value = jvm_field.getChildren()[0].getChildren()
        key = arrow_type_j2py(key_value[0])
        value = arrow_type_j2py(key_value[1])
        return pyarrow.map_(key, value)

    # Union, Dictionary and FixedSizeList should not be relavent to Iceberg
    else:
        raise IcebergError(f"Unsupported Java Arrow type: {type_str}")
