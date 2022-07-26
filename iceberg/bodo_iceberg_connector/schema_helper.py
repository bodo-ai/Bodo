from typing import List

import pyarrow as pa
import pyarrow.jvm
from bodo_iceberg_connector.errors import IcebergError
from bodo_iceberg_connector.py4j_support import (
    convert_list_to_java,
    get_iceberg_schema_class,
    get_iceberg_type_class,
    launch_jvm,
)


def pyarrow_to_iceberg_schema_str(arrow_schema: pa.Schema) -> str:
    """Convert a PyArrow schema to an JSON-encoded Iceberg schema string"""
    gateway = launch_jvm()
    schema = arrow_to_iceberg_schema(arrow_schema)
    return gateway.jvm.org.apache.iceberg.SchemaParser.toJson(schema)


def arrow_schema_j2py(jvm_schema) -> pa.Schema:
    """
    Construct a Schema from a org.apache.arrow.vector.types.pojo.Schema
    instance.

    Parameters:
        jvm_schema: org.apache.arrow.vector.types.pojo.Schema

    Returns: Equivalent PyArrow Schema object
    """
    try:
        return pa.jvm.schema(jvm_schema)
    except NotImplementedError:
        pass

    # Implementation from PyArrow's source:
    # https://github.com/apache/arrow/blob/9719b374408cfd37087f481c8e3f3a98fc89a3a8/python/pyarrow/jvm.py#L259

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
    return pa.schema(fields, metadata)


def arrow_field_j2py(jvm_field) -> pa.Field:
    """
    Construct a PyArrow Field from a Java Arrow Field instance.

    Parameters:
        jvm_field: org.apache.arrow.vector.types.pojo.Field

    Returns: pyarrow.Field
    """

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
    return pa.field(name, typ, nullable, metadata)


def arrow_type_j2py(jvm_field) -> pa.DataType:
    """
    Constructs the PyArrow Type of a Java Arrow Field instance.

    :param jvm_field: org.apache.arrow.vector.types.pojo.ArrowType
    :return: Corresponding PyArrow DataType
    """
    # BODO CHANGE: Extracted this code into a separate function for only getting PyArrow type
    # A separate function is required for recursively typing the keys and values in Map

    jvm_type = jvm_field.getType()
    type_str: str = jvm_type.getTypeID().toString()

    # Primitive Type Conversion
    if type_str == "Null":
        return pa.null()
    elif type_str == "Int":
        return pa.jvm._from_jvm_int_type(jvm_type)
    elif type_str == "FloatingPoint":
        return pa.jvm._from_jvm_float_type(jvm_type)
    elif type_str == "Utf8":
        return pa.string()
    elif type_str == "Binary":
        return pa.binary()
    elif type_str == "FixedSizeBinary":
        return pa.binary(jvm_type.getByteWidth())
    elif type_str == "Bool":
        return pa.bool_()
    elif type_str == "Time":
        return pa.jvm._from_jvm_time_type(jvm_type)
    elif type_str == "Timestamp":
        return pa.jvm._from_jvm_timestamp_type(jvm_type)
    elif type_str == "Date":
        return pa.jvm._from_jvm_date_type(jvm_type)
    elif type_str == "Decimal":
        return pa.decimal128(jvm_type.getPrecision(), jvm_type.getScale())

    # Complex Type Conversion
    # BODO CHANGE: Implemented Typing for List, Struct, and Map
    elif type_str == "List":
        elem_field = arrow_field_j2py(jvm_field.getChildren()[0])
        return pa.list_(elem_field)

    elif type_str == "Struct":
        fields = [arrow_field_j2py(elem) for elem in jvm_field.getChildren()]
        return pa.struct(fields)

    elif type_str == "Map":
        # For some reason, the first child field of a Map is another Map
        # The second Map contains the key and value types
        # TODO: Check if we can pass in fields to pyarrow.map_ or pyarrow.MapType for key and value
        # Would no longer need arrow_type_j2py and could only have arrow_field_j2py
        # implement conversion for complex types
        key_value = jvm_field.getChildren()[0].getChildren()
        key = arrow_type_j2py(key_value[0])
        value = arrow_type_j2py(key_value[1])
        return pa.map_(key, value)

    # Union, Dictionary and FixedSizeList should not be relavent to Iceberg
    else:
        raise IcebergError(f"Unsupported Java Arrow type: {type_str}")


def arrow_to_iceberg_schema(schema: pa.Schema):
    """
    Construct an Iceberg Java Schema object from a PyArrow Schema instance.
    Unlike reading where we convert from Iceberg to Java Arrow to PyArrow,
    we will directory convert from PyArrow to Iceberg.

    :param schema: PyArrow schema to convert
    :return: Equivalent org.apache.iceberg.Schema object
    """
    field_id_count = [len(schema) + 1]  # Hack in order to pass int by reference
    nested_fields = []
    for id in range(len(schema)):
        field = schema.field(id)
        nested_fields.append(arrow_to_iceberg_field(id + 1, field, field_id_count))

    IcebergSchema = get_iceberg_schema_class()
    return IcebergSchema(convert_list_to_java(nested_fields))


def arrow_to_iceberg_field(id: int, field: pa.Field, field_id_count: List[int]):
    IcebergTypes = get_iceberg_type_class()
    return IcebergTypes.NestedField.of(
        id,
        field.nullable,
        field.name,
        arrow_to_iceberg_type(field.type, field_id_count),
    )


def arrow_to_iceberg_type(field_type: pa.DataType, field_id_count: List[int]):
    """
    Convert a PyArrow data type to the correspoding Iceberg type.
    Handling cases when some PyArrow types are not supported in Iceberg.

    :param field_type: PyArrow DataType to convert
    :return: Corresponding org.apache.iceberg.type object
    """
    IcebergTypes = get_iceberg_type_class()

    if pa.types.is_null(field_type):
        raise IcebergError("Currently Cant Handle Purely Null Fields")
    elif pa.types.is_int32(field_type):
        return IcebergTypes.IntegerType.get()
    elif pa.types.is_int64(field_type):
        return IcebergTypes.LongType.get()
    elif pa.types.is_float32(field_type):
        return IcebergTypes.FloatType.get()
    elif pa.types.is_float64(field_type):
        return IcebergTypes.DoubleType.get()
    elif pa.types.is_string(field_type):
        return IcebergTypes.StringType.get()
    elif pa.types.is_binary(field_type):
        return IcebergTypes.BinaryType.get()
    elif pa.types.is_fixed_size_binary(field_type):
        return IcebergTypes.BinaryType.ofLength(field_type.byte_width)
    elif pa.types.is_boolean(field_type):
        return IcebergTypes.BooleanType.get()
    elif pa.types.is_time(field_type):
        return IcebergTypes.TimeType.get()
    elif pa.types.is_timestamp(field_type):
        if field_type.tz is None:
            return IcebergTypes.TimestampType.withoutZone()
        else:
            return IcebergTypes.TimestampType.withZone()
    elif pa.types.is_date(field_type):
        return IcebergTypes.DateType.get()
    elif pa.types.is_decimal(field_type):
        return IcebergTypes.DecimalType.of(field_type.precision, field_type.scale)

    # Complex Types
    elif pa.types.is_list(field_type) or pa.types.is_fixed_size_list(field_type):
        iceberg_type = arrow_to_iceberg_type(field_type.value_type, field_id_count)
        field_id = field_id_count[0]
        field_id_count[0] += 1
        return (
            IcebergTypes.ListType.ofOptional(field_id, iceberg_type)
            if field_type.value_field.nullable
            else IcebergTypes.ListType.ofRequired(field_id, iceberg_type)
        )

    elif pa.types.is_struct(field_type):
        id_offset = field_id_count[0]
        fields = []
        for id, field in enumerate(field_type):
            fields.append(arrow_to_iceberg_field(id + id_offset, field, field_id_count))

        field_id_count[0] += len(fields)
        struct_fields = convert_list_to_java(fields)
        return IcebergTypes.StructType.of(struct_fields)

    # Other types unable to convert.
    # Map is impossible due to no support in Bodo
    else:
        raise IcebergError(f"Unsupported PyArrow DataType")
