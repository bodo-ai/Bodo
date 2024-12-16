import pyarrow as pa
import pyarrow.jvm

from bodo_iceberg_connector.errors import IcebergError
from bodo_iceberg_connector.py4j_support import (
    convert_list_to_java,
    get_iceberg_schema_class,
    get_iceberg_type_class,
    launch_jvm,
)

# This is the key used for storing the Iceberg Field ID in the
# metadata of the Arrow fields.
# Taken from: https://github.com/apache/arrow/blob/c23a097965b5c626cbc91b229c76a6c13d36b4e8/cpp/src/parquet/arrow/schema.cc#L245.
# Must match the value used in BodoArrowSchemaUtil.java and
# bodo/io/iceberg.py.
ICEBERG_FIELD_ID_MD_KEY = "PARQUET:field_id"

# PyArrow stores the metadata keys and values as bytes, so we need
# to use this encoded version when trying to access existing
# metadata in fields.
b_ICEBERG_FIELD_ID_MD_KEY = str.encode(ICEBERG_FIELD_ID_MD_KEY)


def pyarrow_to_iceberg_schema_str(arrow_schema: pa.Schema) -> str:
    """Convert a PyArrow schema to an JSON-encoded Iceberg schema string"""
    gateway = launch_jvm()
    schema = arrow_to_iceberg_schema(arrow_schema)
    return gateway.jvm.org.apache.iceberg.SchemaParser.toJson(schema)  # type: ignore


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
        key_value = jvm_field.getChildren()
        key = arrow_field_j2py(key_value[0])
        value = arrow_field_j2py(key_value[1])
        return pa.map_(key, value)

    # Union, Dictionary and FixedSizeList should not be relevant to Iceberg
    else:
        raise IcebergError(f"Unsupported Java Arrow type: {type_str}")


def convert_arrow_field_to_large_types(field: pa.Field) -> pa.Field:
    """
    Convert an Arrow field to the corresponding large type for all
    non-large BINARY, STRING, and LIST types.

    Args:
        field (pa.Field): The field to convert.

    Returns:
        pa.Field: The new field with possibly the type information modified.
    """
    field_type = field.type
    if pa.types.is_struct(field_type):
        new_fields = [convert_arrow_field_to_large_types(f) for f in field_type]
        new_type = pa.struct(new_fields)
    elif pa.types.is_map(field_type):
        key_field = convert_arrow_field_to_large_types(field_type.key_field)
        item_field = convert_arrow_field_to_large_types(field_type.item_field)
        new_type = pa.map_(key_field, item_field)
    elif (
        pa.types.is_list(field_type)
        or pa.types.is_large_list(field_type)
        or pa.types.is_fixed_size_list(field_type)
    ):
        elem_field = convert_arrow_field_to_large_types(field_type.field(0))
        new_type = pa.large_list(elem_field)
    elif pa.types.is_binary(field_type) or pa.types.is_fixed_size_binary(field_type):
        new_type = pa.large_binary()
    elif pa.types.is_string(field_type):
        new_type = pa.large_string()
    else:
        new_type = field_type
    return field.with_type(new_type)


def convert_arrow_schema_to_large_types(schema: pa.Schema) -> pa.Schema:
    """
    Converts an arrow schema derived from an Iceberg table's type
    information and converts all non-large BINARY, STRING, and LIST
    types to the "LARGE" variant.

    Args:
        schema (pa.Schema): The original schema to convert.

    Returns:
        pa.Schema: The new schema with the converted types.
    """
    new_fields = [convert_arrow_field_to_large_types(f) for f in schema]
    return pa.schema(new_fields)


def arrow_to_iceberg_schema(
    schema: pa.Schema, column_comments: list[str | None] | None = None
):
    """
    Construct an Iceberg Java Schema object from a PyArrow Schema instance.
    Unlike reading where we convert from Iceberg to Java Arrow to PyArrow,
    we will directory convert from PyArrow to Iceberg.

    :param schema: PyArrow schema to convert
    :column_comments: Column comments to include. None means no comments are provided.
    If not None, column_comments[i] = None means no comments for i-th column.
    If len(column_comments) < # columns, column_comments[i] are treated as None for i >= len(column_comments)
    :return: Equivalent org.apache.iceberg.Schema object
    """
    nested_fields = []
    for id in range(len(schema)):
        field = schema.field(id)
        nested_fields.append(
            arrow_to_iceberg_field(
                field,
                None
                if (column_comments is None or len(column_comments) <= id)
                else column_comments[id],
            )
        )

    IcebergSchema = get_iceberg_schema_class()
    return IcebergSchema(convert_list_to_java(nested_fields))


def arrow_to_iceberg_field(field: pa.Field, column_comment=None):
    IcebergTypes = get_iceberg_type_class()
    iceberg_field_id = int(field.metadata[b_ICEBERG_FIELD_ID_MD_KEY])
    return (
        IcebergTypes.NestedField.of(
            iceberg_field_id,
            field.nullable,
            field.name,
            arrow_to_iceberg_type(field.type),
        )
        if column_comment is None
        else IcebergTypes.NestedField.of(
            iceberg_field_id,
            field.nullable,
            field.name,
            arrow_to_iceberg_type(field.type),
            column_comment,
        )
    )


def arrow_to_iceberg_type(field_type: pa.DataType):
    """
    Convert a PyArrow data type to the corresponding Iceberg type.
    Handling cases when some PyArrow types are not supported in Iceberg.

    :param field_type: PyArrow DataType to convert
    :return: Corresponding org.apache.iceberg.type object
    """
    IcebergTypes = get_iceberg_type_class()

    if pa.types.is_null(field_type):
        raise IcebergError("Currently Cant Handle Purely Null Fields")
    elif (
        pa.types.is_int32(field_type)
        or pa.types.is_int16(field_type)
        or pa.types.is_int8(field_type)
        or pa.types.is_uint16(field_type)
        or pa.types.is_uint8(field_type)
    ):
        return IcebergTypes.IntegerType.get()
    elif pa.types.is_int64(field_type) or pa.types.is_uint32(field_type):
        return IcebergTypes.LongType.get()
    elif pa.types.is_float32(field_type):
        return IcebergTypes.FloatType.get()
    elif pa.types.is_float64(field_type):
        return IcebergTypes.DoubleType.get()
    elif pa.types.is_string(field_type) or pa.types.is_large_string(field_type):
        return IcebergTypes.StringType.get()
    elif pa.types.is_binary(field_type) or pa.types.is_large_binary(field_type):
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
    elif (
        pa.types.is_list(field_type)
        or pa.types.is_fixed_size_list(field_type)
        or pa.types.is_large_list(field_type)
    ):
        value_iceberg_type = arrow_to_iceberg_type(field_type.value_type)
        value_field_id = int(field_type.value_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY])
        return (
            IcebergTypes.ListType.ofOptional(value_field_id, value_iceberg_type)
            if field_type.value_field.nullable
            else IcebergTypes.ListType.ofRequired(value_field_id, value_iceberg_type)
        )
    elif pa.types.is_struct(field_type):
        fields = []
        for i in range(field_type.num_fields):
            fields.append(arrow_to_iceberg_field(field_type.field(i)))

        struct_fields = convert_list_to_java(fields)
        return IcebergTypes.StructType.of(struct_fields)
    elif pa.types.is_map(field_type):
        key_iceberg_type = arrow_to_iceberg_type(field_type.key_type)
        key_field_id = int(field_type.key_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY])
        item_iceberg_type = arrow_to_iceberg_type(field_type.item_type)
        item_field_id = int(field_type.item_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY])

        return (
            IcebergTypes.MapType.ofOptional(
                key_field_id, item_field_id, key_iceberg_type, item_iceberg_type
            )
            if field_type.item_field.nullable
            else IcebergTypes.MapType.ofRequired(
                key_field_id, item_field_id, key_iceberg_type, item_iceberg_type
            )
        )
    # Other types unable to convert.
    else:
        raise IcebergError(f"Unsupported PyArrow DataType: {field_type}")
