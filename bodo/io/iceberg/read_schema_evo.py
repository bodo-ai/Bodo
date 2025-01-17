"""
Helper functions for checking schema compatibility
between an Iceberg tables schema and its underlying Parquet files
"""

import pyarrow as pa

from bodo.io.iceberg.common import b_ICEBERG_FIELD_ID_MD_KEY


def validate_file_schema_field_compatible_with_read_schema_field(
    file_schema_field: pa.Field,
    read_schema_field: pa.Field,
    field_name_for_err_msg: str,
):
    """
    Helper function for 'validate_file_schema_compatible_with_read_schema'
    to validate specific fields recursively.

    Args:
        file_schema_field (pa.Field): Field in the file.
        read_schema_field (pa.Field): "Expected" field from the
            schema group's read_schema.
        field_name_for_err_msg (str): Since this function is
            called recursively, we pass in a string to display
            a more readable name for the nested fields.
            e.g. Instead of saying that the field 'a' is
            incompatible, this allows us to say that the field
            'A.a' is incompatible.

    Raises:
        RuntimeError: If the file_schema_field is incompatible with
            the read_schema_field or if an unsupported field type is
            found in the file_schema_field
    """
    # Check that the field id is what we expect it to be.
    if (file_schema_field.metadata is None) or (
        b_ICEBERG_FIELD_ID_MD_KEY not in file_schema_field.metadata
    ):
        raise RuntimeError(
            f"Field '{field_name_for_err_msg}' doesn't have an Iceberg field ID specified in the file!"
        )

    read_schema_field_iceberg_id = int(
        read_schema_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY]
    )
    file_schema_field_iceberg_id = int(
        file_schema_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY]
    )
    if read_schema_field_iceberg_id != file_schema_field_iceberg_id:
        raise RuntimeError(
            f"Iceberg Field ID mismatch in file for '{field_name_for_err_msg}' field! "
            f"Expected: {read_schema_field_iceberg_id}, got {file_schema_field_iceberg_id} instead."
        )

    field_repr: str = f"field '{field_name_for_err_msg}' (Iceberg Field ID: {read_schema_field_iceberg_id})"

    # Then check the following:
    # - It shouldn't be nullable if the read schema field isn't nullable.
    if (not read_schema_field.nullable) and file_schema_field.nullable:
        raise RuntimeError(f"Required {field_repr} is optional in the file!")

    # - Check that that the types are in the same 'class', i.e. can be upcast safely.
    read_schema_field_type: pa.DataType = read_schema_field.type
    if pa.types.is_signed_integer(read_schema_field_type):
        if not pa.types.is_signed_integer(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a signed integer, got {file_schema_field.type} instead!"
            )
        if read_schema_field_type.bit_width < file_schema_field.type.bit_width:
            raise RuntimeError(
                f"Bit-width of {field_repr} in file is larger ({file_schema_field.type.bit_width}) "
                f"than what's allowed ({read_schema_field_type.bit_width})!"
            )
    elif pa.types.is_unsigned_integer(read_schema_field_type):
        if not pa.types.is_unsigned_integer(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be an unsigned integer, got {file_schema_field.type} instead!"
            )
        if read_schema_field_type.bit_width < file_schema_field.type.bit_width:
            raise RuntimeError(
                f"Bit-width of {field_repr} in file is larger ({file_schema_field.type.bit_width}) "
                f"than what's allowed ({read_schema_field_type.bit_width})!"
            )
    elif pa.types.is_floating(read_schema_field_type):
        if not pa.types.is_floating(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a floating point number, got {file_schema_field.type} instead!"
            )
        if read_schema_field_type.bit_width < file_schema_field.type.bit_width:
            raise RuntimeError(
                f"Bit-width of {field_repr} in file is larger ({file_schema_field.type.bit_width}) "
                f"than what's allowed ({read_schema_field_type.bit_width})!"
            )
    elif pa.types.is_decimal(read_schema_field_type):
        if not pa.types.is_decimal(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a decimal, got {file_schema_field.type} instead!"
            )
        if read_schema_field_type.bit_width < file_schema_field.type.bit_width:
            raise RuntimeError(
                f"Bit-width of {field_repr} in file is larger ({file_schema_field.type.bit_width}) "
                f"than what's allowed ({read_schema_field_type.bit_width})!"
            )
        if read_schema_field_type.scale != file_schema_field.type.scale:
            raise RuntimeError(
                f"Scale of decimal {field_repr} doesn't match exactly. Expected {read_schema_field_type.scale}, "
                f"got {file_schema_field.type.scale} instead."
            )
        if read_schema_field_type.precision < file_schema_field.type.precision:
            raise RuntimeError(
                f"Precision of decimal {field_repr} in file is larger ({file_schema_field.type.precision}) "
                f"than what's allowed ({read_schema_field_type.precision})!"
            )
    elif pa.types.is_boolean(read_schema_field_type):
        if not pa.types.is_boolean(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a boolean, got {file_schema_field.type} instead!"
            )
    elif pa.types.is_string(read_schema_field_type) or pa.types.is_large_string(
        read_schema_field_type
    ):
        if not (
            pa.types.is_string(file_schema_field.type)
            or pa.types.is_large_string(file_schema_field.type)
        ):
            raise RuntimeError(
                f"Expected {field_repr} to be a string, got {file_schema_field.type} instead!"
            )
    elif (
        pa.types.is_binary(read_schema_field_type)
        or pa.types.is_large_binary(read_schema_field_type)
        or pa.types.is_fixed_size_binary(read_schema_field_type)
    ):
        if not (
            pa.types.is_binary(file_schema_field.type)
            or pa.types.is_large_binary(file_schema_field.type)
            or pa.types.is_fixed_size_binary(file_schema_field.type)
        ):
            raise RuntimeError(
                f"Expected {field_repr} to be a binary, got {file_schema_field.type} instead!"
            )
    elif pa.types.is_date(read_schema_field_type):
        if not pa.types.is_date(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a date, got {file_schema_field.type} instead!"
            )
        if read_schema_field_type.bit_width < file_schema_field.type.bit_width:
            raise RuntimeError(
                f"Bit-width of {field_repr} in file is larger ({file_schema_field.type.bit_width}) "
                f"than what's allowed ({read_schema_field_type.bit_width})!"
            )
    elif pa.types.is_time(read_schema_field_type):
        if not pa.types.is_time(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a time, got {file_schema_field.type} instead!"
            )
        if read_schema_field_type.bit_width < file_schema_field.type.bit_width:
            raise RuntimeError(
                f"Bit-width of {field_repr} in file is larger ({file_schema_field.type.bit_width}) "
                f"than what's allowed ({read_schema_field_type.bit_width})!"
            )
    elif pa.types.is_timestamp(read_schema_field_type):
        if not pa.types.is_timestamp(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a timestamp, got {file_schema_field.type} instead!"
            )
        # Timestamps always have a bit-width of 64.
        # XXX TODO Could add checks based on tz/unit here in the future if needed.
    elif (
        pa.types.is_list(read_schema_field_type)
        or pa.types.is_large_list(read_schema_field_type)
        or pa.types.is_fixed_size_list(read_schema_field_type)
    ):
        if not (
            pa.types.is_list(file_schema_field.type)
            or pa.types.is_large_list(file_schema_field.type)
            or pa.types.is_fixed_size_list(file_schema_field.type)
        ):
            raise RuntimeError(
                f"Expected {field_repr} to be a list, got {file_schema_field.type} instead!"
            )
        # Check the value field recursively.
        validate_file_schema_field_compatible_with_read_schema_field(
            file_schema_field.type.value_field,
            read_schema_field_type.value_field,
            field_name_for_err_msg=f"{field_name_for_err_msg}.value",
        )
    elif pa.types.is_map(read_schema_field_type):
        if not pa.types.is_map(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a map, got {file_schema_field.type} instead!"
            )
        # Check the key and item fields recursively.
        validate_file_schema_field_compatible_with_read_schema_field(
            file_schema_field.type.key_field,
            read_schema_field_type.key_field,
            field_name_for_err_msg=f"{field_name_for_err_msg}.key",
        )
        validate_file_schema_field_compatible_with_read_schema_field(
            file_schema_field.type.item_field,
            read_schema_field_type.item_field,
            field_name_for_err_msg=f"{field_name_for_err_msg}.value",
        )
    elif pa.types.is_struct(read_schema_field_type):
        if not pa.types.is_struct(file_schema_field.type):
            raise RuntimeError(
                f"Expected {field_repr} to be a struct, got {file_schema_field.type} instead!"
            )

        # We need all fields in the read schema to exist in the file schema, but not vice-versa.
        # However, all the the fields in the file must be in the same order as they are in the
        # read_schema.
        file_schema_field_type = file_schema_field.type
        file_schema_last_idx: int = -1
        for sub_idx in range(read_schema_field_type.num_fields):
            read_schema_sub_field: pa.Field = read_schema_field_type.field(sub_idx)
            file_schema_sub_field_idx: int = file_schema_field_type.get_field_index(
                read_schema_sub_field.name
            )
            if file_schema_sub_field_idx == -1:
                raise RuntimeError(
                    f"Expected struct {field_repr} to have subfield {read_schema_sub_field.name} "
                    "but it was not found!"
                )
            # file_schema_last_idx should be strictly lower than file_schema_sub_field_idx.
            # If it isn't, then that means the fields are not in the required order.
            # If it always is, then we are guaranteed that the fields are in the
            # required order.
            if file_schema_last_idx >= file_schema_sub_field_idx:
                expected_field_order: list[str] = [
                    read_schema_field_type.field(i).name
                    for i in range(read_schema_field_type.num_fields)
                ]
                actual_field_order: list[str] = [
                    file_schema_field_type.field(i).name
                    for i in range(file_schema_field_type.num_fields)
                ]
                raise RuntimeError(
                    f"Struct {field_repr} does not have subfield {read_schema_sub_field.name} in the right order! "
                    f"Expected ordered subset: {expected_field_order}, but got {actual_field_order} instead."
                )
            file_schema_last_idx = file_schema_sub_field_idx
            file_schema_sub_field: pa.Field = file_schema_field_type.field(
                file_schema_sub_field_idx
            )
            validate_file_schema_field_compatible_with_read_schema_field(
                file_schema_sub_field,
                read_schema_sub_field,
                field_name_for_err_msg=f"{field_name_for_err_msg}.{read_schema_sub_field.name}",
            )
    else:
        raise RuntimeError(
            "bodo.io.iceberg.validate_file_schema_field_compatible_with_read_schema_field: "
            f"Unsupported dtype '{read_schema_field_type}' for {field_repr}."
        )


def validate_file_schema_compatible_with_read_schema(
    file_schema: pa.Schema, read_schema: pa.Schema
):
    """
    Validate that the schema of the Iceberg Parquet file
    is compatible with the read schema of the schema
    group it belongs to.
    At this point, nested fields are expected to match
    exactly, but the top-level fields support all
    types of schema evolution that Iceberg supports.

    Args:
        file_schema (pa.Schema): Schema of the file.
        read_schema (pa.Schema): Schema of the schema group.

        The Iceberg field IDs must be in the metadata
        of the fields in both these schemas.
    """
    for read_schema_field in read_schema:
        # Check if the field exists in the file.
        field_name: str = read_schema_field.name
        if (file_schema_field_idx := file_schema.get_field_index(field_name)) != -1:
            file_schema_field: pa.Field = file_schema.field(file_schema_field_idx)
            validate_file_schema_field_compatible_with_read_schema_field(
                file_schema_field,
                read_schema_field,
                field_name_for_err_msg=field_name,
            )
        else:
            # If a field by that name doesn't exist in the file,
            # then verify that the field is nullable in the read schema.
            iceberg_field_id: int = int(
                read_schema_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY]
            )
            if not read_schema_field.nullable:
                raise RuntimeError(
                    f"Field '{field_name}' (Iceberg Field ID: {iceberg_field_id}) not "
                    "found in the file even though the field is not nullable/optional!"
                )
