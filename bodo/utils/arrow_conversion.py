"""Utilities for converting Pandas/Bodo arrays to Arrow
that can be used without importing the compiler."""

import pyarrow as pa


def convert_arrow_arr_to_dict(arr, arrow_type):
    """Convert PyArrow array's type to a type with dictionary-encoding as specified by
    arrow_type. Not using Arrow's cast() since not working for nested arrays
    (as of Arrow 13).

    Args:
        arr (pa.Array): input PyArrow array
        arrow_type (DataType): target Arrow array type with dictionary encoding

    Returns:
        pa.Array: converted PyArrow array
    """
    if (
        pa.types.is_large_list(arrow_type)
        or pa.types.is_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
    ):
        new_arr = arr.from_arrays(
            arr.offsets, convert_arrow_arr_to_dict(arr.values, arrow_type.value_type)
        )
        # Arrow's from_arrays ignores nulls (bug as of Arrow 14) so we add them back manually
        return pa.Array.from_buffers(
            new_arr.type, len(new_arr), arr.buffers()[:2], children=[new_arr.values]
        )

    if pa.types.is_struct(arrow_type):
        new_arrs = [
            convert_arrow_arr_to_dict(arr.field(i), arrow_type.field(i).type)
            for i in range(arr.type.num_fields)
        ]
        names = [arr.type.field(i).name for i in range(arr.type.num_fields)]
        new_arr = arr.from_arrays(new_arrs, names)
        # Arrow's from_arrays ignores nulls (bug as of Arrow 14) so we add them back manually
        return pa.Array.from_buffers(
            new_arr.type, len(new_arr), arr.buffers()[:1], children=new_arrs
        )

    if pa.types.is_map(arrow_type):
        new_keys = convert_arrow_arr_to_dict(arr.keys, arrow_type.key_type)
        new_items = convert_arrow_arr_to_dict(arr.items, arrow_type.item_type)
        new_arr = arr.from_arrays(arr.offsets, new_keys, new_items)
        # Arrow's from_arrays ignores nulls (bug as of Arrow 14) so we add them back manually
        buffs = new_arr.buffers()
        buffs[0] = pa.compute.invert(arr.is_null()).buffers()[1]
        return new_arr.from_buffers(
            new_arr.type, len(new_arr), buffs[:2], children=[new_arr.values]
        )

    if (
        pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type)
    ) and pa.types.is_dictionary(arrow_type):
        return arr.dictionary_encode()

    return arr
