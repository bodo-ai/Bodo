import bodo
from bodo.tests.utils import pytest_mark_one_rank


@pytest_mark_one_rank
def test_numba_to_cpp_types_decimal():
    """Verifies that decimal types are properly serialized by creating a nested type with decimals are different levels"""
    from bodo.utils.utils import (
        CArrayTypeEnum,
        CTypeEnum,
        numba_to_c_array_types,
        numba_to_c_types,
    )

    precisions = [32, 31, 30, 29]
    scales = [12, 13, 14, 15]
    decimal_arr_types = [
        bodo.types.DecimalArrayType(precisions[i], scales[i])
        for i in range(len(scales))
    ]

    array_type1 = bodo.types.ArrayItemArrayType(decimal_arr_types[0])
    array_type2 = bodo.types.ArrayItemArrayType(decimal_arr_types[1])

    map_type = bodo.types.MapArrayType(array_type2, decimal_arr_types[2])
    struct_type = bodo.types.StructArrayType(
        (array_type1, map_type, decimal_arr_types[3])
    )

    c_array_types = numba_to_c_array_types([struct_type])
    c_types = numba_to_c_types([struct_type])

    expected_c_array_types = [
        CArrayTypeEnum.STRUCT.value,
        3,
        CArrayTypeEnum.ARRAY_ITEM.value,
        CArrayTypeEnum.NULLABLE_INT_BOOL.value,
        precisions[0],
        scales[0],
        CArrayTypeEnum.MAP.value,
        CArrayTypeEnum.ARRAY_ITEM.value,
        CArrayTypeEnum.NULLABLE_INT_BOOL.value,
        precisions[1],
        scales[1],
        CArrayTypeEnum.NULLABLE_INT_BOOL.value,
        precisions[2],
        scales[2],
        CArrayTypeEnum.NULLABLE_INT_BOOL.value,
        precisions[3],
        scales[3],
    ]
    expected_ctypes = [
        CTypeEnum.STRUCT.value,
        3,
        CTypeEnum.LIST.value,
        CTypeEnum.Decimal.value,
        precisions[0],
        scales[0],
        CTypeEnum.Map.value,
        CTypeEnum.LIST.value,
        CTypeEnum.Decimal.value,
        precisions[1],
        scales[1],
        CTypeEnum.Decimal.value,
        precisions[2],
        scales[2],
        CTypeEnum.Decimal.value,
        precisions[3],
        scales[3],
    ]

    assert list(c_array_types) == expected_c_array_types
    assert list(c_types) == expected_ctypes
