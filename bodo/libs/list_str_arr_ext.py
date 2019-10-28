# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Array implementation for list of string objects (e.g. from S.str.split()), which are
usually immutable.
The characters are stored in a contingous data array, and two arrays of offsets mark
the individual strings and lists. For example:
value:             [['a', 'bc'], ['a'], ['aaa', 'b', 'cc']]
data:              [a, b, c, a, a, a, a, b, c, c]
data_offsets:      [0, 1, 3, 4, 7, 8, 10]
index_offsets:     [0, 2, 3, 6]
"""
import operator
import numpy as np
import numba
import bodo
from numba import types
from numba.typing.templates import (
    infer_global,
    AbstractTemplate,
    infer,
    signature,
    AttributeTemplate,
    infer_getattr,
    bound_function,
)
import numba.typing.typeof
from numba.extending import (
    typeof_impl,
    type_callable,
    models,
    register_model,
    NativeValue,
    make_attribute_wrapper,
    lower_builtin,
    box,
    unbox,
    lower_getattr,
    intrinsic,
    overload_method,
    overload,
    overload_attribute,
)
from numba import cgutils
from bodo.libs.str_ext import string_type
from numba.targets.imputils import (
    impl_ret_new_ref,
    impl_ret_borrowed,
    iternext_impl,
    RefType,
)
from numba.targets.hashing import _Py_hash_t
import llvmlite.llvmpy.core as lc
from glob import glob
from bodo.utils.typing import is_overload_true, is_overload_none

char_typ = types.uint8
offset_typ = types.uint32

data_ctypes_type = types.ArrayCTypes(types.Array(char_typ, 1, "C"))
offset_ctypes_type = types.ArrayCTypes(types.Array(offset_typ, 1, "C"))


class ListStringArrayType(types.ArrayCompatible):
    def __init__(self):
        super(ListStringArrayType, self).__init__(name="ListStringArrayType()")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return types.List(string_type)

    def copy(self):
        return ListStringArrayType()


list_string_array_type = ListStringArrayType()


class ListStringArrayPayloadType(types.Type):
    def __init__(self):
        super(ListStringArrayPayloadType, self).__init__(name="ListStringArrayPayloadType()")


list_str_arr_payload_type = ListStringArrayPayloadType()


# XXX: C equivalent in _str_ext.cpp
@register_model(ListStringArrayPayloadType)
class ListStringArrayPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.CPointer(char_typ)),
            ("data_offsets", types.CPointer(offset_typ)),
            ("index_offsets", types.CPointer(offset_typ)),
            ("null_bitmap", types.CPointer(char_typ)),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


list_str_arr_model_members = [
    ("num_items", types.uint64),
    ("num_total_chars", types.uint64),
    ("data", types.CPointer(char_typ)),
    ("data_offsets", types.CPointer(offset_typ)),
    ("index_offsets", types.CPointer(offset_typ)),
    ("null_bitmap", types.CPointer(char_typ)),
    ("meminfo", types.MemInfoPointer(list_str_arr_payload_type)),
]


@register_model(ListStringArrayType)
class ListStringArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):

        models.StructModel.__init__(self, dmm, fe_type, list_str_arr_model_members)


# XXX: should these be exposed?
make_attribute_wrapper(ListStringArrayType, "num_items", "_num_items")
make_attribute_wrapper(ListStringArrayType, "num_total_chars", "_num_total_chars")
make_attribute_wrapper(ListStringArrayType, "data", "_data")
make_attribute_wrapper(ListStringArrayType, "data_offsets", "_data_offsets")
make_attribute_wrapper(ListStringArrayType, "index_offsets", "_index_offsets")
make_attribute_wrapper(ListStringArrayType, "null_bitmap", "_null_bitmap")
