# Copyright (C) 2019 Bodo Inc. All rights reserved.
from numba import types
from numba.typing.templates import (
    signature,
    AbstractTemplate,
    infer,
    ConcreteTemplate,
    AttributeTemplate,
    bound_function,
    infer_global,
)
from numba.extending import models, register_model
from numba.extending import lower_builtin
from numba.targets.imputils import (
    impl_ret_new_ref,
    impl_ret_borrowed,
    iternext_impl,
    RefType,
)
from bodo.libs.str_ext import (
    string_type,
    gen_unicode_to_std_str,
    gen_std_str_to_unicode,
)
from llvmlite import ir as lir
import llvmlite.binding as ll
from bodo.libs import hdict_ext

ll_voidp = lir.IntType(8).as_pointer()


class MultiMapType(types.Opaque):
    def __init__(self, key_typ, val_typ):
        self.key_typ = key_typ
        self.val_typ = val_typ
        super(MultiMapType, self).__init__(
            name="MultiMapType{}{}".format(key_typ, val_typ)
        )

    @property
    def key(self):
        return self.key_typ, self.val_typ

    def is_precise(self):
        return self.key_typ.is_precise() and self.val_typ.is_precise()


register_model(MultiMapType)(models.OpaqueModel)


class MultiMapRangeIteratorType(types.SimpleIteratorType):
    def __init__(self, key_typ, val_typ):
        self.key_typ = key_typ
        self.val_typ = val_typ
        yield_type = val_typ
        super(MultiMapRangeIteratorType, self).__init__(
            "MultiMapRangeIteratorType{}{}".format(key_typ, val_typ), yield_type
        )

        @property
        def iterator_type(self):
            return self

        @property
        def key(self):
            return self.key_typ, self.val_typ

        def is_precise(self):
            return self.key_typ.is_precise() and self.val_typ.is_precise()


multimap_int64_range_iterator_type = MultiMapRangeIteratorType(types.intp, types.intp)

register_model(MultiMapRangeIteratorType)(models.OpaqueModel)

multimap_int64_type = MultiMapType(types.int64, types.int64)
multimap_int64_init = types.ExternalFunction(
    "multimap_int64_init", multimap_int64_type()
)
multimap_int64_insert = types.ExternalFunction(
    "multimap_int64_insert", types.void(multimap_int64_type, types.int64, types.int64)
)
multimap_int64_equal_range = types.ExternalFunction(
    "multimap_int64_equal_range",
    multimap_int64_range_iterator_type(multimap_int64_type, types.int64),
)


# store the iterator pair type in same storage and avoid repeated alloc
multimap_int64_equal_range_alloc = types.ExternalFunction(
    "multimap_int64_equal_range_alloc", multimap_int64_range_iterator_type()
)

multimap_int64_equal_range_dealloc = types.ExternalFunction(
    "multimap_int64_equal_range_dealloc", types.void(multimap_int64_range_iterator_type)
)

multimap_int64_equal_range_inplace = types.ExternalFunction(
    "multimap_int64_equal_range_inplace",
    multimap_int64_range_iterator_type(
        multimap_int64_type, types.int64, multimap_int64_range_iterator_type
    ),
)

ll.add_symbol("multimap_int64_init", hdict_ext.multimap_int64_init)
ll.add_symbol("multimap_int64_insert", hdict_ext.multimap_int64_insert)
ll.add_symbol("multimap_int64_equal_range", hdict_ext.multimap_int64_equal_range)
ll.add_symbol(
    "multimap_int64_equal_range_alloc", hdict_ext.multimap_int64_equal_range_alloc
)
ll.add_symbol(
    "multimap_int64_equal_range_dealloc", hdict_ext.multimap_int64_equal_range_dealloc
)
ll.add_symbol(
    "multimap_int64_equal_range_inplace", hdict_ext.multimap_int64_equal_range_inplace
)
ll.add_symbol("multimap_int64_it_is_valid", hdict_ext.multimap_int64_it_is_valid)
ll.add_symbol("multimap_int64_it_get_value", hdict_ext.multimap_int64_it_get_value)
ll.add_symbol("multimap_int64_it_inc", hdict_ext.multimap_int64_it_inc)


@lower_builtin("getiter", MultiMapRangeIteratorType)
def iterator_getiter(context, builder, sig, args):
    (it,) = args
    # return impl_ret_borrowed(context, builder, sig.return_type, it)
    return it


@lower_builtin("iternext", MultiMapRangeIteratorType)
@iternext_impl(RefType.UNTRACKED)
def iternext_listiter(context, builder, sig, args, result):
    ll_bool = context.get_value_type(types.bool_)  # lir.IntType(1)?

    # is valid
    fnty = lir.FunctionType(ll_bool, [ll_voidp])
    it_is_valid = builder.module.get_or_insert_function(
        fnty, name="multimap_int64_it_is_valid"
    )

    # get value
    val_typ = context.get_value_type(sig.args[0].val_typ)
    fnty = lir.FunctionType(val_typ, [ll_voidp])
    get_value = builder.module.get_or_insert_function(
        fnty, name="multimap_int64_it_get_value"
    )

    # increment
    fnty = lir.FunctionType(lir.VoidType(), [ll_voidp])
    inc_it = builder.module.get_or_insert_function(fnty, name="multimap_int64_it_inc")

    (range_it,) = args

    # it != range.second
    is_valid = builder.call(it_is_valid, [range_it])
    result.set_valid(is_valid)

    with builder.if_then(is_valid):
        # it->second
        val = builder.call(get_value, [range_it])
        result.yield_(val)
        builder.call(inc_it, [range_it])
