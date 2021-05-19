# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Support for PySpark APIs in Bodo JIT functions
"""
from collections import namedtuple

import numba
import numba.cpython.tupleobj
import pyspark
from numba.core import types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import (
    AbstractTemplate,
    infer_global,
    signature,
)
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_builtin,
    models,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)

from bodo.utils.typing import BodoError

# a sentinel value to designate anonymous Row field names
ANON_SENTINEL = "bodo_field_"


class SparkSessionType(types.Opaque):
    """data type for SparkSession object.
    Just a dummy value since it is not needed for computation in Bodo
    """

    def __init__(self):
        super(SparkSessionType, self).__init__(name="SparkSessionType")


spark_session_type = SparkSessionType()
register_model(SparkSessionType)(models.OpaqueModel)


class SparkSessionBuilderType(types.Opaque):
    """data type for SparkSession.builder object.
    Just a dummy value since it is not needed for computation in Bodo
    """

    def __init__(self):
        super(SparkSessionBuilderType, self).__init__(name="SparkSessionBuilderType")


spark_session_builder_type = SparkSessionBuilderType()
register_model(SparkSessionBuilderType)(models.OpaqueModel)


@intrinsic
def init_session(typingctx=None):
    """Create a SparkSession() value.
    creates a null value since the value isn't used
    """

    def codegen(context, builder, signature, args):
        return context.get_constant_null(spark_session_type)

    return spark_session_type(), codegen


@intrinsic
def init_session_builder(typingctx=None):
    """Create a SparkSession.builder value.
    creates a null value since the value isn't used
    """

    def codegen(context, builder, signature, args):
        return context.get_constant_null(spark_session_builder_type)

    return spark_session_builder_type(), codegen


@overload_method(SparkSessionBuilderType, "appName", no_unliteral=True)
def overload_appName(A, s):
    """returns a SparkSession value"""
    # ignoring config value for now (TODO: store in object)
    return lambda A, s: A  # pragma: no cover


@overload_method(SparkSessionBuilderType, "getOrCreate", no_unliteral=True)
def overload_getOrCreate(A):
    """returns a SparkSession value"""
    return lambda A: init_session()  # pragma: no cover


@typeof_impl.register(pyspark.sql.session.SparkSession)
def typeof_session(val, c):
    return spark_session_type


@box(SparkSessionType)
def box_spark_session(typ, val, c):
    """box SparkSession value by just calling SparkSession.builder.getOrCreate() to
    get a new SparkSession object.
    """
    # TODO(ehsan): store the Spark configs in native SparkSession object and set them
    # in boxed object
    mod_name = c.context.insert_const_string(c.builder.module, "pyspark")
    pyspark_obj = c.pyapi.import_module_noblock(mod_name)
    sql_obj = c.pyapi.object_getattr_string(pyspark_obj, "sql")
    session_class_obj = c.pyapi.object_getattr_string(sql_obj, "SparkSession")
    builder_obj = c.pyapi.object_getattr_string(session_class_obj, "builder")

    session_obj = c.pyapi.call_method(builder_obj, "getOrCreate", ())

    c.pyapi.decref(pyspark_obj)
    c.pyapi.decref(sql_obj)
    c.pyapi.decref(session_class_obj)
    c.pyapi.decref(builder_obj)
    return session_obj


@unbox(SparkSessionType)
def unbox_spark_session(typ, obj, c):
    """unbox SparkSession object by just creating a null value since value not used"""
    return NativeValue(c.context.get_constant_null(spark_session_type))


@lower_constant(SparkSessionType)
def lower_constant_spark_session(context, builder, ty, pyval):
    """lower constant SparkSession by returning a null value since value is not used
    in computation.
    """
    return context.get_constant_null(spark_session_type)


# NOTE: subclassing BaseNamedTuple to reuse some of Numba's namedtuple infrastructure
# TODO(ehsan): make sure it fully conforms to Row semantics which is a subclass of tuple
class RowType(types.BaseNamedTuple):
    """data type for Spark Row object."""

    def __init__(self, types, fields):

        self.types = tuple(types)
        self.count = len(self.types)
        self.fields = tuple(fields)
        # set instance_class to reuse Numba's namedtuple support
        self.instance_class = namedtuple("Row", fields)
        name = "Row({})".format(
            ", ".join(f"{f}:{t}" for f, t in zip(self.fields, self.types))
        )
        super(RowType, self).__init__(name)

    @property
    def key(self):
        return self.fields, self.types

    def __getitem__(self, i):
        return self.types[i]

    def __len__(self):
        return len(self.types)

    def __iter__(self):
        return iter(self.types)


@register_model(RowType)
class RowModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [(f, t) for f, t in zip(fe_type.fields, fe_type.types)]
        super(RowModel, self).__init__(dmm, fe_type, members)


@typeof_impl.register(pyspark.sql.types.Row)
def typeof_row(val, c):
    """get Numba type for Row objects, could have field names or not"""
    fields = (
        val.__fields__
        if hasattr(val, "__fields__")
        else tuple(f"{ANON_SENTINEL}{i}" for i in range(len(val)))
    )
    return RowType(tuple(numba.typeof(v) for v in val), fields)


@box(RowType)
def box_row(typ, val, c):
    """
    Convert native value to Row object by calling Row constructor with kws
    """
    # e.g. call Row(a=3, b="A")
    row_class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(pyspark.sql.types.Row))

    # call Row constructor with positional values for anonymous field name cases
    # e.g. Row(3, "A")
    if all(f.startswith(ANON_SENTINEL) for f in typ.fields):
        objects = [
            c.box(t, c.builder.extract_value(val, i)) for i, t in enumerate(typ.types)
        ]
        res = c.pyapi.call_function_objargs(row_class_obj, objects)
        for obj in objects:
            c.pyapi.decref(obj)
        c.pyapi.decref(row_class_obj)
        return res

    args = c.pyapi.tuple_pack([])

    objects = []
    kws_list = []
    for i, t in enumerate(typ.types):
        item = c.builder.extract_value(val, i)
        obj = c.box(t, item)
        kws_list.append((typ.fields[i], obj))
        objects.append(obj)

    kws = c.pyapi.dict_pack(kws_list)
    res = c.pyapi.call(row_class_obj, args, kws)

    for obj in objects:
        c.pyapi.decref(obj)
    c.pyapi.decref(row_class_obj)
    c.pyapi.decref(args)
    c.pyapi.decref(kws)
    return res


@infer_global(pyspark.sql.types.Row)
class RowConstructor(AbstractTemplate):
    def generic(self, args, kws):
        if args and kws:
            raise BodoError("Can not use both args " "and kwargs to create Row")

        arg_names = ", ".join(f"arg{i}" for i in range(len(args)))
        kw_names = ", ".join(f"{a} = ''" for a in kws)
        func_text = f"def row_stub({arg_names}{kw_names}):\n"
        func_text += "    pass\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        row_stub = loc_vars["row_stub"]
        pysig = numba.core.utils.pysignature(row_stub)

        # using positional args creates an anonymous field names
        if args:
            out_row = RowType(
                args, tuple(f"{ANON_SENTINEL}{i}" for i in range(len(args)))
            )
            return signature(out_row, *args).replace(pysig=pysig)

        kws = dict(kws)
        out_row = RowType(tuple(kws.values()), tuple(kws.keys()))
        return signature(out_row, *kws.values()).replace(pysig=pysig)


# constructor lowering is identical to namedtuple
lower_builtin(pyspark.sql.types.Row, types.VarArg(types.Any))(
    numba.cpython.tupleobj.namedtuple_constructor
)
