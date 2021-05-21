# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Support for PySpark APIs in Bodo JIT functions
"""
from collections import namedtuple

import numba
import numba.cpython.tupleobj
import pyspark
from numba.core import cgutils, types
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
    make_attribute_wrapper,
    models,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.utils.typing import BodoError, dtype_to_array_type, is_overload_true

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


@overload_method(
    SparkSessionBuilderType, "getOrCreate", inline="always", no_unliteral=True
)
def overload_getOrCreate(A):
    """returns a SparkSession value"""
    return lambda A: bodo.libs.pyspark_ext.init_session()  # pragma: no cover


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
            raise BodoError(
                "pyspark.sql.types.Row: Cannot use both args and kwargs to create Row"
            )

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


class SparkDataFrameType(types.Type):
    """data type for Spark DataFrame object. It's just a wrapper around a Pandas
    DataFrame in Bodo.
    """

    def __init__(self, df):
        self.df = df
        super(SparkDataFrameType, self).__init__(f"SparkDataFrame({df})")

    @property
    def key(self):
        return self.df

    def copy(self):
        return SparkDataFrameType(self.df)


@register_model(SparkDataFrameType)
class SparkDataFrameModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("df", fe_type.df)]
        super(SparkDataFrameModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(SparkDataFrameType, "df", "_df")


@intrinsic
def init_spark_df(typingctx, df_typ=None):
    """Create a Spark DataFrame value from a Pandas dataframe value"""

    def codegen(context, builder, sig, args):
        (df,) = args
        spark_df = cgutils.create_struct_proxy(sig.return_type)(context, builder)
        spark_df.df = df
        context.nrt.incref(builder, sig.args[0], df)
        return spark_df._getvalue()

    return SparkDataFrameType(df_typ)(df_typ), codegen


@overload_method(
    SparkSessionType, "createDataFrame", inline="always", no_unliteral=True
)
def overload_create_df(
    sp_session, data, schema=None, samplingRatio=None, verifySchema=True
):
    """create a Spark dataframe from Pandas DataFrame or list of Rows"""
    # Pandas dataframe input
    if isinstance(data, DataFrameType):

        def impl_df(
            sp_session, data, schema=None, samplingRatio=None, verifySchema=True
        ):
            # allow distributed input to createDataFrame() since doesn't break semantics
            data = bodo.scatterv(data, warn_if_dist=False)
            return bodo.libs.pyspark_ext.init_spark_df(data)

        return impl_df

    # check for list(RowType)
    if not (isinstance(data, types.List) and isinstance(data.dtype, RowType)):
        raise BodoError(
            f"SparkSession.createDataFrame(): 'data' should be a Pandas dataframe or list of Rows, not {data}"
        )

    columns = data.dtype.fields
    n_cols = len(data.dtype.types)
    func_text = "def impl(sp_session, data, schema=None, samplingRatio=None, verifySchema=True):\n"
    func_text += f"  n = len(data)\n"

    # allocate data arrays
    arr_types = []
    for i, t in enumerate(data.dtype.types):
        arr_typ = dtype_to_array_type(t)
        func_text += f"  A{i} = bodo.utils.utils.alloc_type(n, arr_typ{i}, (-1,))\n"
        arr_types.append(arr_typ)

    # fill data arrays
    func_text += f"  for i in range(n):\n"
    func_text += f"    r = data[i]\n"
    for i in range(n_cols):
        func_text += f"    A{i}[i] = bodo.utils.conversion.unbox_if_timestamp(r[{i}])\n"

    data_args = "({}{})".format(
        ", ".join(f"A{i}" for i in range(n_cols)), "," if len(columns) == 1 else ""
    )

    func_text += (
        "  index = bodo.hiframes.pd_index_ext.init_range_index(0, n, 1, None)\n"
    )
    func_text += f"  pdf = bodo.hiframes.pd_dataframe_ext.init_dataframe({data_args}, index, {columns})\n"
    func_text += f"  pdf = bodo.scatterv(pdf)\n"
    func_text += f"  return bodo.libs.pyspark_ext.init_spark_df(pdf)\n"
    loc_vars = {}
    _global = {"bodo": bodo}
    for i in range(n_cols):
        # NOTE: may not work for categorical arrays
        _global[f"arr_typ{i}"] = arr_types[i]
    exec(func_text, _global, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(SparkDataFrameType, "toPandas", inline="always", no_unliteral=True)
def overload_to_pandas(spark_df, _is_bodo_dist=False):
    """toPandas() gathers input data by default to follow Spark semantics but the
    user can specify distributed data
    """
    # no gather if dist flag is set in untyped pass
    if is_overload_true(_is_bodo_dist):
        return lambda spark_df, _is_bodo_dist=False: spark_df._df  # pragma: no cover

    def impl(spark_df, _is_bodo_dist=False):  # pragma: no cover
        # gathering data to follow toPandas() semantics
        return bodo.gatherv(spark_df._df)

    return impl
