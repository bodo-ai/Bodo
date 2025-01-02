"""
Support for PySpark APIs in Bodo JIT functions
"""

from collections import namedtuple

import numba
import numba.cpython.tupleobj
import numpy as np
import pyspark
import pyspark.sql.functions as F
from numba.core import cgutils, ir_utils, types
from numba.core.imputils import lower_constant
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    infer_global,
    signature,
)
from numba.extending import (
    NativeValue,
    box,
    infer_getattr,
    intrinsic,
    lower_builtin,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)

import bodo
from bodo.hiframes.pd_dataframe_ext import (
    DataFrameType,
    check_runtime_cols_unsupported,
)
from bodo.utils.typing import (
    BodoError,
    ColNamesMetaType,
    assert_bodo_error,
    check_unsupported_args,
    dtype_to_array_type,
    get_overload_const_list,
    get_overload_const_str,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_true,
)

# a sentinel value to designate anonymous Row field names
ANON_SENTINEL = "bodo_field_"


class SparkSessionType(types.Opaque):
    """data type for SparkSession object.
    Just a dummy value since it is not needed for computation in Bodo
    """

    def __init__(self):
        super().__init__(name="SparkSessionType")


spark_session_type = SparkSessionType()
register_model(SparkSessionType)(models.OpaqueModel)


class SparkSessionBuilderType(types.Opaque):
    """data type for SparkSession.builder object.
    Just a dummy value since it is not needed for computation in Bodo
    """

    def __init__(self):
        super().__init__(name="SparkSessionBuilderType")


spark_session_builder_type = SparkSessionBuilderType()
register_model(SparkSessionBuilderType)(models.OpaqueModel)


@intrinsic
def init_session(typingctx):
    """Create a SparkSession() value.
    creates a null value since the value isn't used
    """

    def codegen(context, builder, signature, args):
        return context.get_constant_null(spark_session_type)

    return spark_session_type(), codegen


@intrinsic
def init_session_builder(typingctx):
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
    pyspark_obj = c.pyapi.import_module(mod_name)
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
        super().__init__(name)

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
        members = list(zip(fe_type.fields, fe_type.types))
        super().__init__(dmm, fe_type, members)


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
        super().__init__(f"SparkDataFrame({df})")

    @property
    def key(self):
        return self.df

    def copy(self):
        return SparkDataFrameType(self.df)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@register_model(SparkDataFrameType)
class SparkDataFrameModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("df", fe_type.df)]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(SparkDataFrameType, "df", "_df")


@intrinsic
def init_spark_df(typingctx, df_typ):
    """Create a Spark DataFrame value from a Pandas DataFrame value"""

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
    check_runtime_cols_unsupported(data, "spark.createDataFrame()")
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
    func_text += "  n = len(data)\n"

    # allocate data arrays
    arr_types = []
    for i, t in enumerate(data.dtype.types):
        arr_typ = dtype_to_array_type(t)
        func_text += f"  A{i} = bodo.utils.utils.alloc_type(n, arr_typ{i}, (-1,))\n"
        arr_types.append(arr_typ)

    # fill data arrays
    func_text += "  for i in range(n):\n"
    func_text += "    r = data[i]\n"
    for i in range(n_cols):
        func_text += (
            f"    A{i}[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(r[{i}])\n"
        )

    data_args = "({}{})".format(
        ", ".join(f"A{i}" for i in range(n_cols)), "," if len(columns) == 1 else ""
    )

    func_text += (
        "  index = bodo.hiframes.pd_index_ext.init_range_index(0, n, 1, None)\n"
    )
    func_text += f"  pdf = bodo.hiframes.pd_dataframe_ext.init_dataframe({data_args}, index, __col_name_meta_value_create_df)\n"
    func_text += "  pdf = bodo.scatterv(pdf)\n"
    func_text += "  return bodo.libs.pyspark_ext.init_spark_df(pdf)\n"
    loc_vars = {}
    _global = {
        "bodo": bodo,
        "__col_name_meta_value_create_df": ColNamesMetaType(tuple(columns)),
    }
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
        # Spark dataframe may be replicated, e.g. sdf.select(F.sum(F.col("A")))
        return bodo.gatherv(spark_df._df, warn_if_rep=False)

    return impl


@overload_method(SparkDataFrameType, "limit", inline="always", no_unliteral=True)
def overload_limit(spark_df, num):
    """returns the first `num` rows"""

    def impl(spark_df, num):  # pragma: no cover
        return bodo.libs.pyspark_ext.init_spark_df(spark_df._df.iloc[:num])

    return impl


def _df_to_rows(df):
    pass


@overload(_df_to_rows)
def overload_df_to_rows(df):
    """convert dataframe to list of Rows"""
    func_text = "def impl(df):\n"
    for i in range(len(df.columns)):
        func_text += (
            f"  A{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})\n"
        )
    func_text += "  n = len(df)\n"
    func_text += "  out = []\n"
    func_text += "  for i in range(n):\n"
    row_in = ", ".join(f"{c}=A{i}[i]" for i, c in enumerate(df.columns))
    func_text += f"    out.append(Row({row_in}))\n"
    func_text += "  return out\n"

    loc_vars = {}
    _global = {"bodo": bodo, "Row": pyspark.sql.types.Row}
    exec(func_text, _global, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(SparkDataFrameType, "collect", inline="always", no_unliteral=True)
def overload_collect(spark_df):
    """returns all rows as a list of Rows"""

    def impl(spark_df):  # pragma: no cover
        data = bodo.gatherv(spark_df._df, warn_if_rep=False)
        return _df_to_rows(data)

    return impl


@overload_method(SparkDataFrameType, "take", inline="always", no_unliteral=True)
def overload_take(spark_df, num):
    """returns the first `num` rows as a list of Rows"""

    def impl(spark_df, num):  # pragma: no cover
        return spark_df.limit(num).collect()

    return impl


@infer_getattr
class SparkDataFrameAttribute(AttributeTemplate):
    key = SparkDataFrameType

    def generic_resolve(self, sdf, attr):
        """return Column object for column selection"""
        if attr in sdf.df.columns:
            return ColumnType(ExprType("col", (attr,)))


# don't convert literal types to non-literal and rerun the typing template
SparkDataFrameAttribute._no_unliteral = True


# NOTE: inlining in SeriesPass since stararg is not supported in inline_closurecall()
@overload_method(SparkDataFrameType, "select", no_unliteral=True)
def overload_df_select(spark_df, *cols):
    return _gen_df_select(spark_df, cols)


def _gen_df_select(spark_df, cols, avoid_stararg=False):
    """generate code for SparkDataFrame.select()
    'avoid_stararg=True' avoids the unnecessary stararg argument to enable inlining in
    SeriesPass
    """
    df_type = spark_df.df

    # Numba passes a tuple of actual types, or a 1-tuple of StarArgTuple
    if (
        isinstance(cols, tuple)
        and len(cols) == 1
        and isinstance(cols[0], (types.StarArgTuple, types.StarArgUniTuple))
    ):
        cols = cols[0]

    # user may pass a list
    if len(cols) == 1 and is_overload_constant_list(cols[0]):
        cols = get_overload_const_list(cols[0])

    func_text = f"def impl(spark_df, {'' if avoid_stararg else '*cols'}):\n"
    func_text += "  df = spark_df._df\n"

    out_col_names = []
    out_data = []
    for col in cols:
        col = get_overload_const_str(col) if is_overload_constant_str(col) else col
        out_col_names.append(_get_col_name(col))
        data, code = _gen_col_code(col, df_type)
        func_text += code
        out_data.append(data)

    return _gen_init_spark_df(func_text, out_data, out_col_names)


def _gen_init_spark_df(func_text, out_data, out_col_names):
    """generate code for initializing a new Spark dataframe and return the
    implementation
    """

    data_args = "({}{})".format(", ".join(out_data), "," if len(out_data) == 1 else "")

    length = "0" if not out_data else f"len({out_data[0]})"
    func_text += f"  n = {length}\n"
    func_text += (
        "  index = bodo.hiframes.pd_index_ext.init_range_index(0, n, 1, None)\n"
    )
    func_text += f"  pdf = bodo.hiframes.pd_dataframe_ext.init_dataframe({data_args}, index, __col_name_meta_value_init_spark_df)\n"
    func_text += "  return bodo.libs.pyspark_ext.init_spark_df(pdf)\n"

    loc_vars = {}
    _global = {
        "bodo": bodo,
        "np": np,
        "__col_name_meta_value_init_spark_df": ColNamesMetaType(tuple(out_col_names)),
    }
    exec(func_text, _global, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(SparkDataFrameType, "show", inline="always", no_unliteral=True)
def overload_show(spark_df, n=20, truncate=True, vertical=False):
    """
    Just print df.head() for now. TODO(ehsan): create accurate output based on
    Spark's code:
    https://github.com/apache/spark/blob/e8631660ecf316e4333210650d1f40b5912fb11b/python/pyspark/sql/dataframe.py#L442
    https://github.com/apache/spark/blob/34284c06496cd621792c0f9dfc90435da0ab9eb5/sql/core/src/main/scala/org/apache/spark/sql/Dataset.scala#L335
    """
    unsupported_args = {
        "truncate": truncate,
        "vertical": vertical,
    }
    arg_defaults = {"truncate": True, "vertical": False}
    check_unsupported_args("SparkDataFrameType.show", unsupported_args, arg_defaults)

    def impl(spark_df, n=20, truncate=True, vertical=False):  # pragma: no cover
        print(spark_df._df.head(n))

    return impl


@overload_method(SparkDataFrameType, "printSchema", inline="always", no_unliteral=True)
def overload_print_schema(spark_df):
    """
    Just print df.dtypes() for now. TODO(ehsan): create accurate output
    """

    def impl(spark_df):  # pragma: no cover
        print(spark_df._df.dtypes)

    return impl


@overload_method(SparkDataFrameType, "withColumn", inline="always", no_unliteral=True)
def overload_with_column(spark_df, colName, col):
    """generate code for SparkDataFrame.withColumn(), which creates a new SparkDataFrame
    with existing columns and a new column.
    """

    _check_column(col)
    if not is_overload_constant_str(colName):  # pragma: no cover
        raise BodoError(
            f"SparkDataFrame.withColumn(): 'colName' should be a constant string, not {colName}"
        )

    col_name = get_overload_const_str(colName)
    curr_columns = spark_df.df.columns

    new_columns = (
        curr_columns if col_name in curr_columns else curr_columns + (col_name,)
    )
    new_col_var, new_col_code = _gen_col_code(col, spark_df.df)

    out_data = [
        new_col_var
        if c == col_name
        else f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {curr_columns.index(c)})"
        for c in new_columns
    ]

    func_text = "def impl(spark_df, colName, col):\n"
    func_text += "  df = spark_df._df\n"
    func_text += new_col_code

    return _gen_init_spark_df(func_text, out_data, new_columns)


@overload_method(
    SparkDataFrameType, "withColumnRenamed", inline="always", no_unliteral=True
)
def overload_with_column_renamed(spark_df, existing, new):
    """generate code for SparkDataFrame.withColumnRenamed(), which creates a new
    SparkDataFrame with a column potentially renamed.
    """

    if not (
        is_overload_constant_str(existing) and is_overload_constant_str(new)
    ):  # pragma: no cover
        raise BodoError(
            f"SparkDataFrame.withColumnRenamed(): 'existing' and 'new' should be a constant strings, not ({existing}, {new})"
        )

    old_colname = get_overload_const_str(existing)
    new_colname = get_overload_const_str(new)
    curr_columns = spark_df.df.columns

    # this is a no-op if 'old_colname' is not in the schema
    # 'new_colname' could be in the schema since Spark allows repeated column names
    new_columns = tuple(new_colname if c == old_colname else c for c in curr_columns)

    # data is the same as before
    out_data = [
        f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})"
        for i in range(len(curr_columns))
    ]

    func_text = "def impl(spark_df, existing, new):\n"
    func_text += "  df = spark_df._df\n"

    return _gen_init_spark_df(func_text, out_data, new_columns)


@overload_attribute(SparkDataFrameType, "columns", inline="always")
def overload_dataframe_columns(spark_df):
    """support 'columns' attribute which returns a string list of column names"""
    # embedding column names in generated function instead of returning a freevar since
    # there is no constant lowering for lists in Numba (TODO: support)
    col_names = [str(a) for a in spark_df.df.columns]
    func_text = "def impl(spark_df):\n"
    func_text += f"  return {col_names}\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    impl = loc_vars["impl"]
    return impl


class ColumnType(types.Type):
    """data type for Spark Column object"""

    def __init__(self, expr):
        self.expr = expr
        super().__init__(f"Column({expr})")

    @property
    def key(self):
        return self.expr

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


register_model(ColumnType)(models.OpaqueModel)


class ExprType(types.Type):
    """data type for Spark Column Expression"""

    def __init__(self, op, children):
        self.op = op
        self.children = children
        super().__init__(f"{op}({children})")

    @property
    def key(self):
        return self.op, self.children

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


register_model(ExprType)(models.OpaqueModel)


@intrinsic(prefer_literal=True)
def init_col_from_name(typingctx, col):
    """create Column object from column name"""
    assert_bodo_error(is_overload_constant_str(col))
    col_str = get_overload_const_str(col)
    col_type = ColumnType(ExprType("col", (col_str,)))

    def codegen(context, builder, signature, args):
        return context.get_constant_null(col_type)

    return col_type(col), codegen


@overload(F.col, no_unliteral=True)
@overload(F.column, no_unliteral=True)
def overload_f_col(col):
    """create a Column object from column name"""
    if not is_overload_constant_str(col):
        raise BodoError(
            f"pyspark.sql.functions.col(): column name should be a constant string, not {col}"
        )

    return lambda col: init_col_from_name(col)  # pragma: no cover


@intrinsic
def init_f_sum(typingctx, col):
    """create a Column object for F.sum"""
    col_type = ColumnType(ExprType("sum", (col.expr,)))

    def codegen(context, builder, signature, args):
        return context.get_constant_null(col_type)

    return col_type(col), codegen


@overload(F.sum, no_unliteral=True)
def overload_f_sum(col):
    """create a Column object for F.sum"""
    if is_overload_constant_str(col):
        return lambda col: init_f_sum(init_col_from_name(col))  # pragma: no cover

    if not isinstance(col, ColumnType):
        raise BodoError(
            f"pyspark.sql.functions.sum(): input should be a Column object or a constant string, not {col}"
        )

    return lambda col: init_f_sum(col)  # pragma: no cover


def _get_col_name(col):
    """get output column name for Column value 'col'"""
    if isinstance(col, str):
        return col

    # TODO: generate code for other Column exprs
    _check_column(col)
    return _get_col_name_exr(col.expr)


def _get_col_name_exr(expr):
    """get output column name for Expr 'expr'"""

    if expr.op == "sum":
        return f"sum({_get_col_name_exr(expr.children[0])})"

    assert expr.op == "col"
    return expr.children[0]


def _gen_col_code(col, df_type):
    """generate code for Column value 'col' for dataframe 'df_type'"""
    if isinstance(col, str):
        return _gen_col_code_colname(col, df_type)

    _check_column(col)
    return _gen_col_code_expr(col.expr, df_type)


def _gen_col_code_expr(expr, df_type):
    """generate code for Expr 'expr'"""
    # TODO: generate code for other Column exprs

    if expr.op == "col":
        return _gen_col_code_colname(expr.children[0], df_type)

    if expr.op == "sum":
        in_out, in_func_text = _gen_col_code_expr(expr.children[0], df_type)
        i = ir_utils.next_label()
        func_text = f"  A{i} = np.asarray([bodo.libs.array_ops.array_op_sum({in_out}, True, 0)])\n"
        return f"A{i}", in_func_text + func_text


def _gen_col_code_colname(col_name, df_type):
    """generate code for referencing column name 'col_name' of dataframe 'df_type'"""
    # str column name case
    col_ind = df_type.columns.index(col_name)
    i = ir_utils.next_label()
    func_text = (
        f"  A{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {col_ind})\n"
    )
    return f"A{i}", func_text


def _check_column(col):
    """raise error if 'col' is not a Column"""
    if not isinstance(col, ColumnType):
        raise BodoError("Column object expected")
