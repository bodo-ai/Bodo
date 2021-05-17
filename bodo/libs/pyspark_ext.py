# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Support for PySpark APIs in Bodo JIT functions
"""
import pyspark
from numba.core import types
from numba.core.imputils import lower_constant
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    models,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)


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
