"""
Contains information used to access the Java package from JPype.
"""
import os

import jpype

# Java Classes used by the connector
BodoReaderIcebergReaderClass = None
DateTypeClass = None
TimestampTypeClass = None
LiteralClass = None
IntegerClass = None
LongClass = None
FloatClass = None
DoubleClass = None
BooleanClass = None
LinkedListClass = None
OpEnumClass = None


# Dictionary mapping table info -> Reader obj
table_dict = {}


def launch_jvm():
    my_dir = os.path.dirname(os.path.abspath(__file__))
    base_jar_src = my_dir + "/../iceberg-reader/target/"
    # TODO: Fix the path for a deployment
    class_path_list = [f"{base_jar_src}/{name}" for name in os.listdir(base_jar_src)]
    if jpype.isJVMStarted() and not jpype.isThreadAttachedToJVM():
        # Note: If the JVM is already started its too late to update the class path.
        jpype.attachThreadToJVM()
        jpype.java.lang.Thread.currentThread().setContextClassLoader(
            jpype.java.lang.ClassLoader.getSystemClassLoader()
        )
    if not jpype.isJVMStarted():
        jpype.startJVM("-ea", classpath=class_path_list, convertStrings=False)


def get_bodo_iceberg_reader_class():
    """
    Wrapper around getting the constructor to
    load the BodoReaderIcebergReaderClass on first request.
    """
    global BodoReaderIcebergReaderClass

    # We may need to launch the JVM if we haven't loaded the class.
    if BodoReaderIcebergReaderClass is None:
        launch_jvm()
        BodoReaderIcebergReaderClass = jpype.JClass(
            "com.bodo.iceberg.BodoIcebergReader"
        )
    return BodoReaderIcebergReaderClass


def get_date_type_class():
    """
    Wrapper around getting the constructor to
    load the DateTypeClass on first request.
    """
    global DateTypeClass

    # We may need to launch the JVM if we haven't loaded the class.
    if DateTypeClass is None:
        launch_jvm()
        DateTypeClass = jpype.JClass("org.apache.iceberg.types.Types.DateType")
    return DateTypeClass


def get_timestamp_type_class():
    """
    Wrapper around getting the constructor to
    load the DateTypeClass on first request.
    """
    global TimestampTypeClass

    # We may need to launch the JVM if we haven't loaded the class.
    if TimestampTypeClass is None:
        launch_jvm()
        TimestampTypeClass = jpype.JClass(
            "org.apache.iceberg.types.Types.TimestampType"
        )
    return TimestampTypeClass


def get_literal_class():
    """
    Wrapper around getting the constructor to
    load the LiteralClass on first request.
    """
    global LiteralClass

    # We may need to launch the JVM if we haven't loaded the class.
    if LiteralClass is None:
        launch_jvm()
        LiteralClass = jpype.JClass("org.apache.iceberg.expressions.Literal")
    return LiteralClass


def get_long_class():
    """
    Wrapper around getting the constructor to
    load the LongClass on first request.
    """
    global LongClass

    # We may need to launch the JVM if we haven't loaded the class.
    if LongClass is None:
        launch_jvm()
        LongClass = jpype.JClass("java.lang.Long")
    return LongClass


def get_integer_class():
    """
    Wrapper around getting the constructor to
    load the IntegerClass on first request.
    """
    global IntegerClass

    # We may need to launch the JVM if we haven't loaded the class.
    if IntegerClass is None:
        launch_jvm()
        IntegerClass = jpype.JClass("java.lang.Integer")
    return IntegerClass


def get_float_class():
    """
    Wrapper around getting the constructor to
    load the FloatClass on first request.
    """
    global FloatClass

    # We may need to launch the JVM if we haven't loaded the class.
    if FloatClass is None:
        launch_jvm()
        FloatClass = jpype.JClass("java.lang.Float")
    return FloatClass


def get_double_class():
    """
    Wrapper around getting the constructor to
    load the DoubleClass on first request.
    """
    global DoubleClass

    # We may need to launch the JVM if we haven't loaded the class.
    if DoubleClass is None:
        launch_jvm()
        DoubleClass = jpype.JClass("java.lang.Double")
    return DoubleClass


def get_boolean_class():
    """
    Wrapper around getting the constructor to
    load the BooleanClass on first request.
    """
    global BooleanClass

    # We may need to launch the JVM if we haven't loaded the class.
    if BooleanClass is None:
        launch_jvm()
        BooleanClass = jpype.JClass("java.lang.Boolean")
    return BooleanClass


def get_linkedlist_class():
    """
    Wrapper around getting the constructor to
    load the LinkedListClass on first request.
    """
    global LinkedListClass

    # We may need to launch the JVM if we haven't loaded the class.
    if LinkedListClass is None:
        launch_jvm()
        LinkedListClass = jpype.JClass("java.util.LinkedList")
    return LinkedListClass


def get_java_list(val):
    """
    Converts a Python list to a Java ArrayList
    """
    return jpype.java.util.ArrayList(val)


def get_op_enum_class():
    """
    Wrapper around getting the constructor to
    load the OpEnumClass on first request.
    """
    global OpEnumClass

    # We may need to launch the JVM if we haven't loaded the class.
    if OpEnumClass is None:
        launch_jvm()
        OpEnumClass = jpype.JClass("com.bodo.iceberg.OpEnum")
    return OpEnumClass


def get_iceberg_java_table_reader(warehouse, schema, table):
    reader_class = get_bodo_iceberg_reader_class()
    key = (warehouse, schema, table)
    if key not in table_dict:
        table_dict[key] = reader_class(warehouse, schema, table)
    return table_dict[key]
