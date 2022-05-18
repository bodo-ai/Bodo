"""
Contains information used to lower the filters used
by Bodo in filter pushdown into a parsable Java format.

This passes a constructs the proper Java literals needed
to construct expressions. However, Java code is responsible
for constructing the proper filter pushdown expression,
including considering any Iceberg transformations.
"""
import datetime

import numpy as np
import pandas as pd

from bodoicebergconnector.bodo_apis.jpype_support import (
    get_boolean_class,
    get_date_type_class,
    get_double_class,
    get_float_class,
    get_integer_class,
    get_java_list,
    get_linkedlist_class,
    get_literal_class,
    get_long_class,
    get_op_enum_class,
    get_timestamp_type_class,
)


def op_to_enum(op, op_enum_class):
    """
    Convert an op string to an enum value.
    """
    if op == "==":
        return op_enum_class.EQ
    elif op == "!=":
        return op_enum_class.NE
    elif op == "<":
        return op_enum_class.LT
    elif op == ">":
        return op_enum_class.GT
    elif op == "<=":
        return op_enum_class.LE
    elif op == ">=":
        return op_enum_class.GE
    elif op == "in":
        return op_enum_class.IN
    elif op == "not in":
        return op_enum_class.NOT_IN
    elif op == "starts with":
        return op_enum_class.STARTS_WITH
    elif op == "not starts with":
        return op_enum_class.NOT_STARTS_WITH
    else:
        raise TypeError(f"Unsupported op {op} to enum")


def convert_expr_to_java_parsable(expr):
    """
    Takes a filter expression written in Arrow DNS format
    and converts it into an expression that can be parsed by
    Java.

    To simplify parsing we opt to represent the Java info as a single
    list of values and represent operations using an enum. For example,
    imagine we have the following operation in arrow format.

    [[('A', '=', 'dog'), ('B', '>' 2)], [('B', '=', 0)]]

    The left most value in the tuple is column name, the middle
    value is the operator, and the last value is the scalar being
    used for comparison.

    We would convert this into the following representation

    [ANDSTART 'A', EQ, 'dog', 'B', GT, 2, ANDEND, ANDSTART, 'B', EQ, 0, ANDEND]

    We don't need an or start and or end becuase ANDs are always combined by or.
    """
    linkedlist_class = get_linkedlist_class()
    op_enum_class = get_op_enum_class()
    expr_list = linkedlist_class()
    if expr is not None:
        for or_expr in expr:
            # Group all expressions that would be "and" together.
            expr_list.add(op_enum_class.AND_START)
            for and_expr in or_expr:
                # Append the column name, which should be a string.
                col_name = str(and_expr[0])
                # Create the op
                if and_expr[1:] == ("is not", "NULL"):
                    op_enum = op_enum_class.NOT_NULL
                elif and_expr[1:] == ("is", "NULL"):
                    op_enum = op_enum_class.IS_NULL
                else:
                    op_enum = op_to_enum(and_expr[1], op_enum_class)
                # Create the scalar
                iceberg_scalar = convert_scalar(and_expr[2])
                # Determine if we have an unsupported
                # scalar. If so we don't append anything to the list
                # (Java will treat this as always True).
                if iceberg_scalar is not None:
                    # Append the col_name
                    expr_list.add(col_name)
                    # Append the op
                    expr_list.add(op_enum)
                    # Append the scalar
                    expr_list.add(iceberg_scalar)
            # End and expressions
            expr_list.add(op_enum_class.AND_END)
    return expr_list


def convert_scalar(val):
    """
    Converts a Python scalar into its Java Iceberg Literal
    representation.
    """
    if isinstance(val, pd.Timestamp):
        # Note timestamp is subclass of datetime.date,
        # so this must be visited first.
        return convert_timestamp(val)
    elif isinstance(val, datetime.date):
        return convert_date(val)
    elif isinstance(val, (bool, np.bool_)):
        # This needs to go befor int because it may be a subclass
        return convert_bool(val)
    elif isinstance(val, (np.int64, int, np.uint64, np.uint32)):
        return convert_long(val)
    elif isinstance(val, (np.int32, np.int16, np.int8, np.uint8, np.uint16)):
        return convert_integer(val)
    elif isinstance(val, str):
        return convert_string(val)
    elif isinstance(val, np.float32):
        return convert_float32(val)
    elif isinstance(val, (float, np.float64)):
        return convert_float64(val)
    elif isinstance(val, np.datetime64):
        return convert_dt64(val)
    elif isinstance(val, list):
        # NOTE: Iceberg takes regular Java lists in this case, not Literal lists.
        # see predicate(Expression.Operation op, java.lang.String name,
        #               java.lang.Iterable<T> values)
        # https://iceberg.apache.org/javadoc/0.13.1/index.html?org/apache/iceberg/types/package-summary.html
        return get_java_list(val)
    else:
        # If we don't support a scalar return None and
        # we will generate a NOOP
        return None


def convert_timestamp(val):
    """
    Convert a Python Timestamp into an Iceberg Java
    Timestamp Literal.
    """
    return convert_nanoseconds(val.value)


def convert_dt64(val):
    """
    Convert a Python datetime64 into an Iceberg Java
    Timestamp Literal.
    """
    return convert_nanoseconds(val.view("int64"))


def convert_nanoseconds(num_nanoseconds):
    """
    Convert an integer in nanoseconds into an Iceberg Java
    Timestamp Literal.
    """
    # Get the Java classes
    long_class = get_long_class()
    literal_class = get_literal_class()
    timestamp_type_class = get_timestamp_type_class()
    # Convert the dt64 to integer and round down to microseconds
    num_microseconds = num_nanoseconds // 1000
    # Convert num_microseconds to java
    java_num_microseconds = long_class(num_microseconds)
    # Return the literal
    return literal_class.of(java_num_microseconds).to(
        timestamp_type_class.withoutZone()
    )


def convert_date(val):
    """
    Convert a Python datetime.date into an Iceberg Java
    date Literal.
    """
    # Get the Java classes
    long_class = get_long_class()
    literal_class = get_literal_class()
    date_type_class = get_date_type_class()
    # Convert the date_val to days
    num_days = np.datetime64(val, "D").view("int64")
    # Convert num_days to java
    java_num_days = long_class(num_days)
    # Return the literal
    return literal_class.of(java_num_days).to(date_type_class.get())


def convert_long(val):
    """
    Convert a Python or Numpy integer value that may
    require a Long.
    """
    # Get the Java classes
    long_class = get_long_class()
    literal_class = get_literal_class()
    # Return the literal
    return literal_class.of(long_class(val))


def convert_integer(val):
    """
    Convert a Numpy integer value that only
    require an Integer.
    """
    # Get the Java classes
    integer_class = get_integer_class()
    literal_class = get_literal_class()
    # Return the literal
    return literal_class.of(integer_class(val))


def convert_string(val):
    """
    Converts a Python string to a
    Literal with a Java string.
    """
    # Get the Java classes
    literal_class = get_literal_class()
    # Return the literal
    return literal_class.of(val)


def convert_float32(val):
    """
    Converts a Python float32 to a
    Literal with a Java float.
    """
    # Get the Java classes
    literal_class = get_literal_class()
    float_class = get_float_class()
    # Return the literal
    return literal_class.of(float_class(val))


def convert_float64(val):
    """
    Converts a Python float or float64 to a
    Literal with a Java double.
    """
    # Get the Java classes
    literal_class = get_literal_class()
    double_class = get_double_class()
    # Return the literal
    return literal_class.of(double_class(val))


def convert_bool(val):
    """
    Converts a Python or Numpy bool to
    a literal with a Java bool.
    """
    # Get the Java classes
    literal_class = get_literal_class()
    boolean_class = get_boolean_class()
    # Return the literal
    return literal_class.of(boolean_class(val))
