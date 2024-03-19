"""
Contains information used to lower the filters used
by Bodo in filter pushdown into a parsable Java format.

This passes a constructs the proper Java literals needed
to construct expressions. However, Java code is responsible
for constructing the proper filter pushdown expression,
including considering any Iceberg transformations.
"""
import abc
import datetime
import typing as pt

import numpy as np
import pandas as pd

from bodo_iceberg_connector.py4j_support import (
    convert_list_to_java,
    get_array_const_class,
    get_column_ref_class,
    get_filter_expr_class,
    get_literal_converter_class,
)


class Filter(metaclass=abc.ABCMeta):
    """
    Base Filter Class for Composing Filters for the Iceberg
    Java library.
    """

    @abc.abstractmethod
    def to_java(self) -> pt.Any:
        """
        Converts the filter to equivalent Java objects
        """
        pass


class ColumnRef(Filter):
    """
    Represents a column reference in a filter.
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"ref({self.name})"

    def to_java(self):
        column_ref_class = get_column_ref_class()
        return column_ref_class(self.name)


class Scalar(Filter):
    """
    Represents an Iceberg constant in a filter.
    """

    def __init__(self, value: pt.Any):
        self.value = value

    def __repr__(self):
        return f"scalar({str(self.value)})"

    def to_java(self):
        return convert_scalar(self.value)


class FilterExpr(Filter):
    """
    Represents a filter expression in a format compatible
    by the Iceberg Java library.
    """

    op: str
    args: pt.List[Filter]

    def __init__(self, op: str, args: pt.List[Filter]):
        self.op = op
        self.args = args

    def __repr__(self):
        return f"{self.op}({', '.join(map(str, self.args))})"

    @classmethod
    def default(cls):
        return cls("ALWAYS_TRUE", [])

    def to_java(self):
        filter_expr_class = get_filter_expr_class()
        return filter_expr_class(
            self.op, convert_list_to_java([arg.to_java() for arg in self.args])
        )


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
        array_const_class = get_array_const_class()
        # NOTE: Iceberg takes regular Java lists in this case, not Literal lists.
        # see predicate(Expression.Operation op, java.lang.String name,
        #               java.lang.Iterable<T> values)
        # https://iceberg.apache.org/javadoc/0.13.1/index.html?org/apache/iceberg/types/package-summary.html
        return array_const_class(convert_list_to_java(val))
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
    converter = get_literal_converter_class()
    # Convert the dt64 to integer and round down to microseconds
    num_microseconds = num_nanoseconds // 1000
    return converter.microsecondsToTimestampLiteral(int(num_microseconds))


def convert_date(val):
    """
    Convert a Python datetime.date into an Iceberg Java
    date Literal.
    """

    converter = get_literal_converter_class()
    # Convert the date_val to days
    num_days = np.datetime64(val, "D").view("int64")

    # Return the literal
    return converter.numDaysToDateLiteral(int(num_days))


def convert_long(val):
    """
    Convert a Python or Numpy integer value that may
    require a Long.
    """
    # Return the literal
    converter = get_literal_converter_class()
    return converter.asLongLiteral(int(val))


def convert_integer(val):
    """
    Convert a Numpy integer value that only
    require an Integer.
    """
    # Return the literal
    converter = get_literal_converter_class()
    return converter.asIntLiteral(int(val))


def convert_string(val):
    """
    Converts a Python string to a
    Literal with a Java string.
    """
    # Get the Java classes
    converter = get_literal_converter_class()
    return converter.asStringLiteral(val)


def convert_float32(val):
    """
    Converts a Python float32 to a
    Literal with a Java float.
    """
    # Get the Java classes
    converter = get_literal_converter_class()
    return converter.asFloatLiteral(float(val))


def convert_float64(val):
    """
    Converts a Python float or float64 to a
    Literal with a Java double.
    """
    # Get the Java classes
    converter = get_literal_converter_class()
    return converter.asDoubleLiteral(float(val))


def convert_bool(val):
    """
    Converts a Python or Numpy bool to
    a literal with a Java bool.
    """
    # Get the Java classes
    converter = get_literal_converter_class()
    return converter.asBoolLiteral(bool(val))
