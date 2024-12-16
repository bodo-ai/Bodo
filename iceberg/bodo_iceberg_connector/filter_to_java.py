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

# Iceberg hard codes the limit for IN filters to 200
# see InclusiveMetricsEvaluator in Iceberg.
ICEBERG_IN_FILTER_LIMIT = 200


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
    args: list[Filter]

    def __init__(self, op: str, args: list[Filter]):
        self.op = op
        self.args = args

    def __repr__(self):
        return f"{self.op}({', '.join(map(str, self.args))})"

    @classmethod
    def default(cls):
        return cls("ALWAYS_TRUE", [])

    def to_java(self):
        filter_expr_class = get_filter_expr_class()
        individual_filter = filter_expr_class(
            self.op, convert_list_to_java([arg.to_java() for arg in self.args])
        )
        if (
            self.op in ("IN", "NOT IN")
            and len(self.args[1].value) > ICEBERG_IN_FILTER_LIMIT
        ):
            # Iceberg doesn't check anything once the limit is reached.
            in_args = self.args[1].value
            min_val = min(in_args)
            max_val = max(in_args)
            if self.op == "IN":
                min_filter = FilterExpr(">=", [self.args[0], Scalar(min_val)])
                max_filter = FilterExpr("<=", [self.args[0], Scalar(max_val)])
                bounds_filter = FilterExpr("AND", [min_filter, max_filter]).to_java()
            else:
                min_filter = FilterExpr("<", [self.args[0], Scalar(min_val)])
                max_filter = FilterExpr(">", [self.args[0], Scalar(max_val)])
                bounds_filter = FilterExpr("OR", [min_filter, max_filter]).to_java()
            return filter_expr_class(
                "AND", convert_list_to_java([individual_filter, bounds_filter])
            )
        else:
            return individual_filter


def convert_scalar(val):
    """
    Converts a Python scalar into its Java Iceberg Literal
    representation.
    """

    # Need to import Bodo here in order to check for bodo.Time
    import bodo

    if isinstance(val, pd.Timestamp):
        # Note timestamp is subclass of datetime.date,
        # so this must be visited first.
        return convert_timestamp(val)
    elif isinstance(val, datetime.date):
        return convert_date(val)
    elif isinstance(val, (bool, np.bool_)):
        # This needs to go before int because it may be a subclass
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
    elif isinstance(val, list) or isinstance(val, np.ndarray):
        converted_val = [convert_scalar(v) for v in val]
        array_const_class = get_array_const_class()
        # NOTE: Iceberg takes regular Java lists in this case, not Literal lists
        # see predicate(Expression.Operation op, java.lang.String name,
        #               java.lang.Iterable<T> values)
        # https://iceberg.apache.org/javadoc/0.13.1/index.html?org/apache/iceberg/types/package-summary.html
        return array_const_class(convert_list_to_java(converted_val))
    elif isinstance(val, pd.core.arrays.ExtensionArray):
        converted_val = []
        null_vals = pd.isna(val)
        for idx, scalar_val in enumerate(val):
            if null_vals[idx]:
                raise RuntimeError(
                    "Impossible state in bodo_iceberg_connector/filter_to_java.py's convert_scalar(): null value in ExtensionArray."
                )
            else:
                converted_val.append(convert_scalar(scalar_val))
        array_const_class = get_array_const_class()
        # NOTE: Iceberg takes regular Java lists in this case, not Literal lists. (see above note)
        return array_const_class(convert_list_to_java(converted_val))
    elif isinstance(val, bytes):
        return convert_bytes(val)
    elif isinstance(val, bodo.Time):
        return convert_time(val)
    else:
        raise NotImplementedError(
            f"Unsupported scalar type in iceberg filter pushdown: {type(val)}"
        )


def convert_time(val):
    """
    Convert a Bodo Time into an Iceberg Java
    Time Literal.
    """
    import bodo

    assert isinstance(val, bodo.Time)
    converter = get_literal_converter_class()
    # All Iceberg times are in microseconds according to
    # https://iceberg.apache.org/spec/#primitive-types.
    # Looking at the constructor for out time type in time_ext,
    # it seems like we
    # always use nanosecond precision, even if we explicitly set
    # the precision to something other than 9. I think this is
    # an separate issue, so I'm going to assume the precision
    # in the type  is correct and convert it to microseconds
    # accordingly.
    val_as_microseconds = val.value * 10 ** (6 - val.precision)
    return converter.asTimeLiteral(int(val_as_microseconds))


def convert_bytes(val):
    """
    Convert a Python bytes object into an Iceberg Java
    binary Literal.
    """
    converter = get_literal_converter_class()
    return converter.asBinaryLiteral(val)


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
