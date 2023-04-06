# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
BodoSQL utils used to help construct Python code.
"""
import py4j


class BodoSQLWarning(Warning):
    """
    Warning class for BodoSQL-related potential issues such as being
    unable to properly cache literals in namedParameters.
    """


def java_error_to_msg(e):
    """
    Convert a error from our calcite application into a string message.
    """
    if isinstance(e, py4j.protocol.Py4JJavaError):
        message = e.java_exception.getMessage()
    elif isinstance(e, py4j.protocol.Py4JNetworkError):
        message = "Unexpected Py4J Network Error: " + str(e)
    elif isinstance(e, py4j.protocol.Py4JError):
        message = "Unexpected Py4J Error: " + str(e)
    else:
        message = "Unexpected Internal Error:" + str(e)
    return message
