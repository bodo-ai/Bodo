"""
BodoSQL utils used to help construct Python code.
"""

import py4j

from bodo.utils.typing import BodoError
from bodosql.imported_java_classes import JavaEntryPoint


class BodoSQLWarning(Warning):
    """
    Warning class for BodoSQL-related potential issues such as being
    unable to properly cache literals in namedParameters.
    """


def error_to_string(e: Exception) -> str:
    """
    Convert a error from our calcite application into a string message.
    There is detailed error handling for a Py4JJavaError. Otherwise
    we default to `str(e)`.

    Args:
        e (Exception): The Py4j raised exception.

    Returns:
        str: A string message of the error.
    """
    if isinstance(e, BodoError):
        # If called from a BodoError we should just return the message
        message = e.msg
    elif isinstance(e, py4j.protocol.Py4JJavaError):
        java_exception = e.java_exception
        message = JavaEntryPoint.getThrowableMessage(java_exception)
        if not message:
            # If the message is None, rather than return None we should provide a stack
            # trace.
            msg_header = (
                "No message found for java exception. Displaying stack trace:\n"
            )
            msg_body = JavaEntryPoint.getStackTrace(java_exception)
            message = msg_header + msg_body
        # Append the cause if it exists
        cause = JavaEntryPoint.getThrowableCause(java_exception)
        while cause is not None:
            cause_message = JavaEntryPoint.getThrowableMessage(cause)
            if cause_message is not None:
                message += f"\nCaused by: {cause_message}"
            cause = JavaEntryPoint.getThrowableCause(cause)
    elif isinstance(e, py4j.protocol.Py4JNetworkError):
        message = "Unexpected Py4J Network Error: " + str(e)
    elif isinstance(e, py4j.protocol.Py4JError):
        message = "Unexpected Py4J Error: " + str(e)
    else:
        message = "Unexpected Internal Error: " + str(e)
    return message


# Used for testing purposes
def levenshteinDistance(s1, s2, max_dist=-1):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_

    if max_dist != -1:
        return min(max_dist, distances[-1])
    return distances[-1]
