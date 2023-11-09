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


def error_to_string(e):
    """
    Convert a error from our calcite application into a string message,
    if the error is a Py4JJavaError. Otherwise, default to "str(e)"
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
