"""
Common location for importing all java classes from Py4j. This is used so they
can be imported in multiple locations.
"""

from typing import Any

from mpi4py import MPI

import bodo
from bodosql.py4j_gateway import configure_java_logging, get_gateway

error = None
# Based on my understanding of the Py4J Memory model, it should be safe to just
# Create/use java objects in much the same way as we did with jpype.
# https://www.py4j.org/advanced_topics.html#py4j-memory-model
saw_error = False
msg = ""
gateway = get_gateway()
if bodo.get_rank() == 0:
    try:
        # Note: Although this isn't used it must be imported.
        SnowflakeDriver = gateway.jvm.net.snowflake.client.jdbc.SnowflakeDriver
        # Note: We call this JavaEntryPoint so its clear the Python code enters java
        # and the class is named PythonEntryPoint to make it clear the Java code
        # is being entered from Python.
        JavaEntryPoint = gateway.jvm.com.bodosql.calcite.application.PythonEntryPoint
        # Initialize logging. Must be done after importing all classes to ensure
        # JavaEntryPoint is available.
        configure_java_logging(bodo.user_logging.get_verbose_level())
    except Exception as e:
        saw_error = True
        msg = str(e)
else:
    JavaEntryPoint = None

comm = MPI.COMM_WORLD
saw_error = comm.bcast(saw_error)
msg = comm.bcast(msg)
if saw_error:
    raise ValueError(msg)


def build_java_array_list(elems: list[Any]):
    if bodo.get_rank() == 0:
        output_list = JavaEntryPoint.buildArrayList()
        for elem in elems:
            JavaEntryPoint.appendToArrayList(output_list, elem)
        return output_list


def build_java_hash_map(d: dict[Any, Any]):
    if bodo.get_rank() == 0:
        output_map = JavaEntryPoint.buildMap()
        for key, value in d.items():
            JavaEntryPoint.mapPut(output_map, key, value)
        return output_map


def build_java_properties(d: dict[str, str]):
    if bodo.get_rank() == 0:
        output_map = JavaEntryPoint.buildProperties()
        for key, value in d.items():
            JavaEntryPoint.setProperty(output_map, key, value)
        return output_map
