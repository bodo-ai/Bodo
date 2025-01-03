"""
Common location for importing all java classes from Py4j. This is used so they
can be imported in multiple locations.
"""

import bodo
from bodo.libs.distributed_api import bcast_scalar
from bodo.utils.typing import BodoError
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
        RelationalAlgebraGeneratorClass = (
            gateway.jvm.com.bodosql.calcite.application.RelationalAlgebraGenerator
        )
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
    RelationalAlgebraGeneratorClass = None
    SnowflakeDriver = None
    JavaEntryPoint = None

saw_error = bcast_scalar(saw_error)
msg = bcast_scalar(msg)
if saw_error:
    raise BodoError(msg)
