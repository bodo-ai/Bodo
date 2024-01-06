# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Common location for importing all java classes from Py4j. This is used so they
can be imported in multiple locations.
"""
import bodo
from bodo.libs.distributed_api import bcast_scalar
from bodo.utils.typing import BodoError
from bodosql.py4j_gateway import get_gateway

error = None
# Based on my understanding of the Py4J Memory model, it should be safe to just
# Create/use java objects in much the same way as we did with jpype.
# https://www.py4j.org/advanced_topics.html#py4j-memory-model
saw_error = False
msg = ""
gateway = get_gateway()
if bodo.get_rank() == 0:
    try:
        ArrayListClass = gateway.jvm.java.util.ArrayList
        ColumnDataEnumClass = (
            gateway.jvm.com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType
        )
        ColumnClass = gateway.jvm.com.bodosql.calcite.table.BodoSQLColumnImpl
        ColumnDataTypeClass = gateway.jvm.com.bodosql.calcite.table.ColumnDataTypeInfo
        LocalTableClass = gateway.jvm.com.bodosql.calcite.table.LocalTable
        LocalSchemaClass = gateway.jvm.com.bodosql.calcite.schema.LocalSchema
        RelationalAlgebraGeneratorClass = (
            gateway.jvm.com.bodosql.calcite.application.RelationalAlgebraGenerator
        )
        PropertiesClass = gateway.jvm.java.util.Properties
        SnowflakeCatalogClass = gateway.jvm.com.bodosql.calcite.catalog.SnowflakeCatalog
        BodoTZInfoClass = gateway.jvm.org.apache.calcite.sql.type.BodoTZInfo
        # Note: Although this isn't used it must be imported.
        SnowflakeDriver = gateway.jvm.net.snowflake.client.jdbc.SnowflakeDriver
        CommonsExceptionUtilsClass = (
            gateway.jvm.org.apache.commons.lang3.exception.ExceptionUtils
        )
        PythonLoggersClass = gateway.jvm.com.bodosql.calcite.application.PythonLoggers
    except Exception as e:
        saw_error = True
        msg = str(e)
else:
    ArrayListClass = None
    ColumnDataEnumClass = None
    ColumnClass = None
    ColumnDataTypeClass = None
    LocalTableClass = None
    LocalSchemaClass = None
    RelationalAlgebraGeneratorClass = None
    PropertiesClass = None
    SnowflakeCatalogClass = None
    BodoTZInfoClass = None
    SnowflakeDriver = None
    CommonsExceptionUtilsClass = None
    PythonLoggersClass = None

saw_error = bcast_scalar(saw_error)
msg = bcast_scalar(msg)
if saw_error:
    raise BodoError(msg)
