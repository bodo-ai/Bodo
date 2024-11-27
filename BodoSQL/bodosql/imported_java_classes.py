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
        ArrayListClass = gateway.jvm.java.util.ArrayList
        HashMapClass = gateway.jvm.java.util.HashMap
        ColumnDataEnum = (
            gateway.jvm.com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType
        )
        ColumnClass = gateway.jvm.com.bodosql.calcite.table.BodoSQLColumnImpl
        ColumnDataTypeClass = gateway.jvm.com.bodosql.calcite.table.ColumnDataTypeInfo
        LocalTableClass = gateway.jvm.com.bodosql.calcite.table.LocalTable
        LocalSchemaClass = gateway.jvm.com.bodosql.calcite.schema.LocalSchema
        PandasCodeSqlPlanPairClass = (
            gateway.jvm.com.bodosql.calcite.schema.PandasCodeSqlPlanPair
        )
        RelationalAlgebraGeneratorClass = (
            gateway.jvm.com.bodosql.calcite.application.RelationalAlgebraGenerator
        )
        PropertiesClass = gateway.jvm.java.util.Properties
        SnowflakeCatalogClass = gateway.jvm.com.bodosql.calcite.catalog.SnowflakeCatalog
        # Note: Although this isn't used it must be imported.
        SnowflakeDriver = gateway.jvm.net.snowflake.client.jdbc.SnowflakeDriver
        CommonsExceptionUtilsClass = (
            gateway.jvm.org.apache.commons.lang3.exception.ExceptionUtils
        )
        PythonLoggersClass = gateway.jvm.com.bodosql.calcite.application.PythonLoggers
        FileSystemCatalogClass = (
            gateway.jvm.com.bodosql.calcite.catalog.FileSystemCatalog
        )
        WriteTargetEnum = gateway.jvm.com.bodosql.calcite.application.write.WriteTarget.WriteTargetEnum
        TabularCatalogClass = gateway.jvm.com.bodosql.calcite.catalog.TabularCatalog
        BodoGlueCatalogClass = gateway.jvm.com.bodosql.calcite.catalog.BodoGlueCatalog
        # Initialize logging. Must be done after importing all classes to ensure
        # PythonLoggersClass is available.
        configure_java_logging(bodo.user_logging.get_verbose_level())
    except Exception as e:
        saw_error = True
        msg = str(e)
else:
    ArrayListClass = None
    HashMapClass = None
    ColumnDataEnum = None
    ColumnClass = None
    ColumnDataTypeClass = None
    LocalTableClass = None
    LocalSchemaClass = None
    PandasCodeSqlPlanPairClass = None
    RelationalAlgebraGeneratorClass = None
    PropertiesClass = None
    SnowflakeCatalogClass = None
    SnowflakeDriver = None
    CommonsExceptionUtilsClass = None
    PythonLoggersClass = None
    FileSystemCatalogClass = None
    WriteTargetEnum = None
    TabularCatalogClass = None
    BodoGlueCatalogClass = None

saw_error = bcast_scalar(saw_error)
msg = bcast_scalar(msg)
if saw_error:
    raise BodoError(msg)
