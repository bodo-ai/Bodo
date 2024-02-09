package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.iceberg.IcebergProject
import com.bodosql.calcite.adapter.iceberg.IcebergTableScan
import com.bodosql.calcite.adapter.iceberg.IcebergToPandasConverter
import com.bodosql.calcite.adapter.snowflake.SnowflakeAggregate
import com.bodosql.calcite.adapter.snowflake.SnowflakeFilter
import com.bodosql.calcite.adapter.snowflake.SnowflakeProject
import com.bodosql.calcite.adapter.snowflake.SnowflakeSort
import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan
import com.bodosql.calcite.adapter.snowflake.SnowflakeToPandasConverter
import com.bodosql.calcite.application.PythonLoggers
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.tools.Program

object IcebergConvertProgram : Program by ShuttleProgram(Visitor) {
    class NonIcebergTableException : Exception("Not an Iceberg-compatible Table")
    class UnsupportedIcebergOpException(message: String) : Exception(message)

    private object Visitor : RelShuttleImpl() {
        // TODO: Expand with additional checks per operator
        fun tryConvertToIceberg(node: RelNode): RelNode {
            return when (node) {
                is SnowflakeTableScan -> if (node.getCatalogTable().isIcebergTable()) {
                    IcebergTableScan(node.cluster, node.traitSet, node.table!!, node.keptColumns, node.getCatalogTable())
                } else {
                    throw NonIcebergTableException()
                }
                is SnowflakeProject -> {
                    val outInput = tryConvertToIceberg(node.input)
                    for (project in node.projects) {
                        if (project !is RexInputRef) {
                            throw UnsupportedIcebergOpException("`SnowflakeProject` with compute functions are not supported with Iceberg tables yet")
                        }
                    }
                    IcebergProject(node.cluster, node.traitSet, outInput, node.projects, node.rowType, node.getCatalogTable())
                }
                is SnowflakeFilter -> throw UnsupportedIcebergOpException("`SnowflakeFilter` is not supported for Iceberg tables yet")
                is SnowflakeSort -> throw UnsupportedIcebergOpException("`SnowflakeSort` is not supported for Iceberg tables yet")
                is SnowflakeAggregate -> throw UnsupportedIcebergOpException("`SnowflakeAggregate` is not supported for Iceberg tables yet")
                else -> throw NotImplementedError()
            }
        }

        override fun visit(node: RelNode): RelNode {
            // Special Handling of Snowflake subtree w/ SnowflakeToPandasConverter
            // as root, so we only need to check for it
            // or continue visiting other nodes
            return if (node is SnowflakeToPandasConverter) {
                try {
                    val newInput = tryConvertToIceberg(node.input)
                    IcebergToPandasConverter(node.cluster, node.traitSet, newInput)
                } catch (e: UnsupportedIcebergOpException) {
                    PythonLoggers.VERBOSE_LEVEL_TWO_LOGGER.info(e.message)
                    node
                } catch (_: NonIcebergTableException) {
                    node
                }
            } else {
                super.visit(node)
            }
        }
    }
}
