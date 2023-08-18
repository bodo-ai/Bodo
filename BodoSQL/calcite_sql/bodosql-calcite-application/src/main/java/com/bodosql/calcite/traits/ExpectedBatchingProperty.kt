package com.bodosql.calcite.traits

import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.application.utils.AggHelpers
import com.bodosql.calcite.schema.CatalogSchemaImpl
import com.bodosql.calcite.table.BodoSqlTable
import com.bodosql.calcite.table.CatalogTableImpl
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.schema.Schema
import org.apache.calcite.sql.type.ArraySqlType
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.util.ImmutableBitSet

/**
 * File that contains the global information for determine the batching properties. Most operators
 * are intended to be always or never streaming if possible, so this class provides generic interfaces
 * for most operators to get their BatchingProperty.
 *
 * In addition, some classes (e.g. Project, Aggregate) may conditionally have streaming properties
 * based on the contents of the node. Each of these should be given their own method for determining
 * operator specific behavior.
 *
 * The purpose of this class is to compress all streaming decisions into a single location and make it possible
 * to assign "global requirements" to all streaming vs non-streaming decisions (for example, never streaming with
 * type X).
 */
class ExpectedBatchingProperty {
    companion object {

        /**
         * Convert a rowType to a list of types. This is
         * done to normalize row types into nodes with more complex
         * type representations.
         *
         * @param rowType The input row type
         * @return A list of the fields as a list of types.
         */
        fun rowTypeToTypes(rowType: RelDataType): List<RelDataType> {
            // Note: Types may be lazily computed so use getType() instead of type
            return rowType.fieldList.map { f -> f.getType() }
        }

        /**
         * Is the given type a type that cannot be supported in streaming.
         *
         * @param type Column type in question.
         */
        private fun isUnsupportedStreamingType(type: RelDataType): Boolean {
            // Note we don't support arrays in streaming, but Snowflake tables may contain arrays,
            // so we must ignore them. We don't support reading arrays yet anyways.
            return (type is ArraySqlType) && (type.componentType!!.sqlTypeName != SqlTypeName.UNKNOWN)
        }

        /**
         * Get the underlying batching property.
         *
         * @param streaming Does this node want streaming.
         * @param nodeTypes List of types that are the output of the node.
         * All of these must be supported in streaming.
         */
        private fun getBatchingProperty(streaming: Boolean, nodeTypes: List<RelDataType>): BatchingProperty {
            val canStream = streaming && !nodeTypes.any { type -> isUnsupportedStreamingType(type) }
            return if (canStream) {
                BatchingProperty.STREAMING
            } else {
                BatchingProperty.SINGLE_BATCH
            }
        }

        @JvmStatic
        fun alwaysSingleBatchProperty(): BatchingProperty = getBatchingProperty(false, listOf())

        @JvmStatic
        fun streamingIfPossibleProperty(nodeTypes: List<RelDataType>): BatchingProperty {
            return getBatchingProperty(true, nodeTypes)
        }

        @JvmStatic
        fun streamingIfPossibleProperty(rowType: RelDataType): BatchingProperty {
            val nodeTypes = rowTypeToTypes(rowType)
            return streamingIfPossibleProperty(nodeTypes)
        }

        /**
         * Determine the streaming trait for either a projection or filter.
         * Streaming is not supported if any RexNodes contain a RexOver
         * (e.g. Window Function).
         *
         * @param nodes This list of nodes to check for streaming.
         *
         * @return The Batch property.
         */
        @JvmStatic
        fun projectFilterProperty(nodes: List<RexNode>): BatchingProperty {
            val canStream = !RexOver.containsOver(nodes, null)
            // Note: Types may be lazily computed so use getType() instead of type
            // Unsupported types are also only supported for the output of streaming
            // (e.g. to cross between operators/write to TableBuildBuffers), so we
            // don't need to check intermediate types.
            val nodeTypes = nodes.map { r -> r.getType() }
            return getBatchingProperty(canStream, nodeTypes)
        }

        /**
         * Determine the streaming trait that can be used for an aggregation.
         *
         * @param groupSets the grouping sets used by the aggregation node.
         * If there is more than one grouping in the sets, or the singleton
         * grouping is a no-groupby aggregation, then the aggregation
         * node is not supported with streaming.
         * @param aggCallList the aggregations being applied to the data. If
         * any of the aggregations require a filter (e.g. for a PIVOT query)
         * then the aggregation node is not supported with streaming.
         * @param rowType The rowType of the output.
         *
         * @return The Batch property.
         */
        @JvmStatic
        fun aggregateProperty(groupSets: List<ImmutableBitSet>, aggCallList: List<AggregateCall>, rowType: RelDataType): BatchingProperty {
            var canStream = RelationalAlgebraGenerator.enableGroupbyStreaming &&
                groupSets.size == 1 && groupSets[0].cardinality() != 0 &&
                !AggHelpers.aggContainsFilter(aggCallList)
            val nodeTypes = rowTypeToTypes(rowType)
            return getBatchingProperty(canStream, nodeTypes)
        }

        @JvmStatic
        fun tableReadProperty(table: BodoSqlTable?, rowType: RelDataType): BatchingProperty {
            // TODO(njriasan): Can we make this non-nullable?
            val canStream = (table != null) && (table.dbType.equals("PARQUET") || table.dbType.equals("ICEBERG"))
            val nodeTypes = rowTypeToTypes(rowType)
            return getBatchingProperty(canStream, nodeTypes)
        }

        @JvmStatic
        fun tableModifyProperty(table: BodoSqlTable, inputRowType: RelDataType): BatchingProperty {
            // We can stream if it's a Snowflake write via the Snowflake Catalog.
            val canStream = (table is CatalogTableImpl) && (table.dbType.equals("SNOWFLAKE"))
            val nodeTypes = rowTypeToTypes(inputRowType)
            return getBatchingProperty(canStream, nodeTypes)
        }

        @JvmStatic
        fun tableCreateProperty(schema: Schema, inputRowType: RelDataType): BatchingProperty {
            val canStream = (schema is CatalogSchemaImpl) && (schema.dbType.equals("SNOWFLAKE"))
            val nodeTypes = rowTypeToTypes(inputRowType)
            return getBatchingProperty(canStream, nodeTypes)
        }
    }
}
