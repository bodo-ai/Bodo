package com.bodosql.calcite.traits

import com.bodosql.calcite.application.operatorTables.ArrayOperatorTable
import com.bodosql.calcite.application.operatorTables.JsonOperatorTable
import com.bodosql.calcite.application.utils.AggHelpers
import com.bodosql.calcite.schema.CatalogSchema
import com.bodosql.calcite.table.BodoSqlTable
import com.bodosql.calcite.table.CatalogTable
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.schema.Schema
import org.apache.calcite.sql.SqlAggFunction
import org.apache.calcite.sql.`fun`.SqlAggOperatorTable
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
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
        @JvmStatic
        fun rowTypeToTypes(rowType: RelDataType): List<RelDataType> {
            // Note: Types may be lazily computed so use getType() instead of type
            return rowType.fieldList.map { f -> f.getType() }
        }

        /**
         * Get the underlying batching property.
         *
         * @param streaming Does this node want streaming.
         * @param nodeTypes List of types that are the output of the node.
         * All of these must be supported in streaming.
         */
        private fun getBatchingProperty(streaming: Boolean, nodeTypes: List<RelDataType>): BatchingProperty {
            return if (streaming) {
                BatchingProperty.STREAMING
            } else {
                BatchingProperty.SINGLE_BATCH
            }
        }

        @JvmStatic
        fun alwaysSingleBatchProperty(): BatchingProperty = getBatchingProperty(false, listOf())

        @JvmStatic
        fun streamingIfPossibleProperty(rowType: RelDataType): BatchingProperty {
            val nodeTypes = rowTypeToTypes(rowType)
            return getBatchingProperty(true, nodeTypes)
        }

        /**
         * Determine the streaming trait for a filter.
         * Streaming is not supported if the RexNode contain a RexOver
         * (e.g. Window Function).
         *
         * @param node The node to check for streaming.
         *
         * @return The Batch property.
         */
        @JvmStatic
        fun filterProperty(node: RexNode): BatchingProperty {
            val canStream = !RexOver.containsOver(node)
            // Note: Types may be lazily computed so use getType() instead of type
            // Unsupported types are also only supported for the output of streaming
            // (e.g. to cross between operators/write to TableBuildBuffers), so we
            // don't need to check intermediate types.
            return getBatchingProperty(canStream, listOf(node.getType()))
        }

        /**
         * Determine the streaming trait for a project.
         * Streaming is not supported if the RexNodes contain a RexOver
         * (e.g. Window Function). If all the nodes are inputRefs or
         * scalars, then we match the given property that was determined
         * based on the input (to avoid jumping to/from streaming just to prune
         * columns).
         *
         * @param nodes The nodes to check for streaming.
         * @param inputTraitSet The traitset of the input RelNode. This is used to
         * check for the desired behavior for column pruning/reordering.
         *
         * @return The Batch property.
         */
        @JvmStatic
        fun projectProperty(nodes: List<RexNode>, inputBatchingProperty: BatchingProperty): BatchingProperty {
            val canStream = if (nodes.all { r -> r is RexInputRef || r is RexLiteral }) {
                columnPruningIsStreaming(inputBatchingProperty)
            } else {
                !RexOver.containsOver(nodes, null)
            }
            // Note: Types may be lazily computed so use getType() instead of type
            // Unsupported types are also only supported for the output of streaming
            // (e.g. to cross between operators/write to TableBuildBuffers), so we
            // don't need to check intermediate types.
            val nodeTypes = nodes.map { r -> r.getType() }
            return getBatchingProperty(canStream, nodeTypes)
        }

        /**
         * Determine if we are just pruning columns and or adding
         * literals, do we want to use Streaming.
         */
        private fun columnPruningIsStreaming(inputBatchingProperty: BatchingProperty): Boolean {
            return inputBatchingProperty.satisfies(BatchingProperty.STREAMING)
        }

        @JvmStatic
        val unsupportedAggregates = setOf(
            SqlStdOperatorTable.PERCENTILE_CONT.name,
            SqlStdOperatorTable.PERCENTILE_DISC.name,
            SqlStdOperatorTable.MODE.name,
            SqlAggOperatorTable.LISTAGG.name,
            ArrayOperatorTable.ARRAY_UNIQUE_AGG.name,
            ArrayOperatorTable.ARRAY_AGG.name,
            JsonOperatorTable.OBJECT_AGG.name,
        )

        @JvmStatic
        private fun streamingSupportedAggFunction(a: SqlAggFunction): Boolean {
            return !unsupportedAggregates.contains(a.name)
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
            var canStream = groupSets.size == 1 && groupSets[0].cardinality() != 0 &&
                !AggHelpers.aggContainsFilter(aggCallList) &&
                aggCallList.all {
                        a ->
                    streamingSupportedAggFunction(a.aggregation)
                }

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
            val canStream = (table is CatalogTable) && (table.dbType.equals("SNOWFLAKE"))
            val nodeTypes = rowTypeToTypes(inputRowType)
            return getBatchingProperty(canStream, nodeTypes)
        }

        @JvmStatic
        fun tableCreateProperty(schema: Schema, inputRowType: RelDataType): BatchingProperty {
            val canStream = (schema is CatalogSchema) && (schema.dbType.equals("SNOWFLAKE"))
            val nodeTypes = rowTypeToTypes(inputRowType)
            return getBatchingProperty(canStream, nodeTypes)
        }
    }
}
