package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.logical.LogicalAggregate
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.type.SqlTypeName

class PandasAggregateRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                LogicalAggregate::class.java, Convention.NONE, PandasRel.CONVENTION,
                "PandasAggregateRule"
            )
            .withRuleFactory { config -> PandasAggregateRule(config) }

        fun isValidNode(node: Aggregate): Boolean {
            for (aggCall: AggregateCall in node.aggCallList) {
                if (aggCall.aggregation.kind == SqlKind.LISTAGG && aggCall.argList.size == 1) {
                    return false;
                }
            }
            return true;
        }

    }

    override fun convert(rel: RelNode): RelNode? {
        val agg = rel as Aggregate


        if (!PandasAggregateRule.isValidNode(agg)) {
            return null;
        }

        val streamingTrait = getStreamingTrait()
        val traitSet = rel.cluster.traitSet().replace(PandasRel.CONVENTION).replace(streamingTrait)
        return PandasAggregate(rel.cluster, traitSet, convert(agg.input, traitSet.replace(streamingTrait)),
            agg.groupSet, agg.groupSets, agg.aggCallList)
    }

    /**
     * Determine the streaming Trait for a newly created Groupby.
     *
     * TODO: restrict this to the operations that are actually supported.
     * e.g.: require a groupby (not just select MAX(A) from table)
     * don't support filters (e.g. the paths that would require groupby.apply)
     */
    fun getStreamingTrait(): BatchingProperty {
        return if (RelationalAlgebraGenerator.enableGroupbyStreaming) BatchingProperty.STREAMING
        else BatchingProperty.SINGLE_BATCH
    }
}
