package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.application.Utils.AggHelpers
import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.logical.LogicalAggregate
import org.apache.calcite.sql.SqlKind

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

        val streamingTrait = PandasAggregate.getStreamingTrait(rel.groupSets, rel.aggCallList)
        val traitSet = rel.cluster.traitSet().replace(PandasRel.CONVENTION).replace(streamingTrait)
        return PandasAggregate(rel.cluster, traitSet, convert(agg.input, traitSet.replace(streamingTrait)),
            agg.groupSet, agg.groupSets, agg.aggCallList)
    }
}
