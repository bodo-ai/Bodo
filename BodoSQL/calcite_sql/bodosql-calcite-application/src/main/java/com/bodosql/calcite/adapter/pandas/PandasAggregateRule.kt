package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.rel.logical.BodoLogicalAggregate
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.sql.SqlKind

class PandasAggregateRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                BodoLogicalAggregate::class.java, Convention.NONE, PandasRel.CONVENTION,
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


        if (!isValidNode(agg)) {
            return null;
        }
        // Note: Types may be lazily computed so use getRowType() instead of rowType
        val batchingProperty = ExpectedBatchingProperty.aggregateProperty(rel.groupSets, rel.aggCallList, rel.getRowType())
        val traitSet = rel.cluster.traitSet().replace(PandasRel.CONVENTION).replace(batchingProperty)
        return PandasAggregate(rel.cluster, traitSet, convert(agg.input, traitSet.replace(batchingProperty)),
            agg.groupSet, agg.groupSets, agg.aggCallList)
    }
}
