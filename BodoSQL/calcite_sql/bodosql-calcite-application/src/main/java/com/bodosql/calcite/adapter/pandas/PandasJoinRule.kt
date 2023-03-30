package com.bodosql.calcite.adapter.pandas

import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.logical.LogicalJoin

class PandasJoinRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                LogicalJoin::class.java, Convention.NONE, PandasRel.CONVENTION,
                "PandasJoinRule")
            .withRuleFactory { config -> PandasJoinRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val join = rel as Join
        val inputs = join.inputs.map { input ->
            convert(input, input.traitSet.replace(PandasRel.CONVENTION))
        }
        return PandasJoin.create(inputs[0], inputs[1], join.condition, join.joinType)
    }
}
