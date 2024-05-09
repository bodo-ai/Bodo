package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.rel.core.Flatten
import com.bodosql.calcite.rel.logical.BodoLogicalFlatten
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule

class BodoPhysicalFlattenRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    BodoLogicalFlatten::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "BodoPhysicalFlattenRule",
                )
                .withRuleFactory { config -> BodoPhysicalFlattenRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val flatten = rel as Flatten
        val input = flatten.input
        val convention = BodoPhysicalRel.CONVENTION
        return BodoPhysicalFlatten.create(
            rel.cluster,
            convert(input, input.traitSet.replace(convention)),
            flatten.call,
            flatten.callType,
            flatten.usedColOutputs,
            flatten.repeatColumns,
        )
    }
}
