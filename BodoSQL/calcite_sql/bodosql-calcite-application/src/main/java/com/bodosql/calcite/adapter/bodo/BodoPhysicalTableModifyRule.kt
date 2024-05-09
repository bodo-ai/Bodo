package com.bodosql.calcite.adapter.bodo

import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.TableModify
import org.apache.calcite.rel.logical.LogicalTableModify

class BodoPhysicalTableModifyRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    LogicalTableModify::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "BodoPhysicalTableModifyRule",
                )
                .withRuleFactory { config -> BodoPhysicalTableModifyRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val tableModify = rel as TableModify
        val traitset = rel.cluster.traitSet().replace(BodoPhysicalRel.CONVENTION)
        return BodoPhysicalTableModify(
            rel.cluster, traitset, tableModify.table!!, tableModify.catalogReader!!,
            convert(
                tableModify.input,
                tableModify.input.traitSet.replace(BodoPhysicalRel.CONVENTION),
            ),
            tableModify.operation, tableModify.updateColumnList, tableModify.sourceExpressionList,
            tableModify.isFlattened,
        )
    }
}
