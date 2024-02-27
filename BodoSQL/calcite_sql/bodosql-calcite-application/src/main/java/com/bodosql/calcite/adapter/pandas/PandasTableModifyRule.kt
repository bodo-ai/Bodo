package com.bodosql.calcite.adapter.pandas

import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.TableModify
import org.apache.calcite.rel.logical.LogicalTableModify

class PandasTableModifyRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    LogicalTableModify::class.java,
                    Convention.NONE,
                    PandasRel.CONVENTION,
                    "PandasTableModifyRule",
                )
                .withRuleFactory { config -> PandasTableModifyRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val tableModify = rel as TableModify
        val traitset = rel.cluster.traitSet().replace(PandasRel.CONVENTION)
        return PandasTableModify(
            rel.cluster, traitset, tableModify.table!!, tableModify.catalogReader!!,
            convert(
                tableModify.input,
                tableModify.input.traitSet.replace(PandasRel.CONVENTION),
            ),
            tableModify.operation, tableModify.updateColumnList, tableModify.sourceExpressionList,
            tableModify.isFlattened,
        )
    }
}
