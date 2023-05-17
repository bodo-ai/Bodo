package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.TableModify
import org.apache.calcite.rel.logical.LogicalTableModify

class PandasTableModifyRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(LogicalTableModify::class.java, Convention.NONE, PandasRel.CONVENTION,
                "PandasTableModifyRule")
            .withRuleFactory { config -> PandasTableModifyRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val tableModify = rel as TableModify
        val traitSet = rel.cluster.traitSetOf(PandasRel.CONVENTION).replace(BatchingProperty.SINGLE_BATCH)
        return PandasTableModify(rel.cluster, traitSet, tableModify.table!!, tableModify.catalogReader!!,
            convert(tableModify.input,
                tableModify.input.traitSet.replace(PandasRel.CONVENTION).replace(BatchingProperty.SINGLE_BATCH)),
            tableModify.operation, tableModify.updateColumnList, tableModify.sourceExpressionList,
            tableModify.isFlattened)
    }
}
