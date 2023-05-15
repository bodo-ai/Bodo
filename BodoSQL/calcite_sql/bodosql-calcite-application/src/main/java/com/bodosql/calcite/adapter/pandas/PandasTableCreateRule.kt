package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.LogicalTableCreate

class PandasTableCreateRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG = Config.INSTANCE
            .withConversion(
                LogicalTableCreate::class.java, Convention.NONE, PandasRel.CONVENTION,
                "PandasTableCreateRule")
            .withRuleFactory { config -> PandasTableCreateRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val create = rel as LogicalTableCreate
        val traitSet = rel.traitSet.replace(PandasRel.CONVENTION).replace(BatchingProperty.SINGLE_BATCH)
        return PandasTableCreate(rel.cluster, traitSet,
            convert(create.input,
                create.input.traitSet.replace(PandasRel.CONVENTION).replace(BatchingProperty.SINGLE_BATCH)),
            create.schema, create.tableName, create.isReplace, create.createTableType, create.schemaPath)
    }
}
