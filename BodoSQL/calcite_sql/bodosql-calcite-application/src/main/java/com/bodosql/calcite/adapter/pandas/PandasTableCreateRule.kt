package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.traits.ExpectedBatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty.Companion.tableCreateProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.LogicalTableCreate

class PandasTableCreateRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG = Config.INSTANCE
            .withConversion(
                LogicalTableCreate::class.java,
                Convention.NONE,
                PandasRel.CONVENTION,
                "PandasTableCreateRule",
            )
            .withRuleFactory { config -> PandasTableCreateRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val create = rel as LogicalTableCreate
        // The output of LogicalTableCreate is always SingleBatch.
        val outputBatchingProperty = ExpectedBatchingProperty.alwaysSingleBatchProperty()
        val traitSet = rel.traitSet.replace(PandasRel.CONVENTION).replace(outputBatchingProperty)

        // Case when output is a Snowflake table
        val createSchema = create.schema
        // The input depends on if our destination supports streaming.
        // Note: Types may be lazily computed so use getRowType() instead of rowType
        val inputBatchingProperty = tableCreateProperty(createSchema, create.input.getRowType())

        return PandasTableCreate(
            rel.cluster,
            traitSet,
            convert(
                create.input,
                create.input.traitSet.replace(PandasRel.CONVENTION).replace(inputBatchingProperty),
            ),
            create.schema,
            create.tableName,
            create.isReplace,
            create.createTableType,
            create.schemaPath,
        )
    }
}
