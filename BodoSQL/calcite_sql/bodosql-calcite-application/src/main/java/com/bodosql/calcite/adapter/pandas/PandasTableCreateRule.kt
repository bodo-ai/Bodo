package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.rel.logical.BodoLogicalTableCreate
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule

class PandasTableCreateRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG =
            Config.INSTANCE
                .withConversion(
                    BodoLogicalTableCreate::class.java,
                    Convention.NONE,
                    PandasRel.CONVENTION,
                    "PandasTableCreateRule",
                )
                .withRuleFactory { config -> PandasTableCreateRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val create = rel as BodoLogicalTableCreate
        val traitSet = rel.traitSet.replace(PandasRel.CONVENTION)
        return PandasTableCreate(
            rel.cluster,
            traitSet,
            convert(
                create.input,
                create.input.traitSet.replace(PandasRel.CONVENTION),
            ),
            create.schema,
            create.tableName,
            create.isReplace,
            create.createTableType,
            create.path,
            create.meta,
        )
    }
}
