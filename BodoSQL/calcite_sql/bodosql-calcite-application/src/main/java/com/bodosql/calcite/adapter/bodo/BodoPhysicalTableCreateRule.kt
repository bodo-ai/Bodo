package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.rel.logical.BodoLogicalTableCreate
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule

class BodoPhysicalTableCreateRule private constructor(
    config: Config,
) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG =
            Config.INSTANCE
                .withConversion(
                    BodoLogicalTableCreate::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "BodoPhysicalTableCreateRule",
                ).withRuleFactory { config -> BodoPhysicalTableCreateRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val create = rel as BodoLogicalTableCreate
        val traitSet = rel.traitSet.replace(BodoPhysicalRel.CONVENTION)
        return BodoPhysicalTableCreate.create(
            rel.cluster,
            traitSet,
            convert(
                create.input,
                create.input.traitSet.replace(BodoPhysicalRel.CONVENTION),
            ),
            create.getSchema(),
            create.tableName,
            create.isReplace,
            create.createTableType,
            create.path,
            create.meta,
        )
    }
}
