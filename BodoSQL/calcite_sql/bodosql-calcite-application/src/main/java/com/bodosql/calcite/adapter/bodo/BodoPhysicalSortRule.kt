package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.rel.logical.BodoLogicalSort
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Sort

class BodoPhysicalSortRule private constructor(
    config: Config,
) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    BodoLogicalSort::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "BodoPhysicalSortRule",
                ).withRuleFactory { config -> BodoPhysicalSortRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val sort = rel as Sort
        return BodoPhysicalSort.create(
            convert(
                sort.input,
                sort.input.traitSet.replace(BodoPhysicalRel.CONVENTION),
            ),
            sort.collation,
            sort.offset,
            sort.fetch,
        )
    }
}
