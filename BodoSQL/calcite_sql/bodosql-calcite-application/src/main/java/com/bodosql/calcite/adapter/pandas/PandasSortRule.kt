package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.rel.logical.BodoLogicalSort
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Sort

class PandasSortRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    BodoLogicalSort::class.java,
                    Convention.NONE,
                    PandasRel.CONVENTION,
                    "PandasSortRule",
                )
                .withRuleFactory { config -> PandasSortRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val sort = rel as Sort
        return PandasSort.create(
            convert(
                sort.input,
                sort.input.traitSet.replace(PandasRel.CONVENTION),
            ),
            sort.collation,
            sort.offset,
            sort.fetch,
        )
    }
}
