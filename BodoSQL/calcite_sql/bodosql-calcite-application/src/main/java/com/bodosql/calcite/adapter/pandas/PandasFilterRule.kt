package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.rel.logical.BodoLogicalFilter
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Filter

class PandasFilterRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    BodoLogicalFilter::class.java,
                    Convention.NONE,
                    PandasRel.CONVENTION,
                    "PandasFilterRule",
                )
                .withRuleFactory { config -> PandasFilterRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val filter = rel as Filter
        return PandasFilter.create(
            rel.cluster,
            convert(
                filter.input,
                filter.input.traitSet
                    .replace(PandasRel.CONVENTION),
            ),
            filter.condition,
        )
    }
}
