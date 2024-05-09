package com.bodosql.calcite.adapter.bodo

import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.logical.LogicalTableScan

class PandasTableScanRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    LogicalTableScan::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "PandasTableScanRule",
                )
                .withRuleFactory { config -> PandasTableScanRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val scan = rel as LogicalTableScan
        val traitSet = rel.traitSet.replace(BodoPhysicalRel.CONVENTION)
        return PandasTableScan(rel.cluster, traitSet, scan.table!!)
    }
}
