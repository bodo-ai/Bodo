package com.bodosql.calcite.adapter.bodo

import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.logical.LogicalTargetTableScan

class PandasTargetTableScanRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    LogicalTargetTableScan::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "PandasTargetTableScanRule",
                )
                .withRuleFactory { config -> PandasTargetTableScanRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val scan = rel as LogicalTargetTableScan
        val traitSet = rel.traitSet.replace(BodoPhysicalRel.CONVENTION)
        return PandasTargetTableScan(rel.cluster, traitSet, scan.table!!)
    }
}
