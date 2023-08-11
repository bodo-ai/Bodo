package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.table.BodoSqlTable
import com.bodosql.calcite.traits.ExpectedBatchingProperty.Companion.tableReadProperty
import org.apache.calcite.plan.Convention
import org.apache.calcite.prepare.RelOptTableImpl
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.logical.LogicalTableScan

class PandasTableScanRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                LogicalTableScan::class.java, Convention.NONE, PandasRel.CONVENTION,
                "PandasTableScanRule")
            .withRuleFactory { config -> PandasTableScanRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val bodoSqlTable = (rel.table as? RelOptTableImpl)?.table() as? BodoSqlTable
        // Note: Types may be lazily computed so use getRowType() instead of rowType
        val batchingProperty = tableReadProperty(bodoSqlTable, rel.getRowType())
        val scan = rel as LogicalTableScan
        val traitSet = rel.traitSet.replace(PandasRel.CONVENTION).replace(batchingProperty)
        return PandasTableScan(rel.cluster, traitSet, scan.table!!)
    }
}
