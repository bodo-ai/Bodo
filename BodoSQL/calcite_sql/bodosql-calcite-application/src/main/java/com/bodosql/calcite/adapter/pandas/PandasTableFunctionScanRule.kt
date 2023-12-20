package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.rel.logical.BodoLogicalTableFunctionScan
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.TableFunctionScan
import org.apache.calcite.rex.RexCall

class PandasTableFunctionScanRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                BodoLogicalTableFunctionScan::class.java,
                Convention.NONE,
                PandasRel.CONVENTION,
                "PandasTableFunctionScanRule",
            )
            .withRuleFactory { config -> PandasTableFunctionScanRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val fnScan = rel as TableFunctionScan
        val convention = PandasRel.CONVENTION
        val inputs = fnScan.inputs.map { convert(it, it.traitSet.replace(convention)) }
        return PandasTableFunctionScan.create(rel.cluster, inputs, fnScan.call as RexCall, fnScan.rowType)
    }
}
