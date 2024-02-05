package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.rel.logical.BodoLogicalWindow
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Window

class PandasWindowRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                BodoLogicalWindow::class.java,
                Convention.NONE,
                PandasRel.CONVENTION,
                "PandasWindowRule",
            )
            .withRuleFactory { config -> PandasWindowRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val window = rel as Window
        val input = window.input
        val convention = PandasRel.CONVENTION
        return PandasWindow.create(window.cluster, window.hints, convert(input, input.traitSet.replace(convention))!!, window.constants, window.rowType, window.groups)
    }
}
