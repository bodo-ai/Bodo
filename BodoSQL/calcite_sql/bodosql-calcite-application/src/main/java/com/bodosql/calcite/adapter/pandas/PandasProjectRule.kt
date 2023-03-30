package com.bodosql.calcite.adapter.pandas

import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.logical.LogicalProject

class PandasProjectRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                LogicalProject::class.java, Convention.NONE, PandasRel.CONVENTION,
                "PandasProjectRule")
            .withRuleFactory { config -> PandasProjectRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val project = rel as Project
        return PandasProject.create(
            convert(project.input,
                project.input.traitSet
                    .replace(PandasRel.CONVENTION)),
            project.projects,
            project.rowType)
    }
}
