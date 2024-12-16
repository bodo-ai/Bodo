package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.rel.logical.BodoLogicalProject
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Project

class BodoPhysicalProjectRule private constructor(
    config: Config,
) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    BodoLogicalProject::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "BodoPhysicalProjectRule",
                ).withRuleFactory { config -> BodoPhysicalProjectRule(config) }
    }

    override fun convert(rel: RelNode): RelNode {
        val project = rel as Project
        return BodoPhysicalProject.create(
            convert(
                project.input,
                project.input.traitSet
                    .replace(BodoPhysicalRel.CONVENTION),
            ),
            project.projects,
            project.rowType,
        )
    }
}
