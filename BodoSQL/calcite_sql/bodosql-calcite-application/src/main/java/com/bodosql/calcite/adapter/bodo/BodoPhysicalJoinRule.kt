package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.application.utils.BodoJoinConditionUtil
import com.bodosql.calcite.rel.logical.BodoLogicalJoin
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Join
import org.apache.calcite.sql.SqlUtil
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE

class BodoPhysicalJoinRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config =
            Config.INSTANCE
                .withConversion(
                    BodoLogicalJoin::class.java,
                    Convention.NONE,
                    BodoPhysicalRel.CONVENTION,
                    "BodoPhysicalJoinRule",
                )
                .withRuleFactory { config -> BodoPhysicalJoinRule(config) }
    }

    override fun convert(rel: RelNode): RelNode? {
        val join = rel as Join

        if (!BodoJoinConditionUtil.isValidNode(join.condition)) {
            if (!BodoJoinConditionUtil.isTransformableToValid(join)) {
                // We cannot recover unless we can prune the whole join. Fail now.
                throw SqlUtil.newContextException(SqlParserPos.ZERO, BODO_SQL_RESOURCE.unsupportedJoinCondition())
            } else {
                // A transformation can fix this node, don't fail.
                return null
            }
        }
        val inputs =
            join.inputs.map { input ->
                convert(input, input.traitSet.replace(BodoPhysicalRel.CONVENTION))
            }
        return BodoPhysicalJoin.create(inputs[0], inputs[1], join.condition, join.joinType)
    }
}
