package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.rel.logical.BodoLogicalJoin
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.type.SqlTypeName.BIGINT
import org.apache.calcite.sql.type.SqlTypeName.BOOLEAN
import org.apache.calcite.sql.type.SqlTypeName.CHAR
import org.apache.calcite.sql.type.SqlTypeName.DECIMAL
import org.apache.calcite.sql.type.SqlTypeName.DOUBLE
import org.apache.calcite.sql.type.SqlTypeName.FLOAT
import org.apache.calcite.sql.type.SqlTypeName.INTEGER
import org.apache.calcite.sql.type.SqlTypeName.REAL
import org.apache.calcite.sql.type.SqlTypeName.SMALLINT
import org.apache.calcite.sql.type.SqlTypeName.TINYINT
import org.apache.calcite.sql.type.SqlTypeName.VARCHAR

class PandasJoinRule private constructor(config: Config) : ConverterRule(config) {
    companion object {
        @JvmField
        val DEFAULT_CONFIG: Config = Config.INSTANCE
            .withConversion(
                BodoLogicalJoin::class.java,
                Convention.NONE,
                PandasRel.CONVENTION,
                "PandasJoinRule",
            )
            .withRuleFactory { config -> PandasJoinRule(config) }

        fun isValidNode(node: RexNode): Boolean {
            return when (node) {
                is RexLiteral -> when (node.type.sqlTypeName) {
                    TINYINT, SMALLINT, INTEGER, BIGINT,
                    FLOAT, REAL, DOUBLE, DECIMAL, CHAR,
                    VARCHAR, BOOLEAN,
                    -> true
                    else -> false
                }

                is RexInputRef -> true
                is RexCall -> when (node.kind) {
                    SqlKind.EQUALS, SqlKind.NOT_EQUALS, SqlKind.GREATER_THAN,
                    SqlKind.GREATER_THAN_OR_EQUAL, SqlKind.LESS_THAN,
                    SqlKind.LESS_THAN_OR_EQUAL, SqlKind.AND, SqlKind.OR, SqlKind.PLUS, SqlKind.MINUS,
                    SqlKind.TIMES, SqlKind.DIVIDE, SqlKind.NOT, SqlKind.IS_NOT_TRUE,
                    ->
                        node.operands.all {
                            isValidNode(it)
                        }

                    else -> when (node.op.toString()) {
                        "POW" -> {
                            node.operands.all {
                                isValidNode(it)
                            }
                        }

                        else -> false
                    }
                }
                else -> false
            }
        }
    }

    override fun convert(rel: RelNode): RelNode? {
        val join = rel as Join

        if (!isValidNode(join.condition)) {
            return null
        }
        val inputs = join.inputs.map { input ->
            convert(input, input.traitSet.replace(PandasRel.CONVENTION))
        }
        return PandasJoin.create(inputs[0], inputs[1], join.condition, join.joinType)
    }
}
