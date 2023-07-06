package com.bodosql.calcite.adapter.pandas.window

import com.bodosql.calcite.ir.Expr
import org.apache.calcite.rex.RexNode

/**
 * Interface for resolving operands for aggregate functions.
 */
internal interface OperandResolver {
    /**
     * Resolves a series operand to a string that references a column
     * name in the original dataframe.
     *
     * This wraps the name of the column in an [Expr.StringLiteral]
     * to be referenced directly.
     */
    fun series(node: RexNode): Expr

    /**
     * Resolves a scalar operand to an expression that will be used
     * for code generation. Generally, these expressions are wrapped
     * in strings for later compilation but they are returned as
     * expressions.
     */
    fun scalar(node: RexNode): Expr
}
