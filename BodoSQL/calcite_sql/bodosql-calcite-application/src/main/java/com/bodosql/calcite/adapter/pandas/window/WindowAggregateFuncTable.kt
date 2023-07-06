package com.bodosql.calcite.adapter.pandas.window

import com.bodosql.calcite.application.BodoSQLOperatorTables.CondOperatorTable
import com.bodosql.calcite.ir.Expr
import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlOperator

/**
 * Mapping of SqlOperator to window functions that can be invoked
 * through the window function.
 */
internal object WindowAggregateFuncTable {

    /**
     * Mapping of [SqlKind] values to [WindowAggregateFunc].
     *
     * This applies a 1:1 mapping between the kind and the code generation.
     */
    private val kindMapping = mapOf(
        SqlKind.ROW_NUMBER to define("row_number"),
        SqlKind.RANK to define("rank"),
        SqlKind.DENSE_RANK to define("dense_rank"),
        SqlKind.PERCENT_RANK to define("percent_rank"),
        SqlKind.CUME_DIST to define("cume_dist"),
        SqlKind.NTILE to define("ntile", ExprType.SCALAR),
    )

    /**
     * Mapping of function names to [WindowAggregateFunc].
     *
     * This is primarily used when the function type is [SqlKind.OTHER] or
     * [SqlKind.OTHER_FUNCTION] because it's not a native Calcite function
     * so we are dependent on the function name to map the function to
     * code generation.
     */
    private val nameMapping = mapOf(
        CondOperatorTable.MIN_ROW_NUMBER_FILTER to define("min_row_number_filter"),
        CondOperatorTable.CONDITIONAL_TRUE_EVENT to define("conditional_true_event", ExprType.SERIES),
        CondOperatorTable.CONDITIONAL_CHANGE_EVENT to define("conditional_change_event", ExprType.SERIES),
    ).mapKeys { it.key.name }

    /**
     * Retrieves the WindowAggregateFunc used for this operator if it is available.
     */
    fun get(op: SqlOperator): WindowAggregateFunc? =
        kindMapping[op.kind] ?: nameMapping[op.name]

    /**
     * Marks whether a particular argument should be treated as a series or a scalar.
     * Scalar types are passed in directly with their raw code text wrapped in a string
     * and series types are evaluated into a column within the dataframe and then referenced
     * by their column name.
     *
     * This determines which method in [OperandResolver] will be invoked for the particular
     * operand at that index.
     */
    private enum class ExprType {
        SERIES,
        SCALAR,
    }

    /**
     * Utility function for defining a window aggregate function.
     *
     * @param name the name of the bodosql window kernel function
     * @param operands list of expression types used to process arguments
     */
    private fun define(name: String, vararg operands: ExprType): WindowAggregateFunc =
        WindowAggregateFunc(fun (call: OverFunc, resolve: OperandResolver): Expr {
            val args = ImmutableList.builder<Expr>()
            args.add(Expr.StringLiteral(name))
            operands.zip(call.operands).forEach { (exprType, operand) ->
                val arg = when (exprType) {
                    ExprType.SERIES -> resolve.series(operand)
                    ExprType.SCALAR -> resolve.scalar(operand)
                }
                args.add(arg)
            }
            return Expr.Tuple(args.build())
        })
}
