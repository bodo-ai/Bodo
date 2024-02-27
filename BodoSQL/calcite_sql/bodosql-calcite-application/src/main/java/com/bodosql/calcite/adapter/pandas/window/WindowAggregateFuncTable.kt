package com.bodosql.calcite.adapter.pandas.window

import com.bodosql.calcite.application.operatorTables.AggOperatorTable
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
    private val kindMapping =
        mapOf(
            SqlKind.ROW_NUMBER to define("row_number"),
            SqlKind.RANK to define("rank"),
            SqlKind.DENSE_RANK to define("dense_rank"),
            SqlKind.PERCENT_RANK to define("percent_rank"),
            SqlKind.CUME_DIST to define("cume_dist"),
            SqlKind.NTILE to define("ntile", ExprType.SCALAR),
            SqlKind.COUNT to WindowAggregateFunc(::count),
            SqlKind.VAR_POP to defineBounded("var_pop", ExprType.SERIES),
            SqlKind.VAR_SAMP to defineBounded("var", ExprType.SERIES),
            SqlKind.STDDEV_POP to defineBounded("std_pop", ExprType.SERIES),
            SqlKind.STDDEV_SAMP to defineBounded("std", ExprType.SERIES),
            SqlKind.AVG to defineBounded("mean", ExprType.SERIES),
            SqlKind.ANY_VALUE to define("any_value", ExprType.SERIES),
            SqlKind.FIRST_VALUE to defineBounded("first", ExprType.SERIES),
            SqlKind.LAST_VALUE to defineBounded("last", ExprType.SERIES),
        )

    /**
     * Mapping of function names to [WindowAggregateFunc].
     *
     * This is primarily used when the function type is [SqlKind.OTHER] or
     * [SqlKind.OTHER_FUNCTION] because it's not a native Calcite function
     * so we are dependent on the function name to map the function to
     * code generation.
     */
    private val nameMapping =
        mapOf(
            AggOperatorTable.MIN_ROW_NUMBER_FILTER to define("min_row_number_filter"),
            AggOperatorTable.RATIO_TO_REPORT to define("ratio_to_report", ExprType.SERIES),
            AggOperatorTable.CONDITIONAL_TRUE_EVENT to define("conditional_true_event", ExprType.SERIES),
            AggOperatorTable.CONDITIONAL_CHANGE_EVENT to define("conditional_change_event", ExprType.SERIES),
            AggOperatorTable.COUNT_IF to defineBounded("count_if", ExprType.SERIES),
            AggOperatorTable.VARIANCE_POP to defineBounded("var_pop", ExprType.SERIES),
            AggOperatorTable.VARIANCE_SAMP to defineBounded("var", ExprType.SERIES),
        ).mapKeys { it.key.name }

    /**
     * Retrieves the WindowAggregateFunc used for this operator if it is available.
     */
    fun get(op: SqlOperator): WindowAggregateFunc? = kindMapping[op.kind] ?: nameMapping[op.name]

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
     * Populates an argument tuple with the expressions for each
     * argument of a window function, processed according to which
     * of the function's arguments are series vs scalars.
     *
     * @param args the list where the argument literals are being stored.
     * @param call the window function call storing information such as
     *             its argument nodes.
     * @param resolve the resolver of any window function arguments into codegen.
     * @param operands list of expression types used to process arguments
     */
    private fun resolveOperands(
        args: ImmutableList.Builder<Expr>,
        call: OverFunc,
        resolve: OperandResolver,
        vararg operands: ExprType,
    ) {
        operands.zip(call.operands).forEach { (exprType, operand) ->
            val arg =
                when (exprType) {
                    ExprType.SERIES -> resolve.series(operand)
                    ExprType.SCALAR -> resolve.scalar(operand)
                }
            args.add(arg)
        }
    }

    /**
     * Utility function for defining a window aggregate function that
     * does not accept window bounds.
     *
     * @param name the name of the bodosql window kernel function
     * @param operands list of expression types used to process arguments
     */
    private fun define(
        name: String,
        vararg operands: ExprType,
    ): WindowAggregateFunc =
        WindowAggregateFunc { call, resolve ->
            val args = ImmutableList.builder<Expr>()
            args.add(Expr.StringLiteral(name))
            resolveOperands(args, call, resolve, *operands)
            Expr.Tuple(args.build())
        }

    /**
     * Utility function for defining a window aggregate function that
     * does accept window bounds.
     *
     * @param name the name of the bodosql window kernel function
     * @param operands list of expression types used to process arguments
     */
    private fun defineBounded(
        name: String,
        vararg operands: ExprType,
    ): WindowAggregateFunc =
        WindowAggregateFunc { call, resolve ->
            val args = ImmutableList.builder<Expr>()
            args.add(Expr.StringLiteral(name))
            resolveOperands(args, call, resolve, *operands)
            args.add(resolve.bound(call.window.lowerBound))
            args.add(resolve.bound(call.window.upperBound))
            Expr.Tuple(args.build())
        }

    /**
     * Utility function for defining the window aggregate function COUNT
     * which has a special implementation for COUNT(*) if the aggregate
     * is provided no arguments.
     */
    private fun count(
        call: OverFunc,
        resolve: OperandResolver,
    ): Expr {
        val args = ImmutableList.builder<Expr>()
        if (call.operands.isEmpty()) {
            // If the function has zero arguments, it is a COUNT(*)
            args.add(Expr.StringLiteral("size"))
        } else {
            // Otherwise, it is a regular COUNT(X)
            args.add(Expr.StringLiteral("count"))
            resolveOperands(args, call, resolve, ExprType.SERIES)
        }
        args.add(resolve.bound(call.window.lowerBound))
        args.add(resolve.bound(call.window.upperBound))
        return Expr.Tuple(args.build())
    }
}
