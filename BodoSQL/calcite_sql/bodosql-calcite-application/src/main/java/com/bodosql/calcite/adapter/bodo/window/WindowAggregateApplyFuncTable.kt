package com.bodosql.calcite.adapter.bodo.window

import com.bodosql.calcite.application.operatorTables.AggOperatorTable
import com.bodosql.calcite.application.utils.BodoArrayHelpers
import com.bodosql.calcite.ir.Doc
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Frame
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.ir.bodoSQLKernel
import com.google.common.collect.ImmutableList
import org.apache.calcite.rex.RexOver
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.type.SqlTypeName

internal object WindowAggregateApplyFuncTable {
    /**
     * Mapping of [SqlKind] values to [WindowAggregateApplyFunc].
     *
     * This applies a 1:1 mapping between the kind and the code generation.
     */
    private val kindMapping =
        mapOf(
            SqlKind.SUM to ::sum,
            SqlKind.SUM0 to ::sum,
            // count is an actual function in map so need to fully qualify this.
            SqlKind.COUNT to WindowAggregateApplyFuncTable::count,
            SqlKind.AVG to ::avg,
            SqlKind.MIN to ::min,
            SqlKind.MAX to ::max,
            SqlKind.MEDIAN to ::median,
            SqlKind.MODE to ::mode,
            SqlKind.VAR_SAMP to ::varianceSamp,
            SqlKind.VAR_POP to ::variancePop,
            SqlKind.COVAR_SAMP to ::covarSample,
            SqlKind.COVAR_POP to ::covarPop,
            SqlKind.STDDEV_SAMP to ::stddev,
            SqlKind.STDDEV_POP to ::stddevPop,
            SqlKind.ROW_NUMBER to ::rowNumber,
            SqlKind.RANK to ::rank,
            SqlKind.DENSE_RANK to ::rank,
            SqlKind.CUME_DIST to ::rank,
            SqlKind.PERCENT_RANK to ::percentRank,
            SqlKind.NTILE to ::ntile,
            SqlKind.LEAD to ::lead,
            SqlKind.LAG to ::lag,
            SqlKind.FIRST_VALUE to ::nthValue,
            SqlKind.LAST_VALUE to ::nthValue,
            SqlKind.ANY_VALUE to ::nthValue,
            SqlKind.NTH_VALUE to ::nthValue,
        )

    /**
     * Mapping of function names to [WindowAggregateApplyFunc].
     *
     * This is primarily used when the function type is [SqlKind.OTHER] or
     * [SqlKind.OTHER_FUNCTION] because it's not a native Calcite function
     * so we are dependent on the function name to map the function to
     * code generation.
     */
    private val nameMapping =
        mapOf(
            AggOperatorTable.COUNT_IF to ::countIf,
            AggOperatorTable.RATIO_TO_REPORT to ::ratioToReport,
            AggOperatorTable.CORR to ::corr,
            AggOperatorTable.CONDITIONAL_CHANGE_EVENT to ::conditionalChangeEvent,
            AggOperatorTable.SKEW to ::skew,
            AggOperatorTable.KURTOSIS to ::kurtosis,
            AggOperatorTable.VARIANCE_SAMP to ::varianceSamp,
            AggOperatorTable.VARIANCE_POP to ::variancePop,
            AggOperatorTable.BOOLOR_AGG to ::boolorAgg,
            AggOperatorTable.BOOLAND_AGG to ::boolandAgg,
            AggOperatorTable.BOOLXOR_AGG to ::boolxorAgg,
            AggOperatorTable.CONDITIONAL_TRUE_EVENT to ::conditionalTrueEvent,
            AggOperatorTable.APPROX_PERCENTILE to ::approxPercentile,
            AggOperatorTable.BITOR_AGG to ::bitorAgg,
            AggOperatorTable.BITAND_AGG to ::bitandAgg,
            AggOperatorTable.BITXOR_AGG to ::bitxorAgg,
            AggOperatorTable.OBJECT_AGG to ::objectAgg,
        ).mapKeys { it.key.name }

    fun get(op: SqlOperator): WindowAggregateApplyFunc? =
        (kindMapping[op.kind] ?: nameMapping[op.name])?.let {
            WindowAggregateApplyFunc(it)
        }

    /**
     * Returns if this [RexOver] can produce an unordered result.
     *
     * If true, this allows the code generation to avoid resorting
     * the resulting array.
     */
    fun isUnorderedResult(call: RexOver): Boolean =
        when (call.kind) {
            // If doing FIRST_VALUE/ANY_VALUE on a prefix window, or LAST_VALUE
            // on a suffix window, the optimized version can be used which
            // does NOT require the sorting to be reverted at the end.
            // TODO(jsternberg): Copied this comment over from its original location,
            // but the logic doesn't make sense to me. It seems reasonable that
            // FIRST_VALUE with an unbounded lower bound and LAST_VALUE with an unbounded
            // upper bound would not require ordering since the result would be a constant
            // value, but it's unclear to me why the original logic included making sure
            // the other bound was the current row. It seems like they could be anything?
            // https://bodo.atlassian.net/browse/BSE-699
            SqlKind.FIRST_VALUE, SqlKind.ANY_VALUE ->
                call.window.lowerBound.isUnbounded && call.window.upperBound.isCurrentRow
            SqlKind.LAST_VALUE ->
                call.window.upperBound.isUnbounded && call.window.lowerBound.isCurrentRow
            // Most window functions are order sensitive.
            // TODO(jsternberg): Is this true? If this method is useful
            // for things like first_value and last_value when they produce
            // only one value, why doesn't it work for things where both sides
            // are unbounded? The original comment said SUM is an example of an ordered
            // aggregation, but the order of aggregation doesn't matter if SUM is used
            // on an unbounded window.
            else -> false
        }

    private fun sum(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_sum", operands)

    private fun count(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr {
        val args = ImmutableList.builder<Expr>()
        val kernelFuncName =
            if (operands.isEmpty()) {
                args.add(ctx.len)
                "windowed_count_star"
            } else {
                args.addAll(operands)
                "windowed_count"
            }
        args.addAll(evalBounds(ctx))
        return bodoSQLKernel(kernelFuncName, args.build())
    }

    private fun countIf(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_count_if", operands)

    private fun avg(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_avg", operands)

    private fun min(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_min", operands)

    private fun max(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_max", operands)

    private fun median(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_median", operands)

    private fun mode(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_mode", operands)

    private fun ratioToReport(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_ratio_to_report", operands)

    private fun covarSample(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_covar_samp", operands)

    private fun covarPop(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_covar_pop", operands)

    private fun corr(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_corr", operands)

    private fun conditionalChangeEvent(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = bodoSQLKernel("change_event", operands)

    private fun stddev(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_stddev_samp", operands)

    private fun stddevPop(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_stddev_pop", operands)

    private fun objectAgg(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = bodoSQLKernel("windowed_object_agg", operands)

    private fun rowNumber(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr =
        Expr.Call(
            "np.arange",
            Expr.One,
            Expr.Binary("+", ctx.len, Expr.One),
        )

    private fun rank(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr {
        val methodName =
            when (call.kind) {
                SqlKind.DENSE_RANK -> "dense"
                SqlKind.CUME_DIST -> "max"
                else -> "min"
            }

        // Add sorting column to the tuple of arguments.
        val sortedCols =
            Expr.Tuple(
                ctx.orderKeys.map {
                    Expr.Call("bodo.hiframes.pd_series_ext.get_series_data", it)
                },
            )

        val pctExpr =
            if (call.kind == SqlKind.CUME_DIST) {
                Expr.True
            } else {
                Expr.False
            }

        return Expr.Call(
            "bodosql.kernels.rank_sql",
            args = listOf(sortedCols),
            namedArgs =
                listOf(
                    "method" to Expr.StringLiteral(methodName),
                    "pct" to pctExpr,
                ),
        )
    }

    private fun approxPercentile(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = bodoSQLKernel("windowed_approx_percentile", operands)

    private fun percentRank(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr {
        // TODO(jsternberg): https://bodo.atlassian.net/browse/BSE-690
        val rankExpr = rank(ctx, call, operands)

        // Augment the calculation as follows:
        //
        // _tmp_arr = [rank expr] - 1
        // if argumentDfLen == 1:
        //   _tmp_arr[:] = 0.0
        // else:
        //   _tmp_arr /= (argumentDfLen - 1)
        val tempArr = ctx.builder.symbolTable.genGenericTempVar()
        ctx.builder.add(Op.Assign(tempArr, Expr.Binary("-", rankExpr, Expr.IntegerLiteral(1))))
        ctx.builder.add(
            Op.If(
                Expr.Binary("==", ctx.len, Expr.IntegerLiteral(1)),
                StatementList(
                    // TODO(jsternberg): Need an op code for this.
                    Op.Code("${tempArr.emit()}[:] = 0.0"),
                ),
                StatementList(
                    Op.Code("${tempArr.emit()} /= (${ctx.len.emit()} - 1)"),
                ),
            ),
        )
        return tempArr
    }

    private fun ntile(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr =
        Expr.Call(
            "bodosql.libs.ntile_helper.ntile_helper",
            ctx.len,
            operands[0],
        )

    private fun lead(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr {
        val newOperands =
            when (operands.size) {
                1 -> listOf(operands[0], Expr.Unary("-", Expr.Call("np.int32", Expr.One)))
                else -> {
                    // Replace the second argument with one that uses
                    // the unary negative.
                    buildList {
                        add(operands[0])
                        add(Expr.Unary("-", operands[1]))
                        if (operands.size > 2) {
                            addAll(operands.slice(2 until operands.size))
                        }
                    }
                }
            }
        // Use the lag implementation now that we've modified the operands.
        return lag(ctx, call, newOperands)
    }

    private fun lag(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr {
        val column = operands[0]
        val shift = operands.getOrElse(1) { Expr.Call("np.int32", Expr.One) }
        val fill = operands.getOrElse(2) { Expr.None }
        val ignoreNulls = Expr.BooleanLiteral(call.ignoreNulls())
        return Expr.Call(
            "bodosql.kernels.lead_lag.lead_lag_seq",
            column,
            shift,
            fill,
            ignoreNulls,
        )
    }

    fun skew(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_skew", operands)

    fun kurtosis(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_kurtosis", operands)

    fun varianceSamp(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_var_samp", operands)

    fun variancePop(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_var_pop", operands)

    fun bitorAgg(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_bitor_agg", operands)

    fun bitandAgg(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_bitand_agg", operands)

    fun bitxorAgg(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_bitxor_agg", operands)

    fun boolorAgg(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_boolor", operands)

    fun boolandAgg(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_booland", operands)

    fun boolxorAgg(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr = boundedNativeKernel(ctx, "windowed_boolxor", operands)

    private fun conditionalTrueEvent(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr =
        Expr.Call(
            "bodo.hiframes.pd_series_ext.get_series_data",
            Expr.Call(
                Expr.Attribute(
                    Expr.Call(
                        Expr.Attribute(
                            operands[0],
                            "astype",
                        ),
                        Expr.StringLiteral("uint32"),
                    ),
                    "cumsum",
                ),
            ),
        )

    // TODO(jsternberg): This entire function should be refactored into a kernel.
    // The codegen here is far more complicated than is worth keeping in the Java code.
    private fun nthValue(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr {
        // Check if we can use the unordered version of this function.
        if (isUnorderedResult(call)) {
            return nthValueUnordered(ctx, call, operands)
        }

        val targetValue =
            when (call.kind) {
                SqlKind.FIRST_VALUE, SqlKind.ANY_VALUE -> Expr.Raw("cur_lower_bound")
                SqlKind.LAST_VALUE -> Expr.Binary("-", Expr.Raw("cur_upper_bound"), Expr.IntegerLiteral(1))
                SqlKind.NTH_VALUE -> {
                    val n = Variable("n")
                    ctx.builder.add(Op.Assign(n, Expr.Call("max", Expr.One, operands[1])))
                    Expr.Raw("cur_lower_bound + ${n.emit()} - 1")
                }
                else -> throw AssertionError("invalid nthValue kind")
            }

        val outputArray = ctx.builder.symbolTable.genArrayVar()
        ctx.builder.add(
            Op.Assign(
                outputArray,
                Expr.Raw(BodoArrayHelpers.sqlTypeToNullableBodoArray(ctx.len.emit(), call.type, ctx.defaultTZInfo.zoneExpr)),
            ),
        )

        val inputArray = ctx.builder.symbolTable.genArrayVar()
        ctx.builder.add(Op.Assign(inputArray, Expr.Call("bodo.hiframes.pd_series_ext.get_series_data", operands[0])))

        val loop =
            Op.For("i", Expr.Call("range", ctx.len)) { index, body ->
                val lowerBound =
                    if (ctx.bounds.lower != null) {
                        Expr.Call(
                            "min",
                            ctx.len,
                            Expr.Call(
                                "max",
                                Expr.Zero,
                                Expr.Binary("+", index, ctx.bounds.lower),
                            ),
                        )
                    } else {
                        Expr.Zero
                    }
                val lowerBoundVar = Variable("cur_lower_bound")
                body.add(Op.Assign(lowerBoundVar, lowerBound))

                val upperBound =
                    if (ctx.bounds.upper != null) {
                        Expr.Call(
                            "min",
                            ctx.len,
                            Expr.Call(
                                "max",
                                Expr.Zero,
                                Expr.Binary(
                                    "+",
                                    Expr.Binary("+", index, ctx.bounds.upper),
                                    Expr.One,
                                ),
                            ),
                        )
                    } else {
                        ctx.len
                    }
                val upperBoundVar = Variable("cur_upper_bound")
                body.add(Op.Assign(upperBoundVar, upperBound))

                val currentIndex = Variable("cur_idx")
                body.add(Op.Assign(currentIndex, targetValue))

                // Collect the list of conditions for determining if
                // the index is valid. When used with NTH_VALUE,
                // there's an additional condition to ensure the selected
                // value is within the current window.
                val conditions = mutableListOf<Expr>()
                conditions.add(Expr.Binary(">=", lowerBoundVar, upperBoundVar))
                if (call.kind == SqlKind.NTH_VALUE) {
                    conditions.add(Expr.Binary("<", Expr.Binary("-", upperBoundVar, lowerBoundVar), Variable("n")))
                }
                conditions.add(Expr.Call("bodo.libs.array_kernels.isna", inputArray, currentIndex))

                // If statement to determine if this value will be null
                // or if we will select the value at the given index.
                // We join the conditions above with logical or operations.
                val nullCondition = conditions.reduce { a, b -> Expr.Binary("or", a, b) }
                body.add(
                    Op.If(
                        nullCondition,
                        StatementList(
                            Op.Stmt(Expr.Call("bodo.libs.array_kernels.setna", outputArray, index)),
                        ),
                        StatementList(
                            // TODO(jsternberg): Need an op code for this.
                            Op.Code("${outputArray.emit()}[i] = ${inputArray.emit()}[${currentIndex.emit()}]"),
                        ),
                    ),
                )
            }
        ctx.builder.add(loop)
        return outputArray
    }

    private fun nthValueUnordered(
        ctx: WindowAggregateContext,
        call: RexOver,
        operands: List<Expr>,
    ): Expr {
        val targetIndex =
            if (call.kind == SqlKind.LAST_VALUE) {
                Expr.Binary("-", ctx.len, Expr.IntegerLiteral(1))
            } else {
                Expr.Zero
            }

        // Generate the logic that extracts the first/last value.
        // Checks if it is null and then executes one of the two branches.
        val inputArray = ctx.builder.symbolTable.genArrayVar()
        ctx.builder.add(Op.Assign(inputArray, Expr.Call("bodo.hiframes.pd_series_ext.get_series_data", operands[0])))

        // Generate the output series variable that we will assign to.
        val outputArray = ctx.builder.symbolTable.genArrayVar()

        // Branch to use if the value is null.
        val typeName = call.type.sqlTypeName
        val nullBranch =
            StatementList(
                Op.Assign(
                    outputArray,
                    when {
                        SqlTypeName.CHAR_TYPES.contains(typeName) ->
                            Expr.Call(
                                "bodo.libs.str_arr_ext.gen_na_str_array_lens",
                                ctx.len,
                                Expr.Zero,
                                Expr.Call("np.empty", Expr.One, Expr.Raw("np.int64")),
                            )

                        SqlTypeName.BINARY_TYPES.contains(typeName) ->
                            Expr.Call(
                                "bodo.libs.str_arr_ext.pre_alloc_binary_array",
                                ctx.len,
                                Expr.Zero,
                            )

                        else ->
                            Expr.Raw(
                                BodoArrayHelpers.sqlTypeToNullableBodoArray(ctx.len.emit(), call.type, ctx.defaultTZInfo.zoneExpr),
                            )
                    },
                ),
                Op.For("j", Expr.Call("range", Expr.Len(outputArray))) { index, body ->
                    body.add(
                        Op.Stmt(Expr.Call("bodo.libs.array_kernels.setna", outputArray, index)),
                    )
                },
            )

        val evalBranch =
            StatementList(
                when {
                    SqlTypeName.CHAR_TYPES.contains(typeName) ->
                        listOf(
                            Op.Assign(
                                outputArray,
                                Expr.Call(
                                    "bodo.libs.str_arr_ext.pre_alloc_string_array",
                                    ctx.len,
                                    Expr.Binary(
                                        "*",
                                        Expr.Call(
                                            "bodo.libs.str_arr_ext.get_str_arr_item_length",
                                            inputArray,
                                            targetIndex,
                                        ),
                                        ctx.len,
                                    ),
                                ),
                            ),
                            Op.For("j", Expr.Call("range", Expr.Call("len", outputArray))) { index, body ->
                                body.add(
                                    Op.Stmt(
                                        Expr.Call(
                                            "bodo.libs.str_arr_ext.get_str_arr_item_copy",
                                            outputArray,
                                            index,
                                            inputArray,
                                            targetIndex,
                                        ),
                                    ),
                                )
                            },
                        )
                    SqlTypeName.BINARY_TYPES.contains(typeName) -> {
                        val tempVar = ctx.builder.symbolTable.genGenericTempVar()
                        listOf(
                            Op.Assign(tempVar, Expr.Index(inputArray, targetIndex)),
                            Op.Assign(
                                outputArray,
                                Expr.Call(
                                    "bodo.libs.str_arr_ext.pre_alloc_binary_array",
                                    ctx.len,
                                    Expr.Binary(
                                        "*",
                                        Expr.Call("len", tempVar),
                                        ctx.len,
                                    ),
                                ),
                            ),
                            Op.For("j", Expr.Call("range", Expr.Call("len", outputArray))) { index, body ->
                                body.add(Op.Code("${outputArray.emit()}[${index.emit()}] = ${tempVar.emit()}"))
                            },
                        )
                    }
                    else -> {
                        val tempVar = ctx.builder.symbolTable.genGenericTempVar()
                        listOf(
                            Op.Assign(tempVar, Expr.Index(inputArray, targetIndex)),
                            Op.Assign(
                                outputArray,
                                Expr.Raw(
                                    BodoArrayHelpers.sqlTypeToNullableBodoArray(ctx.len.emit(), call.type, ctx.defaultTZInfo.zoneExpr),
                                ),
                            ),
                            Op.For("j", Expr.Call("range", Expr.Call("len", outputArray))) { index, body ->
                                body.add(Op.Code("${outputArray.emit()}[${index.emit()}] = ${tempVar.emit()}"))
                            },
                        )
                    }
                },
            )

        ctx.builder.add(
            Op.If(
                Expr.Call("bodo.libs.array_kernels.isna", inputArray, targetIndex),
                nullBranch,
                evalBranch,
            ),
        )
        return outputArray
    }

    /**
     * Used to invoke a bodosql windowed aggregate kernel that supports bounded
     * options with the given name.
     *
     * This will implicitly add the lower and upper bounds to the call as the final
     * arguments.
     */
    private fun boundedNativeKernel(
        ctx: WindowAggregateContext,
        kernelName: String,
        operands: List<Expr>,
    ): Expr = bodoSQLKernel(kernelName, operands + evalBounds(ctx))

    /**
     * Uses the [WindowAggregateContext] to generate the lower and upper bound
     * arguments for a bounded window function call. See [boundedNativeKernel]
     * as an example.
     *
     * If the lower or upper boundary are unbounded, replaces them
     * with an appropriate expression.
     */
    private fun evalBounds(ctx: WindowAggregateContext): List<Expr> =
        listOf(
            ctx.bounds.lower ?: Expr.Unary("-", ctx.len),
            ctx.bounds.upper ?: ctx.len,
        )

    // TODO(jsternberg): Refactor frames inside of the IR.
    // This will be easier to do when more of the logic for frames and the builder
    // are moved out of BodoCodeGenVisitor, but that's hard for now so just want
    // to be able to utilize Op.If without rewriting everything around it.
    private class StatementList(
        private val body: List<Op>,
    ) : Frame {
        constructor(vararg body: Op) : this(body.toList())

        override fun emit(doc: Doc) = body.forEach { it.emit(doc) }

        override fun add(op: Op) = throw NotImplementedError()

        override fun addAll(ops: List<Op>) = throw NotImplementedError()

        override fun append(code: String) = throw NotImplementedError()

        override fun prependAll(ops: List<Op>) = throw NotImplementedError()

        override fun addBeforeReturn(op: Op) = throw NotImplementedError()
    }
}
