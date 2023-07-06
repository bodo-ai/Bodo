package com.bodosql.calcite.adapter.pandas.window

import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.adapter.pandas.RexToPandasTranslator
import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.ir.*
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.rel.RelFieldCollation
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.*
import org.apache.calcite.sql.SqlKind

internal class Group(
    val cluster: RelOptCluster,
    val input: Dataframe,
    aggregates: List<RexOver>,
    val window: RexWindow,
) {
    private val _aggregates = rewriteOperands(aggregates)
    private val aggregates: List<OverFunc> get() = _aggregates.first
    private val localRefs: List<RexNode> get() = _aggregates.second

    /**
     * Constructs the aggregate function that will be passed to apply.
     */
    fun emit(ctx: PandasRel.BuildContext): List<Variable> {
        val rexTranslator = ctx.rexTranslator(input)

        // Retrieve the partition keys and the ordering keys.
        val partitionKeys = partitionBy(rexTranslator)
        val orderKeys = orderBy(rexTranslator)
        val fields = aggregateInputs(rexTranslator)

        // Generate the appropriate function invocation for this group of aggregates.
        // This will always generate a dataframe with aggregate outputs that correspond
        // to the list of aggregates in this group.
        //
        // It may also generate additional code that supports the aggregation
        // depending on if this group chooses to use apply, window, or a direct function call.
        val builder = ctx.builder()
        val (windowExpr, windowFnOutputs) = emit(ctx, partitionKeys, orderKeys, fields)
        val windowDf = builder.symbolTable.genWindowedAggDf()
        builder.add(Op.Assign(windowDf, windowExpr))

        // Extract the series data from the returned dataframe into corresponding
        // variables that match 1:1 with the aggregates in this group.
        return windowFnOutputs.map { aggOutputName ->
            val expr = Expr.Call(
                "bodo.hiframes.pd_series_ext.get_series_data",
                Expr.Index(
                    windowDf,
                    Expr.StringLiteral(aggOutputName),
                )
            )
            val seriesVar = builder.symbolTable.genSeriesVar()
            builder.add(Op.Assign(seriesVar, expr))
            seriesVar
        }
    }

    /**
     * Performs the code generation for this group of common partition keys and order keys
     * depending on the contents of the aggregate functions.
     *
     * There are presently three scenarios.
     * 1. ROW_NUMBER is used without a partition key.
     * 2. All aggregates support the window function (from [WindowAggregateFuncTable]) and
     *    code generation is used with window.
     * 3. A function is generated for use with the apply function (from [WindowAggregateApplyFuncTable]).
     *
     * One of these three will be used. Versions 2 and 3 require a partition key because
     * the resulting code only works if there is at least one group.
     */
    private fun emit(ctx: PandasRel.BuildContext, partitionKeys: List<PartitionKey>, orderKeys: List<OrderKey>, fields: List<Field>): Pair<Expr, List<String>> {
        // TODO(jsternberg): Refactor this into something more generic.
        // Row number is supported without a partition which means it doesn't have to go
        // through the groupby pipeline. More aggregations may be included in the future.
        if (aggregates.size == 1 && partitionKeys.isEmpty() && aggregates.first().kind == SqlKind.ROW_NUMBER) {
            return emitRowNumber(partitionKeys, orderKeys, fields)
        }

        // Following the above, we require partition key.
        if (partitionKeys.isEmpty()) {
            val aggregateNames = aggregates.map { it.op.name }
            throw BodoSQLCodegenException("Windowed aggregates $aggregateNames require a partition key")
        }

        // Determine if we can use the groupby.window function with specialized kernels.
        // We iterate over the list of aggregates in this group and see if all of them
        // have a definition that can be used with groupby.window.
        // If one is missing, we switch to using groupby.apply.
        val windowFuncs = aggregates.map {
            val func = WindowAggregateFuncTable.get(it.op) ?:
                // Break out of the loop and move to the apply route.
                // If we can't group them all together, we don't do it at all.
                return emitWindowApply(ctx, partitionKeys, orderKeys, fields)
            Pair(it, func)
        }

        // We have now confirmed we can use the window function.
        // Create the operand resolver to be used during code generation.
        val resolver = OperandResolverImpl(ctx, input, fields)

        // Use the resolver to evaluate each of the window aggregate functions.
        val windowFuncArgs = windowFuncs.map { (agg, w) -> w.emit(agg, resolver) }

        // Generate the windowed dataframe and then process the window
        // function arguments.
        val input = emitGeneratedWindowDataframe(partitionKeys, orderKeys, fields, resolver.extraFields)
        val windowCall = Expr.Call(
            Expr.Attribute(input, "window"),
            Expr.Tuple(windowFuncArgs),
            orderKeys.fieldList(Expr::Tuple),
            orderKeys.ascendingList(Expr::Tuple),
            orderKeys.nullPositionList(Expr::Tuple),
        )

        // Output fields are hard-coded by the window function.
        val outputFields = windowFuncs.indices.map {
                i -> "AGG_OUTPUT_$i"
        }
        return Pair(windowCall, outputFields)
    }

    /**
     * Emits the specialized code for row_number when it is used by itself in a group.
     */
    private fun emitRowNumber(partitionKeys: List<PartitionKey>, orderKeys: List<OrderKey>, fields: List<Field>): Pair<Expr, List<String>> {
        // TODO(jsternberg): Unsafe index access, but this has been checked before invoking this function.
        val input = emitGeneratedWindowDataframe(partitionKeys, orderKeys, fields)
        return Pair(
            Expr.Call(
                "bodo.libs.bodosql_array_kernels.row_number",
                input,
                orderKeys.fieldList(),
                orderKeys.ascendingList(),
                orderKeys.nullPositionList(),
            ),
            listOf("ROW_NUMBER"),
        )
    }

    /**
     * Analyzes the operands and rewrites arguments that contain an InputRef
     * so the InputRef is evaluated outside the apply/window function.
     *
     * Scalar arguments are untouched.
     *
     * The first returned argument is for the list of rewritten [RexOver]
     * nodes. These will have their operand references rewritten to refer to [RexLocalRef]
     * when there is a series argument and the original [RexNode] for a scalar argument.
     * These nodes are valid from within the groupby call.
     *
     * The second returned argument are the [RexNode] values to construct the [RexLocalRef].
     * These should be evaluated outside the groupby and refer to the index of the [RexNode]
     * within the generated dataframe.
     */
    private fun rewriteOperands(aggregates: List<RexOver>): Pair<List<OverFunc>, List<RexNode>> {
        // For each RexOver, we need to determine whether the operand
        // references a column or a scalar argument. Column references will
        // have an InputRef somewhere while scalar arguments will not.
        // When an operand references an InputRef, we need to evaluate the whole
        // operand outside of the apply to reduce compilation time.
        val extractedNodes = mutableListOf<RexNode>()
        val rewrittenAggregates = aggregates.map { agg ->
            // Process the operands to determine if it is a series or scalar
            // argument.
            val operands = agg.operands.map { op ->
                if (RexUtil.containsInputRef(op)) {
                    // Store this node in the list of extracted
                    // nodes and replace the operand with a RexLocalRef.
                    extractedNodes.add(op)
                    RexLocalRef(extractedNodes.lastIndex, op.type)
                } else {
                    // No modification needed so return as normal.
                    op
                }
            }
            OverFunc(agg, operands)
        }
        return Pair(rewrittenAggregates, extractedNodes.toList())
    }

    /**
     * Generate the window code for use with groupby.apply.
     *
     * This generates an apply function that will take a generated dataframe
     * as input. The generated dataframe will include columns that are referenced
     * by the underlying aggregate functions.
     *
     * This dataframe will be grouped by the partition key and then will invoke
     * apply on that grouping using the generated window function.
     */
    private fun emitWindowApply(ctx: PandasRel.BuildContext, partitionKeys: List<PartitionKey>, orderKeys: List<OrderKey>, fields: List<Field>): Pair<Expr, List<String>> {
        val (windowFnName, windowFnOutputs) = emitWindowApplyFunc(ctx, orderKeys, fields)

        // Keep track of the original position of each column.
        val extraFields = listOf(Pair(
            "ORIG_POSITION_COL",
            Expr.Call("np.arange", Expr.Len(input))
        ))
        val input = emitGeneratedWindowDataframe(partitionKeys, orderKeys, fields, extraFields)

        // Construct the apply call.
        val applyCall = Expr.Call(
            Expr.Attribute(input, "apply"),
            windowFnName,
        )
        return Pair(applyCall, windowFnOutputs)
    }

    private data class WindowApplyFunc(val name: Variable, val outputFields: List<String>)

    /**
     * Generates the window function to be invoked with apply for [emitWindowApply].
     *
     * This method only emits the code related to the function and does not emit
     * any of the code that uses the function. It returns metadata about
     * the generated function which is its name and the names of the generated
     * outputs from the function.
     */
    private fun emitWindowApplyFunc(ctx: PandasRel.BuildContext, orderKeys: List<OrderKey>, fields: List<Field>): WindowApplyFunc {
        // Use the order keys and the fields to generate a row type which we'll use
        // for the remainder of the function.
        // emit header
        // emit sorting? maybe only if order keys is not empty
        val builder = Module.Builder()
        val argumentDf = Dataframe("argument_df", getTypeFromFields(fields))
        val header = emitWindowApplyFuncHeader(builder, argumentDf, orderKeys, fields)

        // perform rex over operation on each
        // Construct local refs so we can access the data.
        val localRefs = fields.map {
            Expr.Index(header.input, Expr.StringLiteral(it.name))
        }
        val rexTranslator = ctx.rexTranslator(header.input, localRefs)
        val windowContext = WindowAggregateContext(
            builder = builder,
            len = header.len,
            orderKeys = orderKeys.map {
                Expr.Index(header.input, Expr.StringLiteral(it.field))
            },
            // Fake default bounds. We retrieve the bounds per-aggregate
            // and fill it in, but we'd also like to keep this as non-null.
            bounds = Bounds(lower = null, upper = null),
        )
        val arrs = emitWindowApplyFuncAggregates(windowContext, aggregates, rexTranslator)
        val outputFields = emitWindowApplyFuncFooter(
            builder,
            revertSortIfNeeded(
                builder,
                arrs,
                header,
                orderKeys,
            ),
            header,
        )

        val name = ctx.builder().symbolTable.genWindowedAggFnName()
        val func = builder.buildFunction(name.name, listOf(argumentDf))
        ctx.builder().add(func)
        return WindowApplyFunc(name, outputFields)
    }

    /**
     * Emits the aggregation code within the window apply function. For each
     * aggregate, a new series is generated. This method returns the
     * list of generated series variables.
     */
    private fun emitWindowApplyFuncAggregates(ctxTemplate: WindowAggregateContext, aggregates: List<OverFunc>, rexTranslator: RexToPandasTranslator): List<Variable> {
        return aggregates.mapIndexed { i, agg ->
            val out = Variable("arr$i")
            val aggFunc = WindowAggregateApplyFuncTable.get(agg.op)
                ?: throw BodoSQLCodegenException("Unrecognized window function: ${agg.op.name}")

            val bounds = calculateWindowBounds(agg.window, rexTranslator)
            val ctx = ctxTemplate.copy(bounds = bounds)
            val operands = evaluateOperands(agg.operands, rexTranslator)
            ctx.builder.add(Op.Assign(out, aggFunc.emit(ctx, agg.over, operands)))
            out
        }
    }

    /**
     * Uses the [RexToPandasTranslator] to calculate the window bounds if they are present.
     */
    private fun calculateWindowBounds(window: RexWindow, rexTranslator: RexToPandasTranslator): Bounds =
        Bounds(
            lower = calculateWindowBound(window.lowerBound, rexTranslator),
            upper = calculateWindowBound(window.upperBound, rexTranslator),
        )

    /**
     * Calculates a single window bound depending on the type of window.
     *
     * If a window is unbounded, null is returned. Otherwise, code is generated to evaluate
     * the window bounding.
     */
    private fun calculateWindowBound(windowBound: RexWindowBound, rexTranslator: RexToPandasTranslator): Expr? =
        when {
            windowBound.isUnbounded -> null
            windowBound.isPreceding -> Expr.Unary("-", windowBound.offset!!.accept(rexTranslator))
            windowBound.isFollowing -> windowBound.offset!!.accept(rexTranslator)
            windowBound.isCurrentRow -> Expr.Call("np.int64", Expr.IntegerLiteral(0))
            else -> throw AssertionError("invalid window bound")
        }

    private fun getTypeFromFields(fields: List<Field>): RelDataType =
        cluster.typeFactory.createStructType(
            fields.map { it.type },
            fields.map { it.name },
        )

    private fun evaluateOperands(operands: List<RexNode>, rexTranslator: RexToPandasTranslator): List<Expr> {
        return operands.map {
            it.accept(rexTranslator)
        }
    }

    /**
     * Emit the initialization code for the generated window apply function.
     */
    private fun emitWindowApplyFuncHeader(builder: Module.Builder, argumentDf: Dataframe, orderKeys: List<OrderKey>, fields: List<Field>): WindowApplyFuncHeader {
        // Name each of the columns that come in from the function argument.
        // Use the same variable from the input.
        val locNames = fields.map { Expr.StringLiteral(it.name) } +
                orderKeys.map { Expr.StringLiteral(it.field) } +
                Expr.StringLiteral("ORIG_POSITION_COL")
        // TODO(jsternberg): I think this doesn't translate the logic correctly.
        // Come back to this and look at [pruneColumns] to get the logic right.
        builder.add(
            Op.Assign(
                argumentDf,
                Expr.Index(
                    Expr.Attribute(argumentDf, "loc"),
                    Expr.Slice(),
                    Expr.List(locNames),
                )
            )
        )

        // Retrieve the original index for later use.
        val index = Variable("argument_df_orig_index")
        builder.add(Op.Assign(index, Expr.Attribute(argumentDf, "index")))

        // Retrieve the length for later use.
        val len = Variable("argument_df_len")
        builder.add(Op.Assign(len, Expr.Len(argumentDf)))

        val output = sortDataframeIfNeeded(builder, argumentDf, orderKeys)
        return WindowApplyFuncHeader(output, index, len, "ORIG_POSITION_COL")
    }

    /**
     * Emit sorting code for the input dataframe based on the order keys if the order
     * keys are not empty.
     *
     * This will do nothing if the order keys are empty.
     */
    private fun sortDataframeIfNeeded(builder: Module.Builder, input: Dataframe, orderKeys: List<OrderKey>): Dataframe {
        if (orderKeys.isEmpty()) {
            // No sort is needed.
            return input
        }

        // Determine the name, the order
        val sortedDataframe = Expr.Call(
            Expr.Attribute(
                input,
                "sort_values",
            ),
            namedArgs = listOf(
                "by" to orderKeys.fieldList(),
                "ascending" to orderKeys.ascendingList(),
                "na_position" to orderKeys.nullPositionList(),
            ),
        )

        val target = Dataframe("sorted_df", input.rowType)
        builder.add(Op.Assign(target, sortedDataframe))
        return target
    }

    /**
     * Emits the return code for the window apply function. This code is generated
     * after the aggregations have been performed and is used to store the results
     * in a new dataframe.
     */
    private fun emitWindowApplyFuncFooter(builder: Module.Builder, arrs: List<Expr>, header: WindowApplyFuncHeader): List<String> {
        val outputFields = arrs.mapIndexed { i, arr -> Expr.StringLiteral("AGG_OUTPUT_$i") to arr }
        val retval = Variable("retval")
        builder.add(
            Op.Assign(
                retval, Expr.Call(
                    "pd.DataFrame",
                    args = listOf(Expr.Dict(outputFields)),
                    namedArgs = listOf(
                        "index" to header.index
                    )
                )
            )
        )
        builder.add(Op.ReturnStatement(retval))
        return outputFields.map { it.first.arg }
    }

    /**
     * Reverts the sorted order of the output dataframe using the original position index
     * if required. The original sort order is needed when:
     *
     * 1. An ordering was specified.
     * 2. The result ordering is meaningful.
     *
     * If no ordering was specified or if the result of the aggregate operation doesn't
     * have an ordering, we can save time by not reverting the sort.
     *
     * @see [WindowAggregateApplyFuncTable.isUnorderedResult]
     */
    private fun revertSortIfNeeded(builder: Module.Builder, arrs: List<Variable>, header: WindowApplyFuncHeader, orderKeys: List<OrderKey>): List<Expr> {
        if (orderKeys.isEmpty() || aggregates.all { WindowAggregateApplyFuncTable.isUnorderedResult(it.over) }) {
            return arrs
        }

        val target = Variable("_tmp_sorted_df")
        val args = arrs.mapIndexed { i, arr -> Expr.StringLiteral("AGG_OUTPUT_$i") to arr }
        val positionArg = Expr.StringLiteral(header.position) to
                Expr.Index(header.input, Expr.StringLiteral(header.position))
        val sortedDf = Expr.Call(
            Expr.Attribute(
                Expr.Call(
                    "pd.DataFrame",
                    Expr.Dict(args + positionArg)
                ),
                "sort_values",
            ),
            namedArgs = listOf(
                "by" to Expr.List(Expr.StringLiteral("ORIG_POSITION_COL")),
                "ascending" to Expr.List(Expr.BooleanLiteral(true)),
            )
        )
        builder.add(Op.Assign(target, sortedDf))

        // Need to return a new way of accessing the columns.
        return args.map { Expr.Index(target, it.first) }
    }

    /**
     * Emits the dataframe that results from applying the windowing function.
     */
    private fun emitGeneratedWindowDataframe(
        partitionKeys: List<PartitionKey>,
        orderKeys: List<OrderKey>,
        fields: List<Field>,
        extraFields: List<Pair<String, Expr>> = listOf(),
    ): Expr {
        // Build a list of parameters for the dataframe
        // initialization using the combination of partition keys,
        // order keys, and fields that will be needed in the body.
        val params = ImmutableList.builder<Pair<Expr, Expr>>()
        params.addAll(partitionKeys.map(::dictParam))
        params.addAll(orderKeys.map(::dictParam))
        params.addAll(fields.map(::dictParam))

        // Add any requested additional fields.
        params.addAll(extraFields.map { (k, v) -> Expr.StringLiteral(k) to v })

        // Construct a new dataframe from the partitions, orderings, and any referenced
        // columns.
        val windowDataframe = Expr.Call(
            "pd.DataFrame",
            Expr.Dict(params.build())
        )

        if (partitionKeys.isEmpty()) {
            return windowDataframe
        }

        // Group by the partition keys.
        return Expr.Groupby(
            windowDataframe,
            keys = Expr.List(partitionKeys.map { (k, _) -> Expr.StringLiteral(k) }),
            asIndex = false,
            dropna = false
        )
    }

    private fun dictParam(partitionKey: PartitionKey): Pair<Expr, Expr> =
        Expr.StringLiteral(partitionKey.field) to partitionKey.expr
    private fun dictParam(orderKey: OrderKey): Pair<Expr, Expr> =
        Expr.StringLiteral(orderKey.field) to orderKey.expr
    private fun dictParam(field: Field): Pair<Expr, Expr> =
        Expr.StringLiteral(field.name) to field.expr

    /**
     * Computes the partition column names along with the accessor expression for constructing the column.
     */
    private fun partitionBy(rexTranslator: RexToPandasTranslator): List<PartitionKey> =
        window.partitionKeys
            .mapIndexed { i, n -> PartitionKey("GRPBY_COL_$i", n.accept(rexTranslator)) }

    /**
     * Computes the ordering column names along with the accessor expression for constructing the column.
     */
    private fun orderBy(rexTranslator: RexToPandasTranslator): List<OrderKey> =
        window.orderKeys
            .mapIndexed { i, collation ->
                val expr = collation.left.accept(rexTranslator)
                val (name, asc) = when (collation.direction) {
                    // Choose a name to help make this more readable.
                    RelFieldCollation.Direction.ASCENDING -> Pair("ASC_COL_$i", true)
                    RelFieldCollation.Direction.DESCENDING -> Pair("DEC_COL_$i", false)
                    else -> throw AssertionError("invalid direction")
                }
                val nullsFirst = collation.nullDirection == RelFieldCollation.NullDirection.FIRST
                OrderKey(name, asc, nullsFirst, expr)
            }

    /**
     * Maps the inputs to the aggregate function to names for the windowed function.
     */
    private fun aggregateInputs(rexTranslator: RexToPandasTranslator): List<Field> =
        localRefs.mapIndexed { i, n -> Field("AGG_OP_$i", n.accept(rexTranslator), n.type) }
}
