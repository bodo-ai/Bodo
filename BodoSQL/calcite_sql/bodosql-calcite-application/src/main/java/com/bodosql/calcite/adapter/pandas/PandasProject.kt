package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.utils.BodoArrayHelpers
import com.bodosql.calcite.application.utils.IsScalar
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.rel.core.ProjectBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty.Companion.projectProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLocalRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexSlot
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.sql.validate.SqlValidatorUtil

class PandasProject(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    projects: List<RexNode>,
    rowType: RelDataType,
) : ProjectBase(cluster, traitSet.replace(PandasRel.CONVENTION), ImmutableList.of(), input, projects, rowType), PandasRel {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        projects: List<RexNode>,
        rowType: RelDataType,
    ): PandasProject {
        return PandasProject(cluster, traitSet, input, projects, rowType)
    }

    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
        val inputVar = implementor.visitChild(this.input, 0)
        // Choose the build implementation.
        // TODO(jsternberg): Go over this interface again. It feels to me
        // like the implementor could just choose the correct version
        // for us rather than having us check this condition ourselves.
        if (isStreaming()) {
            return emitStreaming(implementor, inputVar)
        } else {
            return emitSingleBatch(implementor, inputVar)
        }
    }

    private fun emitStreaming(
        implementor: PandasRel.Implementor,
        inputVar: BodoEngineTable,
    ): BodoEngineTable {
        return implementor.buildStreaming(
            { ctx -> initStateVariable(ctx) },
            {
                    ctx, stateVar ->
                val (projectExprs, localRefs) = genDataFrameWindowInputs(ctx, inputVar)
                val translator = ctx.streamingRexTranslator(inputVar, localRefs, stateVar)
                generateDataFrame(ctx, inputVar, translator, projectExprs, localRefs)
            },
            { ctx, stateVar -> deleteStateVariable(ctx, stateVar) },
        )
    }

    private fun emitSingleBatch(
        implementor: PandasRel.Implementor,
        inputVar: BodoEngineTable,
    ): BodoEngineTable {
        return implementor::build {
                ctx ->
            // Extract window aggregates and update the nodes.
            val (projectExprs, localRefs) = genDataFrameWindowInputs(ctx, inputVar)
            val translator = ctx.rexTranslator(inputVar, localRefs)
            generateDataFrame(ctx, inputVar, translator, projectExprs, localRefs)
        }
    }

    /**
     * Generate the additional inputs to generateDataFrame after handling the Window
     * Functions.
     */
    private fun genDataFrameWindowInputs(
        ctx: PandasRel.BuildContext,
        inputVar: BodoEngineTable,
    ): Pair<List<RexNode>, MutableList<Variable>> {
        val (windowAggregate, projectExprs) = extractWindows(cluster, inputVar, projects)
        // Emit the windows and turn this into a mutable list.
        // This is a bit strange, but we're going to add to this list
        // in this next section by evaluating any expressions that aren't
        // a RexSlot.
        val localRefs = windowAggregate.emit(ctx).toMutableList()
        return Pair(projectExprs, localRefs)
    }

    private fun generateDataFrame(
        ctx: PandasRel.BuildContext,
        inputVar: BodoEngineTable,
        translator: RexToPandasTranslator,
        projectExprs: List<RexNode>,
        localRefs: MutableList<Variable>,
    ): BodoEngineTable {
        try {
            if (canUseLoc()) {
                return generateLocCode(ctx, inputVar)
            }
            return generateProject(ctx, inputVar, translator, projectExprs, localRefs)
        } catch (ex: Exception) {
            throw ex
        }
    }

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        val builder = ctx.builder()
        val currentPipeline = builder.getCurrentStreamingPipeline()
        val readerVar = builder.symbolTable.genStateVar()
        currentPipeline.addInitialization(Op.Assign(readerVar, Expr.Call("bodo.libs.stream_dict_encoding.init_dict_encoding_state")))
        return readerVar
    }

    override fun deleteStateVariable(
        ctx: PandasRel.BuildContext,
        stateVar: StateVariable,
    ) {
        val currentPipeline = ctx.builder().getCurrentStreamingPipeline()
        val deleteState = Op.Stmt(Expr.Call("bodo.libs.stream_dict_encoding.delete_dict_encoding_state", listOf(stateVar)))
        currentPipeline.addTermination(deleteState)
    }

    /**
     * Determines if we can use loc when generating code output.
     */
    private fun canUseLoc(): Boolean {
        val seen = hashSetOf<Int>()
        namedProjects.forEach { (r, _) ->
            if (r !is RexInputRef) {
                // If we have a non input ref we can't use the loc path
                return false
            }

            if (r.index in seen) {
                // When we have a situation with common subexpressions like "sum(A) as alias2, sum(A) as
                // alias from table1 groupby D" Calcite generates a plan like: LogicalProject(alias2=[$1],
                // alias=[$1]) LogicalAggregate(group=[{0}], alias=[SUM($1)]) In this case, we can't use
                // loc, as it would lead to duplicate column names in the output dataframe See
                // test_repeat_columns in BodoSQL/bodosql/tests/test_agg_groupby.py
                return false
            }
            seen.add(r.index)
        }
        return true
    }

    /**
     * Uses table_subset to create a projection from [RexInputRef] values.
     *
     * This function assumes the projection only contains [RexInputRef] values
     * and those values do not have duplicates. The method [canUseLoc]
     * should be invoked before calling this function.
     */
    private fun generateLocCode(
        ctx: PandasRel.BuildContext,
        input: BodoEngineTable,
    ): BodoEngineTable {
        val colIndices = getProjects().map { r -> Expr.IntegerLiteral((r as RexInputRef).index) }
        val typeCall = Expr.Call("MetaType", Expr.Tuple(colIndices))
        val colNamesMeta = ctx.lowerAsGlobal(typeCall)
        val resultExpr = Expr.Call("bodo.hiframes.table.table_subset", input, colNamesMeta, Expr.False)
        return ctx.returns(resultExpr)
    }

    /**
     * Generate rename code if aliases are involved.
     *
     * This method can only operate when using the [generateLocCode] path and assumes
     * the inputs are [RexInputRef] values.
     *
     * If a rename is not necessary, this method returns its input.
     */
    private fun generateRenameIfNeeded(
        input: Expr,
        rowType: RelDataType,
    ): Expr {
        val renameMap =
            namedProjects.asSequence()
                .map { (r, alias) -> rowType.fieldNames[(r as RexInputRef).index] to alias }
                .filter { (name, alias) -> name != alias }
                .toList()
        if (renameMap.isEmpty()) {
            return input
        }

        return Expr.Method(
            input,
            "rename",
            namedArgs =
                listOf(
                    "columns" to
                        Expr.Dict(
                            renameMap.map { (name, alias) ->
                                Expr.StringLiteral(name) to Expr.StringLiteral(alias)
                            },
                        ),
                    "copy" to Expr.BooleanLiteral(false),
                ),
        )
    }

    /**
     * Generate the standard projection code. This is in contrast [generateLocCode]
     * which acts as just an index/rename operation.
     *
     * This is the general catch-all for most projections.
     */
    private fun generateProject(
        ctx: PandasRel.BuildContext,
        inputVar: BodoEngineTable,
        translator: RexToPandasTranslator,
        projectExprs: List<RexNode>,
        localRefs: MutableList<Variable>,
    ): BodoEngineTable {
        // Evaluate projections into new series.
        // In order to optimize this, we only generate new series
        // for projections that are non-trivial (aka not a RexInputRef)
        // or ones that haven't already been computed (aka not a RexLocalRef).
        // Similar to over expressions, we replace non-trivial projections
        // with a RexLocalRef that reference our computed set of local variables.
        val builder = ctx.builder()

        // newProjectRefs will be a list of RexSlot values (either RexInputRef or RexLocalRef).
        val newProjectRefs =
            projectExprs.map { proj ->
                if (proj is RexSlot) {
                    return@map proj
                }

                val expr =
                    proj.accept(translator).let {
                        if (isScalar(proj)) {
                            coerceScalarToArray(ctx, proj.type, it, inputVar)
                        } else {
                            it
                        }
                    }
                val arr = builder.symbolTable.genArrayVar()
                builder.add(Op.Assign(arr, expr))
                localRefs.add(arr)
                RexLocalRef(localRefs.lastIndex, proj.type)
            }

        // Generate the indices we will reference when creating the table.
        val indices =
            newProjectRefs.map { proj ->
                when (proj) {
                    is RexInputRef -> proj.index
                    is RexLocalRef -> proj.index + input.getRowType().fieldCount
                    else -> throw AssertionError("Internal Error: Projection must be InputRef or LocalRef")
                }
            }
        val logicalTableVar = generateLogicalTableCode(ctx, inputVar, indices, localRefs, input.getRowType().fieldCount)
        return ctx.returns(logicalTableVar)
    }

    /**
     * Determines if a node is a scalar.
     *
     * @param node input node to determine if it is a scalar or not.
     */
    private fun isScalar(node: RexNode): Boolean = IsScalar.isScalar(node)

    /**
     * Generates code to coerce a scalar value into an array.
     *
     * @param implementor the pandas implementor for code generation.
     * @param dataType the sql data type for the array type.
     * @param scalar the expression that refers to the scalar value.
     * @param input the input dataframe.
     */
    private fun coerceScalarToArray(
        ctx: PandasRel.BuildContext,
        dataType: RelDataType,
        scalar: Expr,
        input: BodoEngineTable,
    ): Expr {
        val global = ctx.lowerAsGlobal(BodoArrayHelpers.sqlTypeToBodoArrayType(dataType, true, ctx.builder().defaultTz.zoneExpr))
        return Expr.Call(
            "bodo.utils.conversion.coerce_scalar_to_array",
            scalar,
            Expr.Call("len", input),
            global,
        )
    }

    /**
     * This method constructs a new table from a logical table and additional series.
     *
     * The logical table is constructed from a set of indices. The indices refer to
     * either the input dataframe or one of the additional series provided in the series list.
     *
     * @param implementor the pandas implementor for code generation.
     * @param input input table.
     * @param indices list of indices to initialize the table with.
     * @param seriesList additional series that should be included in the list of indices.
     * @param colsBeforeProject number of columns in the input table before any projection occurs.
     */
    private fun generateLogicalTableCode(
        ctx: PandasRel.BuildContext,
        input: BodoEngineTable,
        indices: List<Int>,
        seriesList: List<Variable>,
        colsBeforeProject: Int,
    ): Variable {
        // Use the list of indices to generate a meta type with the column numbers.
        val metaType =
            ctx.lowerAsGlobal(
                Expr.Call(
                    "MetaType",
                    Expr.Tuple(indices.map { Expr.IntegerLiteral(it) }),
                ),
            )

        // Generate the output table with logical_table_to_table.
        val logicalTableExpr =
            Expr.Call(
                "bodo.hiframes.table.logical_table_to_table",
                input,
                Expr.Tuple(seriesList),
                metaType,
                Expr.IntegerLiteral(colsBeforeProject),
            )
        val builder = ctx.builder()
        val logicalTableVar = builder.symbolTable.genTableVar()
        builder.add(Op.Assign(logicalTableVar, logicalTableExpr))
        return logicalTableVar
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return projectProperty(projects, inputBatchingProperty)
    }

    companion object {
        fun create(
            input: RelNode,
            projects: List<RexNode>,
            fieldNames: List<String?>?,
        ): PandasProject {
            val cluster = input.cluster
            val rowType =
                RexUtil.createStructType(
                    cluster.typeFactory,
                    projects,
                    fieldNames,
                    SqlValidatorUtil.F_SUGGESTER,
                )
            return create(input, projects, rowType)
        }

        fun create(
            input: RelNode,
            projects: List<RexNode>,
            rowType: RelDataType,
        ): PandasProject {
            val cluster = input.cluster
            val mq = cluster.metadataQuery
            val traitSet =
                cluster.traitSet()
                    .replaceIfs(RelCollationTraitDef.INSTANCE) {
                        RelMdCollation.project(mq, input, projects)
                    }
            return PandasProject(cluster, traitSet, input, projects, rowType)
        }
    }
}
