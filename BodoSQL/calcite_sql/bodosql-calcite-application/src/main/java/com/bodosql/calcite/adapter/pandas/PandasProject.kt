package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.Utils.BodoArrayHelpers
import com.bodosql.calcite.application.Utils.IsScalar
import com.bodosql.calcite.ir.*
import com.bodosql.calcite.plan.Cost
import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.rel.core.ProjectBase
import com.bodosql.calcite.traits.BatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.*

class PandasProject(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    projects: List<RexNode>,
    rowType: RelDataType
) : ProjectBase(cluster, traitSet, ImmutableList.of(), input, projects, rowType), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(traitSet: RelTraitSet, input: RelNode, projects: List<RexNode>, rowType: RelDataType): Project {
        return PandasProject(cluster, traitSet, input, projects, rowType)
    }

    override fun emit(implementor: PandasRel.Implementor): Dataframe {
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

    private fun emitStreaming(implementor: PandasRel.Implementor, inputVar: Dataframe): Dataframe {
        return implementor.buildStreaming (
            {ctx -> initStateVariable(ctx)},
            {ctx, stateVar -> generateDataFrame(ctx, inputVar, stateVar)},
            {ctx, stateVar -> deleteStateVariable(ctx, stateVar)}
        )
    }

    private fun emitSingleBatch(implementor: PandasRel.Implementor, inputVar: Dataframe): Dataframe {
        return implementor::build {ctx -> generateDataFrame(ctx, inputVar)}
    }

    private fun generateDataFrame(ctx: PandasRel.BuildContext, inputVar: Dataframe, streamingStateVar: StateVariable? = null): Dataframe {
        try {
            if (canUseLoc()) {
                return generateLocCode(ctx, inputVar)
            }
            return generateProject(ctx, inputVar, streamingStateVar)
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

    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
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
     * Uses the .loc call to create a projection from [RexInputRef] values.
     *
     * This function assumes the projection only contains [RexInputRef] values
     * and those values do not have duplicates. The method [canUseLoc]
     * should be invoked before calling this function.
     */
    private fun generateLocCode(ctx: PandasRel.BuildContext, input: Dataframe): Dataframe {
        val colNames = namedProjects.asSequence()
            // Retrieve the input column names we are going to reference.
            // TODO(jsternberg): Can we just reference these by index? It would make
            // the generated code a bit harder to follow, but the reliability of the generated
            // code would probably go up.
            .map { (r, _) -> input.rowType.fieldNames[(r as RexInputRef).index] }
            .map { name -> Expr.StringLiteral(name) }
        val locExpr = Expr.Index(
            Expr.Attribute(input, "loc"),
            Expr.Slice(),
            Expr.List(colNames.toList())
        )

        val resultExpr = generateRenameIfNeeded(locExpr, input.rowType)
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
    private fun generateRenameIfNeeded(input: Expr, rowType: RelDataType): Expr {
        val renameMap = namedProjects.asSequence()
            .map { (r, alias) -> rowType.fieldNames[(r as RexInputRef).index] to alias }
            .filter { (name, alias) -> name != alias }
            .toList()
        if (renameMap.isEmpty()) {
            return input
        }

        return Expr.Method(
            input, "rename",
            namedArgs = listOf(
                "columns" to Expr.Dict(renameMap.map { (name, alias) ->
                    Expr.StringLiteral(name) to Expr.StringLiteral(alias)
                }),
                "copy" to Expr.BooleanLiteral(false)
            )
        )
    }

    /**
     * Generate the standard projection code. This is in contrast [generateLocCode]
     * which acts as just an index/rename operation.
     *
     * This is the general catch-all for most projections.
     */
    private fun generateProject(ctx: PandasRel.BuildContext, input: Dataframe, streamingStateVar: StateVariable?): Dataframe {
        // Extract window aggregates and replace them with local references.
        val (windowAggregate, projectExprs) = extractWindows(cluster, input, projects)

        // Emit the windows and turn this into a mutable list.
        // This is a bit strange, but we're going to add to this list
        // in this next section by evaluating any expressions that aren't
        // a RexSlot.
        val localRefs = windowAggregate.emit(ctx).toMutableList()

        // Evaluate projections into new series.
        // In order to optimize this, we only generate new series
        // for projections that are non-trivial (aka not a RexInputRef)
        // or ones that haven't already been computed (aka not a RexLocalRef).
        // Similar to over expressions, we replace non-trivial projections
        // with a RexLocalRef that reference our computed set of local variables.
        val builder = ctx.builder()
        val rexTranslator = ctx.rexTranslator(input, localRefs)

        // newProjectRefs will be a list of RexSlot values (either RexInputRef or RexLocalRef).
        val newProjectRefs = projectExprs.map { proj ->
            if (proj is RexSlot) {
                return@map proj
            }

            val expr = proj.accept(rexTranslator).let {
                if (isScalar(proj)) {
                    coerceScalarToArray(ctx, proj.type, it, input)
                } else {
                    it
                }
            }
            val series = builder.symbolTable.genSeriesVar()
            builder.add(Op.Assign(series, expr))
            localRefs.add(series)
            RexLocalRef(localRefs.lastIndex, proj.type)
        }

        // Generate the indices we will reference when creating the table.
        val indices = newProjectRefs.map { proj ->
            when (proj) {
                is RexInputRef -> proj.index
                is RexLocalRef -> proj.index + input.rowType.fieldCount
                else -> throw AssertionError("Internal Error: Projection must be InputRef or LocalRef")
            }
        }

        val rangeIndex = generateRangeIndex(ctx, input)
        val logicalTableVar = generateLogicalTableCode(ctx, input, indices, localRefs)
        return generateInitDataframeCode(ctx, logicalTableVar, rangeIndex)
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
    private fun coerceScalarToArray(ctx: PandasRel.BuildContext, dataType: RelDataType, scalar: Expr, input: Dataframe): Expr {
        val global = ctx.lowerAsGlobal(BodoArrayHelpers.sqlTypeToBodoArrayType(dataType, true))
        return Expr.Call("bodo.utils.conversion.coerce_scalar_to_array",
            scalar,
            Expr.Call("len", input),
            global)
    }

    /**
     * Produces a dummy index that can be used for the created table.
     *
     * BodoSQL never uses Index values and doing this avoid a MultiIndex issue
     * and allows Bodo to optimize more.
     */
    private fun generateRangeIndex(ctx: PandasRel.BuildContext, input: Dataframe): Variable {
        val builder = ctx.builder()
        val indexVar = builder.symbolTable.genIndexVar()
        builder.add(Op.Assign(indexVar, Expr.Call("bodo.hiframes.pd_index_ext.init_range_index",
            Expr.IntegerLiteral(0),
            Expr.Call("len", input),
            Expr.IntegerLiteral(1),
            Expr.None,
        )))
        return indexVar
    }

    /**
     * This method constructs a new table from a logical table and additional series.
     *
     * The logical table is constructed from a set of indices. The indices refer to
     * either the input dataframe or one of the additional series provided in the series list.
     *
     * @param implementor the pandas implementor for code generation.
     * @param input input dataframe.
     * @param indices list of indices to initialize the table with.
     * @param seriesList additional series that should be included in the list of indices.
     */
    private fun generateLogicalTableCode(ctx: PandasRel.BuildContext, input: Dataframe,
                                         indices: List<Int>, seriesList: List<Variable>): Variable {
        // Use the list of indices to generate a meta type with the column numbers.
        val metaType = ctx.lowerAsGlobal(
            Expr.Call("MetaType",
                Expr.Tuple(indices.map { Expr.IntegerLiteral(it) })
            )
        )

        // Generate the output table with logical_table_to_table.
        val logicalTableExpr = Expr.Call("bodo.hiframes.table.logical_table_to_table",
            Expr.Call("bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data", input),
            Expr.Tuple(seriesList),
            metaType,
            Expr.Index(
                Expr.Attribute(input, "shape"),
                Expr.IntegerLiteral(1),
            )
        )
        val builder = ctx.builder()
        val logicalTableVar = builder.symbolTable.genTableVar()
        builder.add(Op.Assign(logicalTableVar, logicalTableExpr))
        return logicalTableVar
    }

    /**
     * This method takes a logical table created from [generateLogicalTableCode] and initializes
     * the dataframe with the column names and a fake index.
     *
     * @param implementor the pandas implementor for code generation.
     * @param output the target output dataframe
     * @param input the input table with the underlying data
     */
    private fun generateInitDataframeCode(ctx: PandasRel.BuildContext, input: Variable, rangeIndex: Variable): Dataframe {
        val globalVarName = ctx.lowerAsGlobal(
            Expr.Call("ColNamesMetaType",
                Expr.Tuple(rowType.fieldNames.map { Expr.StringLiteral(it) })
            )
        )
        val initDataframeExpr = Expr.Call("bodo.hiframes.pd_dataframe_ext.init_dataframe",
            Expr.Tuple(input),
            rangeIndex,
            globalVarName,
        )
        return ctx.returns(initDataframeExpr)
    }

    companion object {
        fun create(input: RelNode, projects: List<RexNode>, rowType: RelDataType): PandasProject {
            val cluster = input.cluster
            val mq = cluster.metadataQuery
            val containsOver = RexOver.containsOver(projects, null)
            val batchProperty = if (containsOver) BatchingProperty.SINGLE_BATCH else BatchingProperty.STREAMING
            val traitSet = cluster.traitSet().replace(PandasRel.CONVENTION).replace(batchProperty)
                .replaceIfs(RelCollationTraitDef.INSTANCE) {
                    RelMdCollation.project(mq, input, projects)
                }
            return PandasProject(cluster, traitSet, input, projects, rowType)
        }
    }
}
