package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import com.bodosql.calcite.ir.*
import com.bodosql.calcite.table.BodoSqlTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.prepare.RelOptTableImpl
import org.apache.calcite.rel.core.TableScan

class PandasTableScan(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    table: RelOptTable,
) : TableScan(cluster, traitSet, ImmutableList.of(), table), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun getTimerType() = SingleBatchRelNodeTimer.OperationType.IO_BATCH
    override fun operationDescriptor() = "reading table"
    override fun loggingTitle() = "IO TIMING"

    override fun nodeDetails(): String {
        val relTable = table as RelOptTableImpl
        val bodoSqlTable = relTable.table() as BodoSqlTable
        return bodoSqlTable.name
    }

    override fun emit(implementor: PandasRel.Implementor): Dataframe =
        if (isStreaming()) {
            implementor.createStreamingPipeline()
            implementor.buildStreaming (
                {ctx -> initStateVariable(ctx)},
                {ctx, stateVar -> generateStreamingDataFrame(ctx, stateVar)},
                {ctx, stateVar -> deleteStateVariable(ctx, stateVar)}
            )
        } else {
            implementor.build {ctx -> ctx.returns(generateNonStreamingDataframe(ctx))}
        }

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        val builder = ctx.builder()
        val currentPipeline = builder.getCurrentStreamingPipeline()
        val readerVar = builder.symbolTable.genStateVar()

        val bodoSQLTable = (table as RelOptTableImpl).table() as BodoSqlTable
        currentPipeline.addInitialization(Op.Assign(
            readerVar, bodoSQLTable.generateReadCode(true, ctx.streamingOptions())))

        return readerVar
    }

    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
        val currentPipeline = ctx.builder().getCurrentStreamingPipeline()
        val deleteState = Op.Stmt(Expr.Call("bodo.io.arrow_reader.arrow_reader_del", listOf(stateVar)))
        currentPipeline.addTermination(deleteState)
    }

    /**
     * Generate the DataFrame for the body of the streaming code.
     */
    private fun generateStreamingDataFrame(ctx: PandasRel.BuildContext, stateVar: StateVariable): Dataframe {
        val builder = ctx.builder()
        val currentPipeline = builder.getCurrentStreamingPipeline()
        val dfChunkVar = builder.genDataFrameAsTable(this)
        val isLastVar = currentPipeline.exitCond
        val readArrowNextCall = Expr.Call("bodo.io.arrow_reader.read_arrow_next", listOf(stateVar))
        builder.add(Op.TupleAssign(listOf(dfChunkVar, isLastVar), readArrowNextCall))

        // Convert the output to a DataFrame (TODO(kian): Remove)
        val idxVar = generateRangeIndex(ctx, dfChunkVar)
        val dfVar = builder.symbolTable.genDfVar()
        builder.add(Op.Assign(dfVar, generateInitDataframeCode(ctx, dfChunkVar, idxVar)))

        // Generate Cast Code within Loop
        val bodoSQLTable = (table as RelOptTableImpl).table() as BodoSqlTable
        return ctx.returns(bodoSQLTable.generateReadCastCode(dfVar))
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
     * This method takes a table from streaming and initializes
     * the dataframe with the column names and a fake index.
     *
     * @param ctx the pandas BuildContext for code generation.
     * @param input the input table with the underlying data
     * @param rangeIndex Var for index to use in dataframe
     */
    private fun generateInitDataframeCode(ctx: PandasRel.BuildContext, input: Variable, rangeIndex: Variable): Expr {
        val globalVarName = ctx.lowerAsGlobal(
            Expr.Call("ColNamesMetaType",
                Expr.Tuple(rowType.fieldNames.map { Expr.StringLiteral(it) })
            )
        )
        return Expr.Call("bodo.hiframes.pd_dataframe_ext.init_dataframe",
            Expr.Tuple(input),
            rangeIndex,
            globalVarName,
        )
    }


    /**
     * Generate the DataFrame for the body of non-streaming code
     */
    private fun generateNonStreamingDataframe(ctx: PandasRel.BuildContext): Expr {
        val builder = ctx.builder()
        val bodoSQLTable = (table as RelOptTableImpl).table() as BodoSqlTable

        val readDFVar = builder.symbolTable.genDfVar()
        val readExpr = bodoSQLTable.generateReadCode(false, ctx.streamingOptions())

        // Check if Casting DF to Remove Categorical Columns is Necessary
        val castExpr = bodoSQLTable.generateReadCastCode(readDFVar)
        return if (castExpr.equals(readDFVar)) {
            readExpr
        } else {
            builder.add(Op.Assign(readDFVar, readExpr))
            castExpr
        }
    }
}
