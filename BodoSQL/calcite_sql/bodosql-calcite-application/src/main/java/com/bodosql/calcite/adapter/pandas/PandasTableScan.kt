package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.OperatorType
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.table.BodoSqlTable
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.prepare.RelOptTableImpl
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.TableScan

class PandasTableScan(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    table: RelOptTable,
) : TableScan(cluster, traitSet.replace(PandasRel.CONVENTION), ImmutableList.of(), table), PandasRel {

    override fun copy(traitSet: RelTraitSet, inputs: MutableList<RelNode>?): RelNode {
        return PandasTableScan(cluster, traitSet, table)
    }

    override fun getTimerType() = SingleBatchRelNodeTimer.OperationType.IO_BATCH
    override fun operationDescriptor() = "reading table"
    override fun loggingTitle() = "IO TIMING"

    override fun nodeDetails(): String {
        val relTable = table as RelOptTableImpl
        val bodoSqlTable = relTable.table() as BodoSqlTable
        return bodoSqlTable.name
    }

    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable =
        if (isStreaming()) {
            implementor.createStreamingPipeline()
            implementor.buildStreaming(
                { ctx -> initStateVariable(ctx) },
                { ctx, stateVar -> generateStreamingTable(ctx, stateVar) },
                { ctx, stateVar -> deleteStateVariable(ctx, stateVar) },
            )
        } else {
            implementor.build { ctx -> ctx.returns(generateNonStreamingTable(ctx)) }
        }

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        val builder = ctx.builder()
        val currentPipeline = builder.getCurrentStreamingPipeline()
        val readerVar = builder.symbolTable.genStateVar()

        val bodoSQLTable = (table as RelOptTableImpl).table() as BodoSqlTable
        currentPipeline.initializeStreamingState(
            ctx.operatorID(),
            Op.Assign(
                readerVar,
                bodoSQLTable.generateReadCode(true, ctx.streamingOptions()),
            ),
            OperatorType.SNOWFLAKE_READ,
        )

        return readerVar
    }

    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
        val currentPipeline = ctx.builder().getCurrentStreamingPipeline()
        val deleteState = Op.Stmt(Expr.Call("bodo.io.arrow_reader.arrow_reader_del", listOf(stateVar)))
        currentPipeline.deleteStreamingState(ctx.operatorID(), deleteState)
    }

    /**
     * Generate the Table for the body of the streaming code.
     */
    private fun generateStreamingTable(ctx: PandasRel.BuildContext, stateVar: StateVariable): BodoEngineTable {
        val builder = ctx.builder()
        val currentPipeline = builder.getCurrentStreamingPipeline()
        val tableChunkVar = builder.symbolTable.genTableVar()
        val isLastVar = currentPipeline.getExitCond()
        val outputControl = builder.symbolTable.genOutputControlVar()
        currentPipeline.addOutputControl(outputControl)
        val readArrowNextCall = Expr.Call("bodo.io.arrow_reader.read_arrow_next", listOf(stateVar, outputControl))
        builder.add(Op.TupleAssign(listOf(tableChunkVar, isLastVar), readArrowNextCall))

        // Generate Cast Code within Loop
        val bodoSQLTable = (table as RelOptTableImpl).table() as BodoSqlTable
        return ctx.returns(bodoSQLTable.generateReadCastCode(tableChunkVar))
    }

    /**
     * Generate the Table for the body of non-streaming code
     */
    private fun generateNonStreamingTable(ctx: PandasRel.BuildContext): Expr {
        val builder = ctx.builder()
        val bodoSQLTable = (table as RelOptTableImpl).table() as BodoSqlTable

        val readDFVar = builder.symbolTable.genDfVar()
        val readExpr = bodoSQLTable.generateReadCode(false, ctx.streamingOptions())

        if (!bodoSQLTable.readRequiresIO()) {
            builder.add(Op.Assign(readDFVar, readExpr))
            return ctx.convertDfToTable(readDFVar, getRowType())
        }

        // Check if Casting DF to Remove Categorical Columns is Necessary
        val castExpr = bodoSQLTable.generateReadCastCode(readDFVar)
        return if (castExpr.equals(readDFVar)) {
            readExpr
        } else {
            builder.add(Op.Assign(readDFVar, readExpr))
            castExpr
        }
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        // Note: Types may be lazily computed so use getRowType() instead of rowType
        val bodoSqlTable = (table as? RelOptTableImpl)?.table() as? BodoSqlTable
        return ExpectedBatchingProperty.tableReadProperty(bodoSqlTable, getRowType())
    }

    companion object {
        @JvmStatic
        fun create(cluster: RelOptCluster, table: RelOptTable): PandasTableScan {
            return PandasTableScan(cluster, cluster.traitSet(), table)
        }
    }
}
