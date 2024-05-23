package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.codeGeneration.OperatorEmission
import com.bodosql.calcite.codeGeneration.OutputtingPipelineEmission
import com.bodosql.calcite.codeGeneration.OutputtingStageEmission
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.table.BodoSqlTable
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptPlanner
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
    override fun copy(
        traitSet: RelTraitSet,
        inputs: MutableList<RelNode>?,
    ): RelNode {
        return PandasTableScan(cluster, traitSet, table)
    }

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable =
        if (isStreaming()) {
            val stage =
                OutputtingStageEmission(
                    { ctx, stateVar, _ ->
                        generateStreamingTable(ctx, stateVar)
                    },
                    reportOutTableSize = false,
                )
            val pipeline =
                OutputtingPipelineEmission(
                    listOf(stage),
                    true,
                    null,
                )
            val operatorEmission =
                OperatorEmission(
                    { ctx -> initStateVariable(ctx) },
                    { ctx, stateVar -> deleteStateVariable(ctx, stateVar) },
                    listOf(),
                    pipeline,
                    timeStateInitialization = true,
                )
            implementor.buildStreaming(operatorEmission)!!
        } else {
            implementor.build { ctx -> ctx.returns(generateNonStreamingTable(ctx)) }
        }

    fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        val builder = ctx.builder()
        val currentPipeline = builder.getCurrentStreamingPipeline()
        val readerVar = builder.symbolTable.genStateVar()

        val bodoSQLTable = (table as RelOptTableImpl).table() as BodoSqlTable
        currentPipeline.addInitialization(
            Op.Assign(
                readerVar,
                bodoSQLTable.generateReadCode(true, ctx.streamingOptions()),
            ),
        )

        return readerVar
    }

    fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        val currentPipeline = ctx.builder().getCurrentStreamingPipeline()
        val deleteState = Op.Stmt(Expr.Call("bodo.io.arrow_reader.arrow_reader_del", listOf(stateVar)))
        currentPipeline.addTermination(deleteState)
    }

    /**
     * Generate the Table for the body of the streaming code.
     */
    private fun generateStreamingTable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ): BodoEngineTable {
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
    private fun generateNonStreamingTable(ctx: BodoPhysicalRel.BuildContext): Expr {
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

    override fun register(planner: RelOptPlanner) {
        for (rule in PandasRules.rules()) {
            planner.addRule(rule)
        }
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            relOptTable: RelOptTable,
        ): PandasTableScan {
            return PandasTableScan(cluster, traitSet, relOptTable)
        }

        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            table: RelOptTable,
        ): PandasTableScan {
            return create(cluster, cluster.traitSet(), table)
        }
    }
}
