package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.utils.IsScalar.isScalar
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.ir.bodoSQLKernel
import com.bodosql.calcite.rel.core.FilterBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rex.RexNode

class PandasFilter(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    child: RelNode,
    condition: RexNode,
) : FilterBase(cluster, traitSet.replace(PandasRel.CONVENTION), child, condition), PandasRel {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        condition: RexNode,
    ): PandasFilter {
        return PandasFilter(cluster, traitSet, input, condition)
    }

    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
        val inputVar = implementor.visitChild(input, 0)
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
            true,
            { ctx -> initStateVariable(ctx) },
            { ctx, stateVar ->
                // Extract window aggregates and update the nodes.
                val (condition, inputRefs) = genDataFrameWindowInputs(ctx, inputVar)
                val translator = ctx.streamingRexTranslator(inputVar, inputRefs, stateVar)
                emit(ctx, translator, inputVar, condition)
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
            val (condition, inputRefs) = genDataFrameWindowInputs(ctx, inputVar)
            val translator = ctx.rexTranslator(inputVar, inputRefs)
            emit(ctx, translator, inputVar, condition)
        }
    }

    /**
     * Generate the additional inputs to generateDataFrame after handling the Window
     * Functions.
     */
    private fun genDataFrameWindowInputs(
        ctx: PandasRel.BuildContext,
        inputVar: BodoEngineTable,
    ): Pair<RexNode, List<Variable>> {
        val (windowAggregate, condition) = extractWindows(cluster, inputVar, this.condition)
        val localRefs = windowAggregate.emit(ctx)
        return Pair(condition, localRefs)
    }

    /**
     * Function to create the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        val builder = ctx.builder()
        val currentPipeline = builder.getCurrentStreamingPipeline()
        val readerVar = builder.symbolTable.genStateVar()
        currentPipeline.addInitialization(Op.Assign(readerVar, Expr.Call("bodo.libs.stream_dict_encoding.init_dict_encoding_state")))
        return readerVar
    }

    /**
     * Function to delete the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun deleteStateVariable(
        ctx: PandasRel.BuildContext,
        stateVar: StateVariable,
    ) {
        val currentPipeline = ctx.builder().getCurrentStreamingPipeline()
        val deleteState = Op.Stmt(Expr.Call("bodo.libs.stream_dict_encoding.delete_dict_encoding_state", listOf(stateVar)))
        currentPipeline.addTermination(deleteState)
    }

    private fun emit(
        ctx: PandasRel.BuildContext,
        translator: RexToPandasTranslator,
        input: BodoEngineTable,
        condition: RexNode,
    ): BodoEngineTable {
        val conditionExpr =
            condition.accept(translator).let { filter ->
                if (isScalarCondition()) {
                    // If the output of this filter is a scalar, we need to
                    // coerce it to an array value for the filter operation.
                    coerceScalar(input, filter)
                } else {
                    filter
                }
            }
        // Generate the filter table_filter(T, cond) operation and assign to the destination.
        return ctx.returns(Expr.Call("bodo.hiframes.table.table_filter", input, conditionExpr))
    }

    /**
     * Returns true if the condition returns a scalar value.
     */
    private fun isScalarCondition(): Boolean = isScalar(this.condition)

    /**
     * Coerces a scalar value to a boolean array.
     */
    private fun coerceScalar(
        input: BodoEngineTable,
        filter: Expr,
    ): Expr =
        Expr.Call(
            "bodo.utils.utils.full_type",
            Expr.Len(input),
            bodoSQLKernel("is_true", listOf(filter)),
            Expr.Attribute(
                Expr.Raw("bodo"),
                "boolean_array_type",
            ),
        )

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return ExpectedBatchingProperty.filterProperty(condition)
    }

    companion object {
        fun create(
            cluster: RelOptCluster,
            input: RelNode,
            condition: RexNode,
        ): PandasFilter {
            val mq = cluster.metadataQuery
            val traitSet =
                cluster.traitSet().replaceIfs(RelCollationTraitDef.INSTANCE) {
                    RelMdCollation.filter(mq, input)
                }
            return PandasFilter(cluster, traitSet, input, condition)
        }
    }
}
