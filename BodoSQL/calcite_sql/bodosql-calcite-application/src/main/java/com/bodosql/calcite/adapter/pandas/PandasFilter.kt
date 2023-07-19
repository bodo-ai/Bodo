package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.Utils.IsScalar.isScalar
import com.bodosql.calcite.ir.*
import com.bodosql.calcite.ir.BodoSQLKernel
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.rel.core.FilterBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMdCollation
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver

class PandasFilter(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    child: RelNode,
    condition: RexNode
) : FilterBase(cluster, traitSet, child, condition), PandasRel {

    init {
        assert(convention == PandasRel.CONVENTION)
    }

    override fun copy(traitSet: RelTraitSet, input: RelNode, condition: RexNode): PandasFilter {
        return PandasFilter(cluster, traitSet, input, condition)
    }

    override fun emit(implementor: PandasRel.Implementor): Dataframe {
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
        // Extract windows from the condition and emit the code for them.
        val (windowAggregate, condition) = extractWindows(cluster, inputVar, this.condition)
        val localRefs = windowAggregate.emit(ctx)
        // Emit the code for the condition.
        return emit(ctx, inputVar, condition, localRefs, streamingStateVar)
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
    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
        val currentPipeline = ctx.builder().getCurrentStreamingPipeline()
        val deleteState = Op.Stmt(Expr.Call("bodo.libs.stream_dict_encoding.delete_dict_encoding_state", listOf(stateVar)))
        currentPipeline.addTermination(deleteState)
    }

    private fun emit(ctx: PandasRel.BuildContext, input: Dataframe, condition: RexNode, localRefs: List<Variable>, streamingStateVar: StateVariable? = null): Dataframe {
        val translator = ctx.rexTranslator(input, localRefs)
        val conditionExpr = condition.accept(translator).let { filter ->
            if (isScalarCondition()) {
                // If the output of this filter is a scalar, we need to
                // coerce it to an array value for the filter operation.
                coerceScalar(input, filter)
            } else {
                filter
            }
        }
        // Generate the filter df1[df2] operation and assign to the destination.
        return ctx.returns(Expr.GetItem(input, conditionExpr))
    }

    /**
     * Returns true if the condition returns a scalar value.
     */
    private fun isScalarCondition(): Boolean = isScalar(this.condition)

    /**
     * Coerces a scalar value to a boolean array.
     */
    private fun coerceScalar(input: Dataframe, filter: Expr): Expr =
        Expr.Call("bodo.utils.utils.full_type",
            Expr.Len(input),
            BodoSQLKernel("is_true", listOf(filter)),
            Expr.Attribute(
                Expr.Raw("bodo"),
                "boolean_array_type",
            )
        )

    companion object {
        fun create(cluster: RelOptCluster, input: RelNode, condition: RexNode): PandasFilter {
            val mq = cluster.metadataQuery
            val traitSet = cluster.traitSetOf(PandasRel.CONVENTION)
                .replaceIf(BatchingPropertyTraitDef.INSTANCE) {
                    if (RexOver.containsOver(condition)) {
                        BatchingProperty.SINGLE_BATCH
                    } else {
                        BatchingProperty.STREAMING
                    }
                }
                .replaceIfs(RelCollationTraitDef.INSTANCE) {
                    RelMdCollation.filter(mq, input)
                }
            return PandasFilter(cluster, traitSet, input, condition)
        }
    }
}
