package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.adapter.common.ProjectUtils.Companion.generateDataFrame
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
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.sql.validate.SqlValidatorUtil

class BodoPhysicalProject(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    projects: List<RexNode>,
    rowType: RelDataType,
) : ProjectBase(cluster, traitSet.replace(BodoPhysicalRel.CONVENTION), ImmutableList.of(), input, projects, rowType), BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        projects: List<RexNode>,
        rowType: RelDataType,
    ): BodoPhysicalProject {
        return BodoPhysicalProject(cluster, traitSet, input, projects, rowType)
    }

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        val inputVar = implementor.visitChild(this.input, 0)
        // Choose the build implementation.
        // TODO(jsternberg): Go over this interface again. It feels to me
        // like the implementor could just choose the correct version
        // for us rather than having us check this condition ourselves.
        return if (isStreaming()) {
            emitStreaming(implementor, inputVar)
        } else {
            emitSingleBatch(implementor, inputVar)
        }
    }

    private fun emitStreaming(
        implementor: BodoPhysicalRel.Implementor,
        inputVar: BodoEngineTable,
    ): BodoEngineTable {
        return implementor.buildStreaming(
            BodoPhysicalRel.ProfilingOptions(reportOutTableSize = false, timeStateInitialization = false),
            { ctx -> initStateVariable(ctx) },
            {
                    ctx, stateVar ->
                val (projectExprs, localRefs) = genDataFrameWindowInputs(ctx, inputVar)
                val translator = ctx.streamingRexTranslator(inputVar, localRefs, stateVar)
                generateDataFrame(ctx, inputVar, translator, projectExprs, localRefs, projects, input)
            },
            { ctx, stateVar -> deleteStateVariable(ctx, stateVar) },
        )
    }

    private fun emitSingleBatch(
        implementor: BodoPhysicalRel.Implementor,
        inputVar: BodoEngineTable,
    ): BodoEngineTable {
        return implementor::build {
                ctx ->
            // Extract window aggregates and update the nodes.
            val (projectExprs, localRefs) = genDataFrameWindowInputs(ctx, inputVar)
            val translator = ctx.rexTranslator(inputVar, localRefs)
            generateDataFrame(ctx, inputVar, translator, projectExprs, localRefs, projects, input)
        }
    }

    /**
     * Generate the additional inputs to generateDataFrame after handling the Window
     * Functions.
     */
    private fun genDataFrameWindowInputs(
        ctx: BodoPhysicalRel.BuildContext,
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

    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        val builder = ctx.builder()
        val currentPipeline = builder.getCurrentStreamingPipeline()
        val readerVar = builder.symbolTable.genStateVar()
        currentPipeline.addInitialization(Op.Assign(readerVar, Expr.Call("bodo.libs.stream_dict_encoding.init_dict_encoding_state")))
        return readerVar
    }

    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        val currentPipeline = ctx.builder().getCurrentStreamingPipeline()
        val deleteState = Op.Stmt(Expr.Call("bodo.libs.stream_dict_encoding.delete_dict_encoding_state", listOf(stateVar)))
        currentPipeline.addTermination(deleteState)
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return projectProperty(projects, inputBatchingProperty)
    }

    companion object {
        fun create(
            input: RelNode,
            projects: List<RexNode>,
            fieldNames: List<String?>?,
        ): BodoPhysicalProject {
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
        ): BodoPhysicalProject {
            val cluster = input.cluster
            val mq = cluster.metadataQuery
            val traitSet =
                cluster.traitSet()
                    .replaceIfs(RelCollationTraitDef.INSTANCE) {
                        RelMdCollation.project(mq, input, projects)
                    }
            return create(cluster, traitSet, input, projects, rowType)
        }

        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            input: RelNode,
            projects: List<RexNode>,
            fieldNames: List<String?>?,
        ): BodoPhysicalProject {
            val rowType =
                RexUtil.createStructType(
                    cluster.typeFactory,
                    projects,
                    fieldNames,
                    SqlValidatorUtil.F_SUGGESTER,
                )
            return create(cluster, traitSet, input, projects, rowType)
        }

        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            input: RelNode,
            projects: List<RexNode>,
            rowType: RelDataType,
        ): BodoPhysicalProject {
            return BodoPhysicalProject(cluster, traitSet, input, projects, rowType)
        }
    }
}
