package com.bodosql.calcite.rel.core

import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.AbstractRelNode
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.util.ImmutableBitSet

/**
 * Definition for the Flatten RelNode. This RelNode is intended for function with "flatten"
 * properties, which are functions explode a single row into several rows.
 *
 * A Flatten Node contains a single input
 * and then 4 core properties:
 *      - call: The function operation being performed. This RelNode must contain exactly
 *      one function call and every input to the function must be either a RexInputRef, RexDynamicParam/RexNamedParam
 *      or a RexLiteral.
 *      - callType: The return type for the call. All table functions return a cursor by default, so this must be passed
 *      explicitly.
 *      - usedColOutputs: A bitmap to indicate which of the outputs to the function are actually used. This will later
 *      be used as an entry point to enable columns pruning on the individual function outputs.
 *      - repeatColumns: A list of column numbers in the input that should be included in the output. For each of
 *      these columns if the flatten functions results in mapping row i -> rows [k, k + j] then the value in i will
 *      be repeated in every location.
 *
 * The primary motivation for giving this its own RelNode (as opposed to say using projection) is to enable future
 * optimizations. Flatten functions will generally be restricted to a single function and often grow the output
 * (as opposed to projection which keeps it the same). Additionally, the single function impacts all columns in a way
 * that we want to enable significant optimizations.
 *
 * A background into the design process can be found here: https://bodo.atlassian.net/wiki/spaces/B/pages/1469480977/Flatten+Operator+Design
 *
 * TODO(njriasan): Update this to use SingleRel? This node has a lot of the single Rel functionality, but it violates the undocumented
 * typing assumptions, so it will break all of the metadata APIs.
 *
 */
abstract class Flatten(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    var input: RelNode,
    val call: RexCall,
    val callType: RelDataType,
    val usedColOutputs: ImmutableBitSet,
    val repeatColumns: ImmutableBitSet,
) : AbstractRelNode(cluster, traits) {
    // ~ Methods ----------------------------------------------------------------
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): Flatten {
        return copy(traitSet, sole(inputs), call, callType, usedColOutputs, repeatColumns)
    }

    fun copy(
        traitSet: RelTraitSet,
        newInput: RelNode,
        newCall: RexCall,
        newCallType: RelDataType,
    ): Flatten {
        // If the call is changed make sure we mark all output columns as used.
        return copy(traitSet, newInput, newCall, newCallType, ImmutableBitSet.range(newCallType.fieldCount), repeatColumns)
    }

    fun copy(
        traitSet: RelTraitSet,
        newInput: RelNode,
        newCall: RexCall,
        callType: RelDataType,
        usedColOutputs: ImmutableBitSet,
    ): Flatten {
        return copy(traitSet, newInput, newCall, callType, usedColOutputs, repeatColumns)
    }

    abstract fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        call: RexCall,
        callType: RelDataType,
        usedColOutputs: ImmutableBitSet,
        repeatColumns: ImmutableBitSet,
    ): Flatten

    override fun getInputs(): List<RelNode> {
        return listOf(input)
    }

    override fun replaceInput(
        ordinalInParent: Int,
        rel: RelNode?,
    ) {
        assert(ordinalInParent == 0)
        input = rel!!
        recomputeDigest()
    }

    override fun explainTerms(pw: RelWriter): RelWriter {
        super.explainTerms(pw).input("input", input)
        pw.item("Call", call)
        pw.item("Call Return Type", callType)
        pw.item("Kept output columns", usedColOutputs)
        pw.item("Repeated columns", repeatColumns)
        return pw
    }

    override fun accept(shuttle: RexShuttle): RelNode? {
        val call = shuttle.apply(this.call)
        if (call !is RexCall) {
            throw RuntimeException("Flatten's call must be a RexCall")
        }
        return if (call === this.call) {
            this
        } else {
            copy(traitSet, input, call, callType, usedColOutputs, repeatColumns)
        }
    }

    /**
     * Create the row type by placing the kept outputs from the function call first, followed
     * by any columns that are kept.
     */
    override fun deriveRowType(): RelDataType {
        val outputNames = ArrayList<String>()
        val outputTypes = ArrayList<RelDataType>()
        // Keep the columns from the function call
        for (usedOutput in usedColOutputs) {
            outputNames.add(callType.fieldNames[usedOutput])
            outputTypes.add(callType.fieldList[usedOutput].type)
        }
        // Keep any repeat columns
        if (!repeatColumns.isEmpty) {
            val inputRowType = input.getRowType()
            for (keptInput in repeatColumns) {
                outputNames.add(inputRowType.fieldNames[keptInput])
                outputTypes.add(inputRowType.fieldList[keptInput].type)
            }
        }
        return cluster.typeFactory.createStructType(outputTypes, outputNames)
    }
}
