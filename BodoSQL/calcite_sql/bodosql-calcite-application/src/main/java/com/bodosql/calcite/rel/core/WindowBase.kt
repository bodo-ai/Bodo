package com.bodosql.calcite.rel.core

import com.bodosql.calcite.application.logicalRules.BodoSQLReduceExpressionsRule
import com.bodosql.calcite.plan.makeCost
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollation
import org.apache.calcite.rel.RelFieldCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.core.Window
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexFieldCollation
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlAggFunction
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.util.ImmutableBitSet

open class WindowBase(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    hints: List<RelHint>,
    input: RelNode,
    constants: List<RexLiteral>,
    rowType: RelDataType,
    groups: List<Group>,
    val inputsToKeep: ImmutableBitSet,
) :
    Window(cluster, traitSet, hints, input, constants, rowType, groups) {
    override fun explainTerms(pw: RelWriter): RelWriter? {
        super.explainTerms(pw)
        pw.item("constants", constants)
        pw.item("keptInputs", inputsToKeep)
        return pw
    }

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        val rows = mq.getRowCount(input)
        // Count how many aggregation calls there are, but also how many
        // fusion groups they are. Multiple window groups could be the same
        // fusion group so long as they have the same partition/order keys.
        var nAggCalls = 0
        val fusionGroupSet: MutableSet<Pair<ImmutableBitSet, RelCollation>> = mutableSetOf()
        for (group in groups) {
            nAggCalls += group.aggCalls.size
            fusionGroupSet.add(Pair(group.keys, group.orderKeys))
        }
        val nGroups = fusionGroupSet.size
        // The cost scales with the number of agg calls (slightly) and the number of rows,
        // but scales drastically with the number of distinct fusion groups since each is another
        // groupby.apply or groupby.window call.
        val groupsCost = nGroups + nAggCalls / 100
        return planner.makeCost(rows = rows, cpu = rows * groupsCost)
    }

    override fun copy(constants: MutableList<RexLiteral>): WindowBase {
        throw UnsupportedOperationException("Copy must be implemented by WindowBase subclasses")
    }

    // Returns the number of input columns
    private fun getNumInputReferences(): Int {
        return input.rowType.fieldCount
    }

    // Retrieves the list of references from the child node that are
    // passed through.
    private fun getPassThroughReferences(): List<RexNode> {
        return inputsToKeep.map {
            RexInputRef(it, input.rowType.fieldList[it].type)
        }
    }

    // Retrieves the list of references from the aggregation calls
    // that refer to constants instead of references from
    // the inputs.
    fun getConstantReferences(): List<RexNode> {
        return constants.mapIndexed {
                idx, lit ->
            RexInputRef(idx + getNumInputReferences(), lit.type)
        }
    }

    // Creates a list of terms that could be used in a projection to
    // have the same effect as the Window node.
    fun convertToProjExprs(): List<RexNode> {
        val exprs: MutableList<RexNode> = mutableListOf()
        exprs.addAll(getPassThroughReferences())

        // Build a shuttle to replace each constant reference with
        // the actual constant
        val builder = cluster.rexBuilder
        val replacer =
            BodoSQLReduceExpressionsRule.RexReplacer(
                builder,
                getConstantReferences(),
                constants as List<RexNode>,
                constants.map { false },
            )

        // Convert each agg call within each group to a RexOver
        groups.forEach {
                group ->
            val partitionKeys = keyBitSetToInputRefs(group.keys, input)
            val orderKeys = group.collation().fieldCollations.map { relFieldCollationToRexFieldCollation(it, input) }
            group.aggCalls.forEach {
                    aggCall ->
                val asOver =
                    builder.makeOver(
                        aggCall.getType(),
                        aggCall.operator as SqlAggFunction,
                        aggCall.getOperands(),
                        partitionKeys,
                        ImmutableList.copyOf(orderKeys),
                        group.lowerBound,
                        group.upperBound,
                        group.isRows,
                        true,
                        false,
                        aggCall.distinct,
                        aggCall.ignoreNulls,
                    )
                exprs.add(asOver.accept(replacer))
            }
        }
        return exprs
    }

    companion object {
        // Convert a RelFieldCollation used for storing the order keys in a Window.Group to a
        // RexFieldCollation used for doing the same in a RexOver. Takes in the rel node
        // that the references refer to.
        @JvmStatic
        fun relFieldCollationToRexFieldCollation(
            rfc: RelFieldCollation,
            rel: RelNode,
        ): RexFieldCollation {
            val idx = rfc.fieldIndex
            val ref = RexInputRef(idx, rel.rowType.fieldList[idx].type)
            val sortInfo: MutableSet<SqlKind> = HashSet()
            if (rfc.direction.isDescending) {
                sortInfo.add(SqlKind.DESCENDING)
            }
            if (rfc.nullDirection == RelFieldCollation.NullDirection.FIRST) {
                sortInfo.add(SqlKind.NULLS_FIRST)
            } else {
                sortInfo.add(SqlKind.NULLS_LAST)
            }
            return RexFieldCollation(ref, sortInfo)
        }

        // Convert a ImmutableBitSet used for storing the partition keys in a Window.Group to a
        // list of RexNodes used for doing the same in a RexOver. Takes in the rel node
        // that the references refer to.
        @JvmStatic
        fun keyBitSetToInputRefs(
            keys: ImmutableBitSet,
            rel: RelNode,
        ): List<RexNode> {
            return keys.toList().map {
                RexInputRef(it, rel.rowType.fieldList[it].type)
            }
        }
    }
}
