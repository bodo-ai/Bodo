package com.bodosql.calcite.rel.logical

import com.bodosql.calcite.rel.core.WindowBase
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rel.logical.LogicalWindow
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.util.ImmutableBitSet

class BodoLogicalWindow(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    hints: List<RelHint>,
    input: RelNode,
    constants: List<RexLiteral>,
    rowType: RelDataType,
    groups: List<Group>,
    inputsToKeep: ImmutableBitSet,
) :
    WindowBase(cluster, traitSet, hints, input, constants, rowType, groups, inputsToKeep) {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): BodoLogicalWindow {
        return BodoLogicalWindow(cluster, traitSet, hints, inputs[0], constants, rowType, groups, inputsToKeep)
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            hints: List<RelHint>,
            input: RelNode,
            constants: List<RexLiteral>,
            rowType: RelDataType,
            groups: List<Group>,
            inputsToKeep: ImmutableBitSet,
        ): BodoLogicalWindow {
            return BodoLogicalWindow(cluster, traitSet, hints, input, constants, rowType, groups, inputsToKeep)
        }

        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            hints: List<RelHint>,
            input: RelNode,
            constants: List<RexLiteral>,
            rowType: RelDataType,
            groups: List<Group>,
        ): BodoLogicalWindow {
            return BodoLogicalWindow(
                cluster,
                traitSet,
                hints,
                input,
                constants,
                rowType,
                groups,
                ImmutableBitSet.range(input.rowType.fieldCount),
            )
        }

        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            hints: List<RelHint>,
            input: RelNode,
            constants: List<RexLiteral>,
            rowType: RelDataType,
            groups: List<Group>,
            inputsToKeep: ImmutableBitSet,
        ): BodoLogicalWindow {
            return create(cluster, input.traitSet, hints, input, constants, rowType, groups, inputsToKeep)
        }

        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            hints: List<RelHint>,
            input: RelNode,
            constants: List<RexLiteral>,
            rowType: RelDataType,
            groups: List<Group>,
        ): BodoLogicalWindow {
            return create(cluster, hints, input, constants, rowType, groups, ImmutableBitSet.range(input.rowType.fieldCount))
        }

        @JvmStatic
        fun fromLogicalWindow(win: LogicalWindow): BodoLogicalWindow {
            return create(win.cluster, win.traitSet, win.hints, win.input, win.constants, win.rowType, win.groups)
        }
    }
}
