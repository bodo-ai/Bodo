package com.bodosql.calcite.prepare

import com.bodosql.calcite.adapter.iceberg.AbstractIcebergFilterRule.Companion.splitFilterConditions
import com.bodosql.calcite.adapter.iceberg.IcebergFilter
import com.bodosql.calcite.adapter.iceberg.IcebergProject
import com.bodosql.calcite.adapter.iceberg.IcebergRel
import com.bodosql.calcite.adapter.iceberg.IcebergSort
import com.bodosql.calcite.adapter.iceberg.IcebergTableScan
import com.bodosql.calcite.adapter.iceberg.IcebergToPandasConverter
import com.bodosql.calcite.adapter.pandas.PandasAggregate
import com.bodosql.calcite.adapter.pandas.PandasFilter
import com.bodosql.calcite.adapter.pandas.PandasProject
import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.adapter.pandas.PandasSort
import com.bodosql.calcite.adapter.snowflake.SnowflakeAggregate
import com.bodosql.calcite.adapter.snowflake.SnowflakeFilter
import com.bodosql.calcite.adapter.snowflake.SnowflakeProject
import com.bodosql.calcite.adapter.snowflake.SnowflakeRel
import com.bodosql.calcite.adapter.snowflake.SnowflakeSort
import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan
import com.bodosql.calcite.adapter.snowflake.SnowflakeToPandasConverter
import com.bodosql.calcite.application.RelationalAlgebraGenerator
import org.apache.calcite.plan.RelOptLattice
import org.apache.calcite.plan.RelOptMaterialization
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelOptPredicateList
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rel.RelVisitor
import org.apache.calcite.rex.BodoRexSimplify
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexExecutor
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexSimplify
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.tools.Program
import org.apache.calcite.util.Util

object IcebergConvertProgram : Program {
    override fun run(
        planner: RelOptPlanner,
        rel: RelNode,
        requiredOutputTraits: RelTraitSet,
        materializations: MutableList<RelOptMaterialization>,
        lattices: MutableList<RelOptLattice>,
    ): RelNode {
        return if (RelationalAlgebraGenerator.enableSnowflakeIcebergTables) {
            val rexBuilder = rel.cluster.rexBuilder
            val executor: RexExecutor = Util.first(planner.executor, RexUtil.EXECUTOR)
            val simplify = BodoRexSimplify(rexBuilder, RelOptPredicateList.EMPTY, executor)
            val shuttle = Visitor(rexBuilder, simplify)
            rel.accept(shuttle)
        } else {
            rel
        }
    }

    private class Visitor(private val rexBuilder: RexBuilder, private val simplify: RexSimplify) : RelShuttleImpl() {
        /**
         * Note the RelShuttleImpl() is design for logical nodes and therefore
         * isn't designed to run on Physical nodes. It does not have reflection
         * support and as a result we cannot add methods for our individual
         * implementations. We could replace this with a custom ReflectiveVisitor,
         * but this doesn't seem useful given time constraints
         */
        override fun visit(node: RelNode): RelNode {
            // All Snowflake nodes must go through here.
            // This dispatches on the correct implementation.
            return when (node) {
                is SnowflakeToPandasConverter -> {
                    visit(node)
                }

                is SnowflakeProject -> {
                    visit(node)
                }

                is SnowflakeFilter -> {
                    visit(node)
                }

                is SnowflakeSort -> {
                    visit(node)
                }

                is SnowflakeAggregate -> {
                    visit(node)
                }

                is SnowflakeTableScan -> {
                    visit(node)
                }

                else -> {
                    super.visit(node)
                }
            }
        }

        // Physical node implementations
        private fun visit(node: SnowflakeTableScan): RelNode {
            return if (node.getCatalogTable().isIcebergTable()) {
                IcebergTableScan.create(node.cluster, node.table!!, node.getCatalogTable())
                    .cloneWithProject(node.keptColumns)
            } else {
                node
            }
        }

        private fun visit(node: SnowflakeToPandasConverter): RelNode {
            return when (val newInput = visit(node.input)) {
                is SnowflakeRel -> {
                    // A node aborted the conversion process. Just return.
                    node
                }

                is PandasRel -> {
                    // Only partial conversion was possible.
                    newInput
                }

                else -> {
                    // Full conversion succeeded.
                    IcebergToPandasConverter(node.cluster, node.traitSet, newInput)
                }
            }
        }

        private fun visit(node: SnowflakeProject): RelNode {
            return when (val newInput = visit(node.input)) {
                is SnowflakeRel -> {
                    // A node aborted the code generation. Just return.
                    node
                }

                is PandasRel -> {
                    // Only partial conversion was possible.
                    PandasProject.create(newInput, node.projects, node.getRowType())
                }

                else -> {
                    // Try and push this node.
                    if (node.projects.all { x -> x is RexInputRef }) {
                        IcebergProject.create(
                            node.cluster,
                            node.traitSet,
                            newInput,
                            node.projects,
                            node.getRowType(),
                            node.getCatalogTable(),
                        )
                    } else {
                        // This node can't be pushed
                        val converter = IcebergToPandasConverter(node.cluster, node.traitSet, newInput)
                        PandasProject.create(converter, node.projects, node.getRowType())
                    }
                }
            }
        }

        private fun visit(node: SnowflakeFilter): RelNode {
            return when (val newInput = visit(node.input)) {
                is SnowflakeRel -> {
                    // A node aborted the code generation. Just return.
                    node
                }

                is PandasRel -> {
                    // Only partial conversion was possible.
                    PandasFilter.create(node.cluster, newInput, node.condition)
                }

                else -> {
                    val canPush = !(newInput as IcebergRel).containsIcebergSort()
                    // Try and push this node.
                    val (icebergCondition, pandasCondition) = splitFilterConditions(node, rexBuilder, simplify)
                    if (!canPush || icebergCondition == null) {
                        // Nothing can be pushed to Iceberg
                        val converter = IcebergToPandasConverter(node.cluster, node.traitSet, newInput)
                        PandasFilter.create(node.cluster, converter, node.condition)
                    } else if (pandasCondition == null) {
                        // Everything can be pushed to Iceberg
                        IcebergFilter.create(
                            node.cluster,
                            node.traitSet,
                            newInput,
                            icebergCondition,
                            node.getCatalogTable(),
                        )
                    } else {
                        // Part of the condition is pushable to Iceberg. Here we do the part that can
                        // be pushed in Iceberg and the other part in Pandas
                        val icebergFilter =
                            IcebergFilter.create(
                                node.cluster,
                                node.traitSet,
                                newInput,
                                icebergCondition,
                                node.getCatalogTable(),
                            )
                        val converter = IcebergToPandasConverter(node.cluster, node.traitSet, icebergFilter)
                        PandasFilter.create(node.cluster, converter, pandasCondition)
                    }
                }
            }
        }

        private fun visit(node: SnowflakeSort): RelNode {
            return when (val newInput = visit(node.input)) {
                is SnowflakeRel -> {
                    // A node aborted the code generation. Just return.
                    node
                }

                is PandasRel -> {
                    // Only partial conversion was possible.
                    PandasSort.create(newInput, node.collation, node.offset, node.fetch)
                }

                else -> {
                    val canPush = !(newInput as IcebergRel).containsIcebergSort()
                    if (canPush) {
                        IcebergSort.create(
                            node.cluster,
                            node.traitSet,
                            newInput,
                            node.collation,
                            node.offset,
                            node.fetch,
                            node.getCatalogTable(),
                        )
                    } else {
                        val converter = IcebergToPandasConverter(node.cluster, node.traitSet, newInput)
                        PandasSort.create(converter, node.collation, node.offset, node.fetch)
                    }
                }
            }
        }

        private fun visit(node: SnowflakeAggregate): RelNode {
            return when (val newInput = visit(node.input)) {
                is SnowflakeRel -> {
                    // A node aborted the code generation. Just return.
                    node
                }

                else -> {
                    if (!allowAggToIceberg(node, newInput)) {
                        // Abort conversion. Here we have found an Aggregation
                        // that reduces a table to a single scalar with no way of
                        // filtering on the table. This suggests SF could likely compute
                        // the result based on metadata, which would be much slower for us.
                        node
                    } else {
                        val aggInput =
                            if (newInput is IcebergRel) {
                                IcebergToPandasConverter(node.cluster, node.traitSet, newInput)
                            } else {
                                newInput
                            }
                        PandasAggregate.create(node.cluster, aggInput, node.groupSet, node.groupSets, node.aggCallList)
                    }
                }
            }
        }

        companion object {
            /**
             * Determine if the converted inputs for a Snowflake aggregate can avoid
             * a likely significant performance impact if we no longer push down
             * the aggregate. This should be removed once we have IcebergAggregate support.
             * https://bodo.atlassian.net/browse/BSE-2688
             */
            @JvmStatic
            private fun allowAggToIceberg(
                agg: SnowflakeAggregate,
                newInput: RelNode,
            ): Boolean {
                // This says we can compute in Iceberg despite the presence of an aggregate
                // if we are just computing a distinct or scalar computation depends on a filter
                // or limit, which can likely prevent metadata usage.
                return !agg.groupSet.isEmpty || containsIcebergFilterOrLimit(newInput)
            }

            /**
             * Determine if a given plan contains an IcebergFilter
             * or an IcebergSort.
             */
            @JvmStatic
            private fun containsIcebergFilterOrLimit(node: RelNode): Boolean {
                val visitor =
                    object : RelVisitor() {
                        // ~ Methods ----------------------------------------------------------------
                        override fun visit(
                            node: RelNode,
                            ordinal: Int,
                            parent: RelNode?,
                        ) {
                            if (node is IcebergFilter || node is IcebergSort) {
                                throw Util.FoundOne.NULL
                            }
                            node.childrenAccept(this)
                        }
                    }
                return try {
                    visitor.go(node)
                    false
                } catch (e: Util.FoundOne) {
                    true
                }
            }
        }
    }
}
