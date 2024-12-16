package com.bodosql.calcite.application.logicalRules

import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.plan.RelRule
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.rex.RexVisitorImpl
import org.apache.calcite.tools.RelBuilder
import java.util.BitSet
import java.util.Stack

/**
 * Extracts any OVER calls from the join condition.
 *
 * We cannot handle an OVER call inside a join, so we need to extract
 * it from the condition into its own column and then remove the column after
 * the join has been performed.
 */
abstract class AbstractJoinExtractOverRule protected constructor(
    config: Config,
) : RelRule<AbstractJoinExtractOverRule.Config>(config) {
    override fun onMatch(call: RelOptRuleCall?) {
        if (call == null) {
            return
        }

        // Retrieve the join call.
        val join: Join = call.rel(0)

        // Retrieve the builder and construct the extractor with the join condition.
        val extractor = RexOverExtractor(join.condition)
        val rel =
            call
                .builder()
                .also { b -> extractor.push(b, join.left) }
                .also { b -> extractor.push(b, join.right) }
                .also { b -> extractor.join(b, join.joinType) }
                .build()
        call.transformTo(rel)
    }

    private class RexOverExtractor(
        var condition: RexNode,
    ) {
        /**
         * Controls the start index for variables referenced in the condition.
         * Each time we push a relational node, we map the fields of that
         * relational node to the condition. This is an offset for the starting index.
         */
        private var start: Int = 0

        /**
         * Stores the fields that should be preserved after the join is created.
         */
        private var preserve = BitSet()

        /**
         * Pushes the RelNode into the builder and rewrites references in the
         * condition when a RexOver was referenced.
         *
         * This will only modify input references that reference the given relational
         * node. This method should be used on both the left and right side of the
         * join condition to get a full rewrite.
         */
        fun push(
            builder: RelBuilder,
            rel: RelNode,
        ) {
            builder.push(rel)

            // Determine the range of fields that are related to the pushed
            // relation. We keep track of the start index and the number of
            // fields.
            val range = start until rel.rowType.fieldCount + start

            // Retrieve a list of all RexOver's that contain only
            // the set of variables we are looking for and replace them with
            // dummy expressions.
            val nodes = mutableListOf<RexOver>()
            condition.accept(
                object : RexVisitorImpl<Unit>(true) {
                    private val stack = Stack<Boolean>()

                    override fun visitInputRef(inputRef: RexInputRef?) {
                        // Ignore if this input reference is not in an over expression.
                        if (stack.isNotEmpty()) {
                            // Only input references that are included in the selected
                            // range are valid.
                            val isValid = range.contains(inputRef!!.index)
                            stack.push(stack.pop() && isValid)
                        }
                    }

                    override fun visitOver(over: RexOver?) {
                        // Ensure that all input references inside this RexOver
                        // are valid.
                        stack.push(true)
                        super.visitOver(over)
                        val isValid = stack.pop()
                        if (isValid) {
                            // This expression only references input refs that
                            // we are presently rewriting.
                            nodes.add(over!!)
                        }
                    }
                },
            )

            // Modify the bitset to mark each of the fields above as preserved.
            for (i in 0 until rel.rowType.fieldCount) {
                preserve.set(start + i)
            }

            // Now that we've processed the RexOver conditions, adjust the starting offset.
            // The subsequent code in this function relies on this change to produce the correct
            // rewrites.
            //
            // We do this regardless of whether or not we need to rewrite anything because
            // future calls will rely on this change.
            start += rel.rowType.fieldCount

            if (nodes.isNotEmpty()) {
                // Generate a projection that includes the new nodes we are creating.
                project(builder, rel, nodes)

                // Mark these new fields as ones we will not be preserving and rewrite
                // the condition so input refs reference the correct location.
                for (i in 0 until nodes.size) {
                    preserve.clear(start + i)
                }
                condition = conditionRewrite(condition, nodes)

                // Increment the start index now that all input refs refer to the correct
                // index.
                start += nodes.size
            }
        }

        fun project(
            builder: RelBuilder,
            rel: RelNode,
            exprs: List<RexOver>,
        ) {
            val nodes = mutableListOf<RexNode>()
            for (i in 0 until rel.rowType.fieldCount) {
                nodes.add(builder.field(i))
            }
            nodes.addAll(exprs)
            builder.project(nodes)
        }

        /**
         * Rewrite the condition. This is where we rewrite any generated variables
         * to their proper index and shift successive variables to their new indices.
         */
        fun conditionRewrite(
            condition: RexNode,
            nodes: List<RexOver>,
        ): RexNode =
            condition.accept(
                object : RexShuttle() {
                    override fun visitInputRef(inputRef: RexInputRef?): RexNode =
                        when (val index = inputRef!!.index) {
                            // Input reference for a value that is now at a different index.
                            in start..Int.Companion.MAX_VALUE -> RexInputRef(index + nodes.size, inputRef.type)
                            // Input reference we should not touch.
                            else -> inputRef
                        }

                    override fun visitOver(over: RexOver?): RexNode =
                        when (val index = nodes.indexOf(over!!)) {
                            -1 -> over
                            else -> RexInputRef(start + index, over.type)
                        }
                },
            )

        /**
         * Generates the join node using the rewritten condition.
         * If additional nodes were created, this will also generate a projection that
         * removes them.
         */
        fun join(
            builder: RelBuilder,
            joinType: JoinRelType,
        ) {
            builder.join(joinType, condition)

            // If we didn't add any new fields, there's no need for a projection.
            // This should be impossible because this rule wouldn't have run, but
            // might as well just check for it.
            if (preserve.cardinality() == preserve.size()) {
                return
            }

            // Generate a projection with only the fields we want to preserve.
            val nodes = ArrayList<RexNode>(preserve.cardinality())
            for (i in 0 until preserve.size()) {
                if (preserve[i]) {
                    nodes.add(builder.field(i))
                }
            }
            builder.project(nodes)
        }
    }

    interface Config : RelRule.Config
}
