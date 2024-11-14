package com.bodosql.calcite.adapter.common

import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelVisitor

/**
 * Tree visitor that tracks duplicate expressions in a tree
 * and enables "reversing" the tree by generating back edges
 * to each parent.
 */
class TreeReverserDuplicateTracker : RelVisitor() {
    // Store a way to reverse a tree by providing a mapping from node
    // ID to a list of parents. There should be no duplicates in the
    // parent list and this is designed for random access.
    private val treeReverse: HashMap<RelNode, MutableList<RelNode>> = HashMap()

    // Track the number of unique nodes that are parents to this RelNode.
    private val uniqueParentCounter: HashMap<RelNode, Int> = HashMap()

    // Map each RelNode to its maximum height in the tree. The maximum height
    // is 0 for leaves and for all other node 1 more than the maximum height
    // of any child.
    private val maxHeight: HashMap<RelNode, Int> = HashMap()

    override fun visit(
        rel: RelNode,
        ordinal: Int,
        parent: RelNode?,
    ) {
        val seenNode = treeReverse.contains(rel)
        if (seenNode) {
            treeReverse[rel]!!.add(parent!!)
            uniqueParentCounter[rel] = uniqueParentCounter[rel]!! + 1
        } else {
            if (parent != null) {
                treeReverse[rel] = mutableListOf(parent)
            } else {
                treeReverse[rel] = mutableListOf()
            }
            uniqueParentCounter[rel] = 1
        }
        if (!seenNode) {
            rel.childrenAccept(this)
            val maxChildHeight = rel.inputs.maxOfOrNull { maxHeight[it]!! } ?: 0
            maxHeight[rel] = maxChildHeight + 1
        }
    }

    fun getHeights(): HashMap<RelNode, Int> = maxHeight

    fun getReversedTree(): HashMap<RelNode, MutableList<RelNode>> = treeReverse

    fun getNodeCounts(): HashMap<RelNode, Int> = uniqueParentCounter
}
