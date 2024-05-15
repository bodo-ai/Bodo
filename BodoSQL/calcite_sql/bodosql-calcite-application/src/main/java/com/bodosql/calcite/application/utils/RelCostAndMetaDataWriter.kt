/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This file is modified for internal Bodo use. It's heavily modified from
 * the original RelWriterImpl and utilizes common code from there. As it
 * utilizes some of that code, this file is a derivative work and requires
 * the license header.
 */
package com.bodosql.calcite.application.utils

import com.bodosql.calcite.plan.Cost
import com.bodosql.calcite.rel.core.CachedSubPlanBase
import com.bodosql.calcite.rel.core.cachePlanContainers.CacheNodeSingleVisitHandler
import com.bodosql.calcite.rel.metadata.BodoRelMetadataQuery
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelVisitor
import org.apache.calcite.rel.externalize.RelWriterImpl
import org.apache.calcite.rex.RexNode
import org.apache.calcite.util.Pair
import java.io.PrintWriter
import java.text.DecimalFormat

/**
 * Writes relational plans in a textual format.
 *
 * This differs from the default RelWriterImpl because
 * it normalizes the RelNode ids and outputs the cost for
 * each relational operation.
 */
class RelCostAndMetaDataWriter(
    pw: PrintWriter,
    rel: RelNode,
    private val minRelID: Map<Int, Int>,
    private val showCosts: Boolean = true,
) : RelWriterImpl(pw) {
    // Information caching
    private val cacheQueue = CacheNodeSingleVisitHandler()

    // Use for normalizing RexNode values.
    private val rexBuilder = rel.cluster.rexBuilder

    private fun newSection(
        s: StringBuilder,
        sectionName: String,
    ) {
        spacer.spaces(s)
        s.append("# $sectionName: ")
    }

    override fun explain_(
        rel: RelNode,
        values: List<Pair<String, Any?>>,
    ) {
        // Add cached nodes to the queue for explains
        if (rel is CachedSubPlanBase) {
            cacheQueue.add(rel)
        }
        val inputs = rel.inputs
        val uncastedMq = rel.cluster.metadataQuery
        assert(uncastedMq is BodoRelMetadataQuery) {
            "Internal error in RelCostAndMetaDataWriter.explain_: metadataQuery should be of type BodoRelMetadataQuery"
        }
        val mq = uncastedMq as BodoRelMetadataQuery

        val s = StringBuilder()

        if (showCosts) {
            newSection(s, "operator id")
            s.append("${minRelID[rel.id]}\n")
            // Display cost information.
            val cost = mq.getNonCumulativeCost(rel) as Cost
            newSection(s, "cost")
            s.append("${cost.valueString} $cost\n")

            // Assign the physical trait information if that exists
            // for testing purposes
            val batchingProperty = rel.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)
            if (batchingProperty != null) {
                newSection(s, "batching property")
                s.append("$batchingProperty\n")
            }
            // Check if we have distinctiveness information for any of the columns
            // We don't do this in the case that we're determining what metadata is needed,
            // because the act of requesting the metadata "mq.getColumnDistinctCount" will make the system assume
            // that the metadata for this column is needed.
            if ((System.getenv("BODOSQL_TESTING_FIND_NEEDED_METADATA")?.toIntOrNull() ?: 0) == 0) {
                val hasDistinctInfo =
                    (0 until rel.rowType.fieldCount).any { i -> mq.getColumnDistinctCount(rel, i) != null }
                if (hasDistinctInfo) {
                    // If we do,
                    newSection(s, "distinct estimates")
                    for (columnIdx in 0 until rel.rowType.fieldCount) {
                        val distinctiveness = mq.getColumnDistinctCount(rel, columnIdx)
                        if (distinctiveness != null) {
                            val formattedDistinctiveness = DecimalFormat("##0.#E0").format(distinctiveness)
                            s.append("$$columnIdx = $formattedDistinctiveness values")
                            if (columnIdx < rel.rowType.fieldCount - 1) {
                                s.append(", ")
                            }
                        }
                    }
                    s.append("\n")
                }
            } else if ((System.getenv("UPDATE_EXPECT")?.toIntOrNull() ?: 0) != 0) {
                throw RuntimeException("One of UPDATE_EXPECT and BODOSQL_TESTING_FIND_NEEDED_METADATA must be disabled when running tests")
            }

            // Add the types of each output column
            newSection(s, "types")
            val types =
                rel.rowType.fieldList.map {
                    val outString = StringBuilder(it.type.toString())
                    if (!it.type.isNullable) {
                        outString.append(" NOT NULL")
                    }
                    outString
                }
            s.append(types.joinToString(", "))
            s.append("\n")
        }

        // Relational node name and attributes.
        spacer.spaces(s)
        s.append(rel.relTypeName)
        val attrs =
            values.asSequence()
                .filter { it.right !is RelNode }
                .joinToString(separator = ", ") { "${it.left}=[${normalize(it.right)}]" }
        if (attrs.isNotBlank()) {
            s.append("(")
                .append(attrs)
                .append(")")
        }

        pw.println(s)
        spacer.add(2)
        explainInputs(inputs)
        spacer.subtract(2)
    }

    private fun explainInputs(inputs: List<RelNode>) {
        inputs.forEach { input ->
            input.explain(this)
        }
    }

    private fun normalize(value: Any?): Any? =
        value?.let {
            when (it) {
                is RexNode -> normalizeRexNode(it)
                else -> it
            }
        }

    private fun normalizeRexNode(value: RexNode): RexNode = RexNormalizer.normalize(rexBuilder, value)

    fun explainCachedNodes() {
        while (cacheQueue.isNotEmpty()) {
            val rel = cacheQueue.pop()
            pw.println()
            pw.println("CACHED NODE ${rel.cacheID}")
            spacer.add(2)
            val node = rel.cachedPlan.plan.rel
            node.explain(this)
        }
    }

    companion object {
        /**
         * Counts the number of times a RelNode appears in the plan.
         */
        private fun countRelIds(root: RelNode): Map<Int, Int> {
            val visitor =
                object : RelVisitor() {
                    val nodeCount: MutableMap<Int, Int> = mutableMapOf()

                    override fun visit(
                        node: RelNode,
                        ordinal: Int,
                        parent: RelNode?,
                    ) {
                        nodeCount.merge(node.id, 1) { a, b -> a + b }
                        node.childrenAccept(this)
                    }
                }
            visitor.visit(root, 0, null)
            return visitor.nodeCount.toMap()
        }

        /**
         * This function takes a mapping of node ids to the number of times
         * they are present, as returned by countRelIds, and returns a map
         * of remappings for nodes with duplicate entries.
         */
        private fun normalizeDuplicates(ids: Map<Int, Int>): Map<Int, Int> =
            ids.asSequence()
                // Include only pairs where there were duplicates.
                .filter { (_, value) -> value >= 2 }
                // Only want the key.
                .map { (key, _) -> key }
                // Sort for consistency.
                .sorted()
                // Combine with a sequence that counts upwards from one.
                .zip((1..ids.size).asSequence())
                // Convert to a map.
                .toMap()
    }
}
