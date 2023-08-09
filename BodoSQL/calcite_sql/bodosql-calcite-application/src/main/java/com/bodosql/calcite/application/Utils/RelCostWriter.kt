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
package com.bodosql.calcite.application.Utils

import com.bodosql.calcite.plan.Cost
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelVisitor
import org.apache.calcite.rel.externalize.RelWriterImpl
import org.apache.calcite.rex.RexNode
import org.apache.calcite.util.Pair
import java.io.PrintWriter

/**
 * Writes relational plans in a textual format.
 *
 * This differs from the default RelWriterImpl because
 * it normalizes the RelNode ids and outputs the cost for
 * each relational operation.
 */
class RelCostWriter(pw: PrintWriter, rel: RelNode) : RelWriterImpl(pw) {

    // Holds a mapping of RelNode ids to their rewritten normalized
    // versions.
    private val normalizedIdMap: Map<Int, Int> = normalizeDuplicates(countRelIds(rel))

    // Used to normalize RexNode values.
    private val normalizer: com.bodosql.calcite.application.Utils.RexNormalizer =
        com.bodosql.calcite.application.Utils.RexNormalizer(rel.cluster.rexBuilder)

    override fun explain_(rel: RelNode, values: List<Pair<String, Any?>>) {
        val inputs = rel.inputs
        val mq = rel.cluster.metadataQuery

        val s = StringBuilder()

        // Display cost information.
        spacer.spaces(s)
        val cost = mq.getNonCumulativeCost(rel) as Cost
        s.append("# cost = ${cost.valueString} $cost")


        // Assign the physical trait information if that exists
        // for testing purposes
        val batchingProperty = rel.traitSet.getTrait(BatchingPropertyTraitDef.INSTANCE)
        if (batchingProperty != null) {
            s.append(", batchingProperty=[$batchingProperty]")
        }


        // If this relational node appears multiple times in the tree,
        // we will have assigned it a normalized id. If that normalized id
        // exists, output it.
        // We only output normalized ids when there are duplicates to reduce
        // churn on the diffs when plans change. Ids are really only significant
        // when they are duplicates.
        normalizedIdMap[rel.id]?.also { id ->
            s.append(", id = $id")
        }
        s.append("\n")

        // Relational node name and attributes.
        spacer.spaces(s)
        s.append(rel.relTypeName)
        val attrs = values.asSequence()
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

    private fun normalize(value: Any?): Any? = value?.let {
        when (it) {
            is RexNode -> normalizeRexNode(it)
            else -> it
        }
    }

    private fun normalizeRexNode(value: RexNode): RexNode =
        value.accept(normalizer)

    companion object {
        /**
         * Counts the number of times a RelNode appears in the plan.
         */
        private fun countRelIds(root: RelNode): Map<Int, Int> {
            val visitor = object : RelVisitor() {
                val nodeCount: MutableMap<Int, Int> = mutableMapOf()

                override fun visit(node: RelNode, ordinal: Int, parent: RelNode?) {
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
