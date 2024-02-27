package com.bodosql.calcite.application.utils

import com.bodosql.calcite.table.BodoSqlTable
import org.apache.calcite.prepare.RelOptTableImpl
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelVisitor
import org.apache.calcite.rel.core.TableScan
import org.apache.calcite.util.Util

/**
 * Helper functions related to checking table permissions in a RelNode tree.
 */
class CheckTablePermissions {
    companion object {
        /**
         * Check that we can read every table in a tree of RelNodes.
         * Intended to return False if any table cannot be read.
         */
        @JvmStatic
        fun canRead(rel: RelNode): Boolean {
            return try {
                object : RelVisitor() {
                    override fun visit(
                        node: RelNode,
                        ordinal: Int,
                        parent: RelNode?,
                    ) {
                        if (node is TableScan) {
                            val table = node.table as RelOptTableImpl
                            val bodoSQLTable = table.table() as BodoSqlTable
                            if (!bodoSQLTable.canRead()) {
                                throw Util.FoundOne.NULL
                            }
                        }
                        super.visit(node, ordinal, parent)
                    }
                }.go(rel)
                true
            } catch (e: Util.FoundOne) {
                false
            }
        }
    }
}
