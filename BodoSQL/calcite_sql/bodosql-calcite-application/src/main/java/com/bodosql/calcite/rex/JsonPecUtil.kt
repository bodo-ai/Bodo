package com.bodosql.calcite.rex

import com.bodosql.calcite.application.operatorTables.ArrayOperatorTable
import com.bodosql.calcite.application.operatorTables.CastingOperatorTable
import com.bodosql.calcite.application.operatorTables.JsonOperatorTable
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.type.SqlTypeName

class JsonPecUtil {
    companion object {
        /**
         * Entrypoint to detect if a node matches the ParseExtractCast pattern
         * (PEC for short).
         * See <a
         * href="https://bodo.atlassian.net/wiki/spaces/B/pages/1530494991/Support+PARSE+JSON+via+Parse+Extract+Cast+Rewrite">this
         * confluence doc</a> for more details.
         *
         * @param node The node that is being checked to see if it matches
         * the pattern.
         */
        @JvmStatic
        public fun isPec(node: RexNode) = isPecCast(node, false)

        /**
         * Returns whether a function call is a casting function besides
         * regular CAST or TRY_CAST.
         */
        private fun isCastFunc(node: RexCall): Boolean {
            return when (node.operator.name) {
                ArrayOperatorTable.TO_ARRAY.name,
                CastingOperatorTable.TO_VARIANT.name,
                CastingOperatorTable.TO_OBJECT.name,
                CastingOperatorTable.TO_NUMBER.name,
                CastingOperatorTable.TRY_TO_NUMBER.name,
                CastingOperatorTable.TO_DOUBLE.name,
                CastingOperatorTable.TRY_TO_DOUBLE.name,
                CastingOperatorTable.TO_TIMESTAMP.name,
                CastingOperatorTable.TO_TIMESTAMP_NTZ.name,
                CastingOperatorTable.TO_TIMESTAMP_LTZ.name,
                CastingOperatorTable.TO_TIMESTAMP_TZ.name,
                CastingOperatorTable.TRY_TO_TIMESTAMP.name,
                CastingOperatorTable.TRY_TO_TIMESTAMP_NTZ.name,
                CastingOperatorTable.TRY_TO_TIMESTAMP_LTZ.name,
                CastingOperatorTable.TRY_TO_TIMESTAMP_TZ.name,
                CastingOperatorTable.TO_DATE.name,
                CastingOperatorTable.TRY_TO_DATE.name,
                CastingOperatorTable.TO_TIME.name,
                CastingOperatorTable.TO_BOOLEAN.name,
                CastingOperatorTable.TRY_TO_BOOLEAN.name,
                CastingOperatorTable.TRY_TO_TIME.name,
                CastingOperatorTable.TO_VARCHAR.name,
                CastingOperatorTable.TO_BINARY.name,
                CastingOperatorTable.TRY_TO_BINARY.name,
                /** Not including the following because they are aliases
                 * that should be removed by the convertlet table.
                 */
                // CastingOperatorTable.DATE.name,
                // CastingOperatorTable.TO_CHAR.name,
                // CastingOperatorTable.TIME.name,
                // CastingOperatorTable.TO_NUMERIC.name,
                // CastingOperatorTable.TO_DECIMAL.name,
                // CastingOperatorTable.TRY_TO_NUMERIC.name,
                // CastingOperatorTable.TRY_TO_DECIMAL.name,
                -> true
                else -> false
            }
        }

        /**
         * Returns whether a type is a semi-structured type.
         */
        private fun isSemiStructuredType(type: RelDataType): Boolean {
            return when (type.sqlTypeName) {
                SqlTypeName.ARRAY,
                SqlTypeName.MAP,
                SqlTypeName.OTHER,
                -> true
                else -> false
            }
        }

        /**
         * Returns whether a node matches the CAST portion of the PeC
         * pattern.
         *
         * See is_pec_cast from the confluence doc for more clarification.
         *
         * @param node The node being checked.
         * @param semiStructured True if the node is expected to be
         * semi-structured, false if it is expected to be non-semi-structured.
         */
        private fun isPecCast(node: RexNode, semiStructured: Boolean): Boolean {
            return if (node is RexCall && (node.kind == SqlKind.CAST || node.kind == SqlKind.SAFE_CAST || isCastFunc(node))) {
                (semiStructured == isSemiStructuredType(node.type)) && (node.operands[0] is RexCall) && isPecExtract(node.operands[0] as RexCall)
            } else { false }
        }

        /**
         * Returns whether a node matches the EXTRACT portion of the PeC
         * pattern. Currently only allows extractions with constant
         * indices/fields.
         *
         * See is_pec_extract from the confluence doc for more clarification.
         *
         * @param node The node being checked.
         */
        private fun isPecExtract(node: RexCall): Boolean {
            return when (node.operator.name) {
                JsonOperatorTable.PARSE_JSON.name -> true
                SqlStdOperatorTable.ITEM.name,
                JsonOperatorTable.GET_PATH.name,
                ArrayOperatorTable.ARRAY_MAP_GET.name,
                -> (node.operands[0] is RexCall && isPecExtract(node.operands[0] as RexCall)) || isPecCast(node.operands[0], true)
                else -> false
            }
        }

        /**
         * Rewrites a PEC node into a JSON_EXTRACT_PATH_TEXT call followed by a cast.
         * See <a
         * href="https://bodo.atlassian.net/wiki/spaces/B/pages/1530494991/Support+PARSE+JSON+via+Parse+Extract+Cast+Rewrite">this
         * confluence doc</a> for more details.
         *
         * @param castCall The original outermost cast call.
         * @param node The node that is being rewritten. Assumes that
         * isPec has already been called on node and returned true.
         * @param builder A RexBuilder used to construct function calls.
         */
        @JvmStatic
        public fun rewritePec(node: RexCall, builder: RexBuilder): RexNode {
            val (stringToParse, pathToExtract) = rewritePecHelper(node, builder, null)
            val parseExtract = builder.makeCall(JsonOperatorTable.JSON_EXTRACT_PATH_TEXT, stringToParse, pathToExtract)
            val newOperands: MutableList<RexNode> = mutableListOf()
            newOperands.add(parseExtract)
            newOperands.addAll(node.operands.subList(1, node.operands.size))
            return node.clone(node.type, newOperands)
        }

        /**
         * Helper for rewritePec that recursively unravels the cast/extract nodes until it reaches
         * the PARSE_JSON call, extracting the string that it parses and also building up the
         * extraction string along the way.
         *
         * For example, consider this PeC node: `TO_ARRAY(PARSE_JSON(S):foo.bar)[0]::integer`.
         *
         * `rewritePecHelper` will return a pair containing `S` and `['foo']['bar'][0]`.
         *
         * @param node The current level of node being unraveled.
         * @param builder A RexBuilder used to construct function calls.
         * @param pathSoFar Rex node representing the path string based on the EXTRACT
         * components processed so far. Any new extract terms are added to the left.
         * Starts out as null before terms are added to it.
         * requires adding a dot before it.
         */
        private fun rewritePecHelper(node: RexCall, builder: RexBuilder, pathSoFar: RexNode?): Pair<RexNode, RexNode> {
            return when (node.operator.name) {
                // Base case: when we reach the PARSE_JSON call, its input
                // is the string that is to be parsed.
                JsonOperatorTable.PARSE_JSON.name -> {
                    if (pathSoFar != null) {
                        Pair(node.operands[0], pathSoFar)
                    } else {
                        throw Exception("Malformed PEC: expected at least 1 EXTRACT step")
                    }
                }
                // Extract case: for each get/get_path call, recursively build
                // up the extraction string.
                SqlStdOperatorTable.ITEM.name,
                JsonOperatorTable.GET_PATH.name,
                ArrayOperatorTable.ARRAY_MAP_GET.name,
                -> {
                    val newPath = prependToExtractPath(pathSoFar, builder, node.operands[1])
                    rewritePecHelper(node.operands[0] as RexCall, builder, newPath)
                }
                // Otherwise: the node is a casting function that can be ignored
                // so we skip to its input.
                else -> rewritePecHelper(node.operands[0] as RexCall, builder, pathSoFar)
            }
        }

        /**
         * Helper function for rewritePecHelper that prepends a new extraction component
         * to the beginning of the component so far. If the term so far is null, just
         * uses the new term. Otherwise adds it to the front.
         *
         * @param pathSoFar Rex node representing the path string based on the EXTRACT
         * components processed so far. Any new extract terms are added to the left.
         * Starts out as null before terms are added to it.
         * @param builder A RexBuilder used to construct function calls.
         * @param extractComponent The integer/string expression (possibly constant)
         * that is the next extract term to add.
         */
        private fun prependToExtractPath(pathSoFar: RexNode?, builder: RexBuilder, extractComponent: RexNode): RexNode {
            val leftBracket = builder.makeLiteral("[")
            val rightBracket = builder.makeLiteral("]")
            val quoteChar = builder.makeLiteral("'")
            val isString = listOf(SqlTypeName.CHAR, SqlTypeName.VARCHAR).contains(extractComponent.type.sqlTypeName)
            val wrappedComponent = if (isString) {
                // If the component is a string, ensure it is wrapped in quotes
                builder.makeCall(StringOperatorTable.CONCAT, quoteChar, extractComponent, quoteChar)
            } else {
                // If the component is an integer, cast it to a string so it can be part
                // of the path string.
                builder.makeCast(builder.typeFactory.createSqlType(SqlTypeName.VARCHAR), extractComponent)
            }
            // Wrap the component in brackets
            val componentInBrackets = builder.makeCall(StringOperatorTable.CONCAT, leftBracket, wrappedComponent, rightBracket)
            // If this is the first iteration, no concatenation is required. Otherwise,
            // concat this the possibly wrapped term to the existing path, possibly
            // with a dot in between based on the old value of requiresDot.
            return if (pathSoFar == null) { componentInBrackets } else { builder.makeCall(StringOperatorTable.CONCAT, componentInBrackets, pathSoFar) }
        }
    }
}
