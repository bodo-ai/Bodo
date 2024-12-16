package com.bodosql.calcite.sql.func

import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeField
import org.apache.calcite.sql.SqlCall
import org.apache.calcite.sql.SqlCallBinding
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.SqlOperandCountRange
import org.apache.calcite.sql.SqlSpecialOperator
import org.apache.calcite.sql.SqlWriter
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.sql.parser.SqlParserUtil
import org.apache.calcite.sql.type.InferTypes
import org.apache.calcite.sql.type.OperandTypes
import org.apache.calcite.sql.type.ReturnTypes
import org.apache.calcite.sql.type.SqlOperandCountRanges
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.sql.validate.SqlValidator
import org.apache.calcite.sql.validate.SqlValidatorScope
import org.apache.calcite.util.Static.RESOURCE

class SqlLikeQuantifyOperator(
    name: String,
    kind: SqlKind,
    val comparisonKind: SqlKind,
    val caseSensitive: Boolean,
) : SqlSpecialOperator(
        name,
        kind,
        32,
        // LIKE is right-associative, because that makes it easier to capture
        // dangling ESCAPE clauses: "a like b like c escape d" becomes
        // "a like (b like c escape d)".
        false,
        ReturnTypes.BOOLEAN_NULLABLE,
        InferTypes.FIRST_KNOWN,
        null,
    ) {
    init {
        assert(kind == SqlKind.LIKE)
    }

    override fun getOperandCountRange(): SqlOperandCountRange = SqlOperandCountRanges.between(2, 3)

    override fun deriveType(
        validator: SqlValidator,
        scope: SqlValidatorScope,
        call: SqlCall,
    ): RelDataType {
        val operands = call.operandList
        assert(operands.size in 2..3)
        val left = operands[0]
        val right = operands[1]

        val typeFactory = validator.typeFactory
        val leftType = validator.deriveType(scope, left)
        val rightType =
            if (right is SqlNodeList) {
                val rightTypeList =
                    right.map { node ->
                        validator.deriveType(scope, node)
                    }
                var rightType = typeFactory.leastRestrictive(rightTypeList)

                // First check that the expressions in the IN list are compatible
                // with each other. Same rules as the VALUES operator (per
                // SQL:2003 Part 2 Section 8.4, <in predicate>).
                if (rightType == null && validator.config().typeCoercionEnabled()) {
                    // Do implicit type cast if it is allowed to.
                    rightType = validator.typeCoercion.getWiderTypeFor(rightTypeList, true)
                }
                if (rightType == null) {
                    throw validator.newValidationError(right, RESOURCE.incompatibleTypesInList())
                }
                validator.setValidatedNodeType(right, rightType)
                rightType
            } else {
                // Handle the 'LIKE ANY (query)' form.
                // We don't strictly support this in code generation, but
                // snowflake seems to allow it for query parsing.
                validator.deriveType(scope, right)
            }

        val callBinding = SqlCallBinding(validator, scope, call)

        val checker =
            if (operands.size == 3) {
                OperandTypes.STRING_SAME_SAME_SAME
            } else {
                OperandTypes.STRING_SAME_SAME
            }
        if (!checker.checkOperandTypes(callBinding, false)) {
            throw validator.newValidationError(call, RESOURCE.incompatibleValueType(call.operator.name))
        }

        // Result is a boolean, nullable if there are any nullable types
        // on either side.
        return typeFactory.createTypeWithNullability(
            typeFactory.createSqlType(SqlTypeName.BOOLEAN),
            leftType.isNullable || rightType.isNullable,
        )
    }

    override fun unparse(
        writer: SqlWriter,
        call: SqlCall,
        leftPrec: Int,
        rightPrec: Int,
    ) {
        val frame = writer.startList("", "")
        call.operand<SqlNode>(0).unparse(writer, getLeftPrec(), getRightPrec())
        writer.sep(name)

        call.operand<SqlNode>(1).unparse(writer, getLeftPrec(), getRightPrec())
        if (call.operandCount() == 3) {
            writer.sep("ESCAPE")
            call.operand<SqlNode>(2).unparse(writer, getLeftPrec(), getRightPrec())
        }
        writer.endList(frame)
    }

    /**
     * This method is copied from the SqlLikeOperator.
     * See that method in Calcite for details.
     */
    override fun reduceExpr(
        opOrdinal: Int,
        list: TokenSequence,
    ): ReduceResult {
        // Example:
        //   a LIKE b || c ESCAPE d || e AND f
        // |  |    |      |      |      |
        //  exp0    exp1          exp2
        val exp0 = list.node(opOrdinal - 1)
        val op = list.op(opOrdinal)
        assert(op is SqlLikeQuantifyOperator)
        val exp1 =
            SqlParserUtil.toTreeEx(
                list,
                opOrdinal + 1,
                rightPrec,
                SqlKind.ESCAPE,
            )
        var exp2: SqlNode? = null
        if (opOrdinal + 2 < list.size()) {
            if (list.isOp(opOrdinal + 2)) {
                val op2 = list.op(opOrdinal + 2)
                if (op2.getKind() == SqlKind.ESCAPE) {
                    exp2 =
                        SqlParserUtil.toTreeEx(
                            list,
                            opOrdinal + 3,
                            rightPrec,
                            SqlKind.ESCAPE,
                        )
                }
            }
        }
        val operands: Array<SqlNode>
        val end: Int
        if (exp2 != null) {
            operands = arrayOf(exp0, exp1, exp2)
            end = opOrdinal + 4
        } else {
            operands = arrayOf(exp0, exp1)
            end = opOrdinal + 2
        }
        val call = createCall(SqlParserPos.sum(operands), *operands)
        return ReduceResult(opOrdinal - 1, end, call)
    }

    companion object {
        fun anyNullable(fieldList: List<RelDataTypeField>): Boolean =
            fieldList.any { field ->
                field.type.isNullable
            }
    }
}
