package com.bodosql.calcite.sql.type

import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFactory
import org.apache.calcite.sql.SqlCallBinding
import org.apache.calcite.sql.SqlOperandCountRange
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.type.CompositeOperandTypeChecker
import org.apache.calcite.sql.type.SqlOperandTypeChecker
import org.apache.calcite.sql.type.SqlOperandTypeInference
import java.util.function.BiFunction

/**
 * Implementation of CustomConsistencySqlOperandTypeChecker that works as a wrapper around
 * a final SqlOperandTypeChecker and delegates the custom consistency logic to a lambda function.
 * Any functionality which is not "compatible" with delegation (e.g. OR) should not be implemented.
 */
class DelegatingCustomConsistencySqlOperandTypeChecker(
    private val base: SqlOperandTypeChecker,
    private val deriveConsistencyOperandTypesImpl: (List<RelDataType>, RelDataTypeFactory) -> List<RelDataType>,
) : CustomConsistencySqlOperandTypeChecker {
    // ~~~~Delegated implementations~~~~
    override fun checkOperandTypes(
        callBinding: SqlCallBinding?,
        throwOnFailure: Boolean,
    ): Boolean = base.checkOperandTypes(callBinding, throwOnFailure)

    override fun getOperandCountRange(): SqlOperandCountRange = base.operandCountRange

    override fun getAllowedSignatures(
        op: SqlOperator?,
        opName: String?,
    ): String = base.getAllowedSignatures(op, opName)

    override fun isOptional(i: Int): Boolean = base.isOptional(i)

    override fun isFixedParameters(): Boolean = base.isFixedParameters

    override fun typeInference(): SqlOperandTypeInference? = base.typeInference()

    // ~~~~Disabled implementations~~~~
    override fun withGenerator(signatureGenerator: BiFunction<SqlOperator?, String?, String?>?): CompositeOperandTypeChecker? =
        throw UnsupportedOperationException("withGenerator not supported for DelegatingCustomConsistencySqlOperandTypeChecker")

    override fun and(checker: SqlOperandTypeChecker?): SqlOperandTypeChecker? =
        throw UnsupportedOperationException("and not supported for DelegatingCustomConsistencySqlOperandTypeChecker")

    override fun or(checker: SqlOperandTypeChecker?): SqlOperandTypeChecker? =
        throw UnsupportedOperationException("or not supported for DelegatingCustomConsistencySqlOperandTypeChecker")

    // ~Custom consistency logic

    /**
     * Derive the destination type for each input type based on custom input consistency logic.
     * Each implementing class should either provide an implementation to this function or a mechanism
     * to delegate this to a custom lambda function.
     */
    override fun deriveConsistencyOperandTypes(
        inputTypes: List<RelDataType>,
        typeFactory: RelDataTypeFactory,
    ): List<RelDataType> = deriveConsistencyOperandTypesImpl(inputTypes, typeFactory)
}
