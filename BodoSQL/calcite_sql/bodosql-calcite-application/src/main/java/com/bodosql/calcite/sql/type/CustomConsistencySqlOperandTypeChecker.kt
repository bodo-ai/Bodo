package com.bodosql.calcite.sql.type

import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFactory
import org.apache.calcite.sql.type.SqlOperandTypeChecker

/**
 * Extension of SqlOperandTypeChecker that marks the consistency of operand types
 * as custom and provides an interface for deriving a unique type for each input.
 */
interface CustomConsistencySqlOperandTypeChecker : SqlOperandTypeChecker {
    /**
     * Derive the destination type for each input type based on custom input consistency logic.
     * Each implementing class should either provide an implementation to this function or a mechanism
     * to delegate this to a custom lambda function.
     */
    fun deriveConsistencyOperandTypes(
        inputTypes: List<RelDataType>,
        typeFactory: RelDataTypeFactory,
    ): List<RelDataType>

    override fun getConsistency(): SqlOperandTypeChecker.Consistency = SqlOperandTypeChecker.Consistency.CUSTOM
}
