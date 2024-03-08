package com.bodosql.calcite.application.operatorTables

import org.apache.calcite.plan.Strong
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFieldImpl
import org.apache.calcite.rel.type.RelRecordType
import org.apache.calcite.sql.SqlFunctionCategory
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlOperatorBinding
import org.apache.calcite.sql.SqlSyntax
import org.apache.calcite.sql.type.ExplicitOperandTypeChecker
import org.apache.calcite.sql.type.OperandHandlers
import org.apache.calcite.sql.type.ReturnTypes
import org.apache.calcite.sql.validate.SqlMonotonicity

/**
 * SQL Operator for a Snowflake User Defined Function (UDF)
 * that will not be inlined and instead must be processed as a Bodo Kernel.
 */
class SnowflakeNativeUDF private constructor(
    val body: String,
    val language: String,
    argTypes: List<RelDataType>,
    returnType: RelDataType,
) : SqlNullPolicyFunction(
        "SNOWFLAKE_NATIVE_UDF",
        SqlKind.OTHER_FUNCTION,
        SqlSyntax.FUNCTION,
        true,
        ReturnTypes.explicit(returnType),
        null,
        OperandHandlers.DEFAULT,
        ExplicitOperandTypeChecker(argTypesToRecord(argTypes)),
        0,
        SqlFunctionCategory.USER_DEFINED_FUNCTION,
        { _: SqlOperatorBinding -> SqlMonotonicity.NOT_MONOTONIC },
        Strong.Policy.AS_IS,
    ) {
    companion object {
        @JvmStatic
        fun create(
            body: String,
            language: String,
            argTypes: List<RelDataType>,
            returnType: RelDataType,
        ): SnowflakeNativeUDF {
            return SnowflakeNativeUDF(body, language, argTypes, returnType)
        }

        /**
         * Wrap a list of RelDataTypes in a RelRecordType
         * so we can use the ExplicitOperandTypeChecker.
         */
        @JvmStatic
        private fun argTypesToRecord(argTypes: List<RelDataType>): RelRecordType {
            // Generate dummy Field types
            val fields = argTypes.withIndex().map { RelDataTypeFieldImpl("", it.index, it.value) }
            return RelRecordType(fields)
        }
    }
}
