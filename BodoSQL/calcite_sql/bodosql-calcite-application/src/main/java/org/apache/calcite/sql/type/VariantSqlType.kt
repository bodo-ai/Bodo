package org.apache.calcite.sql.type

import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeComparability
import org.apache.calcite.rel.type.RelDataTypeFamily

/**
 * Implementation of the Snowflake Variant type.
 * This type can be converted to by any type but
 * needs to be explicitly tracked by the type system.
 */
class VariantSqlType(nullable: Boolean) : AbstractSqlType(SqlTypeName.OTHER, nullable, null) {

    init {
        computeDigest()
    }

    /**
     * Generates a string representation of this type.
     *
     * @param sb         StringBuilder into which to generate the string
     * @param withDetail when true, all detail information needed to compute a
     * unique digest (and return from getFullTypeString) should
     * be included;
     */
    override fun generateTypeString(sb: StringBuilder, withDetail: Boolean) {
        // Note: We don't use withDetail because there are no details to add
        // yet. If we represent this with a "real runtime type" then we will
        // likely use this then.
        sb.append("VARIANT")
    }

    /**
     * Indicate that the variant type can be used in all comparison
     * operations.
     */
    override fun getComparability(): RelDataTypeComparability {
        return RelDataTypeComparability.ALL
    }

    override fun getFamily(): RelDataTypeFamily {
        return typeFamily
    }

    // If we extract the key or value type (map operations)
    // the result is defined to be variant
    override fun getValueType(): RelDataType? {
        return this
    }

    override fun getKeyType(): RelDataType? {
        return this
    }

    /** Define a variant family so all Variant types even if
     * they differ in nullability are considered comparable. */
    private class VariantDataTypeFamily : RelDataTypeFamily {

        override fun equals(other: Any?): Boolean {
            return other != null && other is VariantDataTypeFamily
        }
    }
    companion object {
        @JvmStatic
        private val typeFamily: VariantDataTypeFamily = VariantDataTypeFamily()
    }
}
