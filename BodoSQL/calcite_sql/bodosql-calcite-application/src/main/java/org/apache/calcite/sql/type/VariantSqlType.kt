package org.apache.calcite.sql.type

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
        sb.append("VARIANT()")
    }
}
