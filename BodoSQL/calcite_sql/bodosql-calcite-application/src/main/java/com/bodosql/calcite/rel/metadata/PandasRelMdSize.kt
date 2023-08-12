package com.bodosql.calcite.rel.metadata

import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMdSize
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.type.SqlTypeName
import kotlin.math.min

class PandasRelMdSize : RelMdSize() {
    /**
     * The default for this is to report that it doesn't know the size.
     *
     * Change the default to just return the average type value size
     * based on the column type.
     */
    override fun averageColumnSizes(rel: RelNode, mq: RelMetadataQuery): List<Double?> =
        rel.rowType.fieldList.map { rowType ->
            averageTypeValueSize(rowType.type)
        }

    override fun averageTypeValueSize(type: RelDataType): Double? =
        // Customize the behavior of VARCHAR and VARBINARY.
        // These two types will give negative values if their precisions
        // are negative so we want to override them to something that makes
        // sense in the case that they are negative.
        when (type.sqlTypeName) {
            SqlTypeName.VARCHAR -> {
                val precision = if (type.precision >= 0) type.precision else DEFAULT_VARCHAR_PRECISION
                // 100.0 is an arbitrary number taken from calcite's core code.
                min(precision.toDouble() * BYTES_PER_CHARACTER, 100.0)
            }

            SqlTypeName.VARBINARY -> {
                val precision = if (type.precision >= 0) type.precision else DEFAULT_VARBINARY_PRECISION
                // 100.0 is an arbitrary number taken from calcite's core code.
                min(precision.toDouble(), 100.0)
            }
            // Keep things sane when an unknown value is present.
            // This really shouldn't occur, but don't want to invalidate
            // an entire plan cost calculation based on an unknown field.
            SqlTypeName.UNKNOWN -> 8.0
            else -> super.averageTypeValueSize(type)
        }

    companion object {
        /**
         * Default precision for a VARCHAR that has no precision set
         * for the purposes of estimating sizes.
         *
         * Number is arbitrary. This seems like a suitable number for
         * something that will likely not be too long but also is still
         * significant for memory use.
         */
        private const val DEFAULT_VARCHAR_PRECISION: Int = 20

        /**
         * Default precision for a VARBINARY that has no precision set
         * for the purposes of estimating sizes.
         *
         * Number is arbitrary. This seems like a suitable number for
         * something that will likely not be too long but also is still
         * significant for memory use.
         */
        private const val DEFAULT_VARBINARY_PRECISION: Int = 20
    }
}
