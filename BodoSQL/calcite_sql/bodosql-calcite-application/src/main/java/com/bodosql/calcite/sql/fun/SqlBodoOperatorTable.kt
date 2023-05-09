package com.bodosql.calcite.sql.`fun`

import org.apache.calcite.sql.SqlKind

/**
 * This is a singleton object that holds the custom operators we've defined
 * outside of Calcite.
 */
object SqlBodoOperatorTable {
    @JvmField
    val ANY_LIKE: SqlLikeQuantifyOperator =
        SqlLikeQuantifyOperator("LIKE ANY", SqlKind.LIKE, SqlKind.SOME, caseSensitive = true)

    @JvmField
    val ALL_LIKE: SqlLikeQuantifyOperator =
        SqlLikeQuantifyOperator("LIKE ALL", SqlKind.LIKE, SqlKind.ALL, caseSensitive = true)

    @JvmField
    val ANY_ILIKE: SqlLikeQuantifyOperator =
        SqlLikeQuantifyOperator("ILIKE ANY", SqlKind.LIKE, SqlKind.SOME, caseSensitive = false)

    @JvmField
    val ALL_ILIKE: SqlLikeQuantifyOperator =
        SqlLikeQuantifyOperator("ILIKE ALL", SqlKind.LIKE, SqlKind.ALL, caseSensitive = false)
}
