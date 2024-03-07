package com.bodosql.calcite.table

import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFactory
import org.apache.calcite.sql.type.BodoTZInfo
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.sql.type.VariantSqlType

/**
 * Data info class for BodoSQL columns. These are designed to support nested data.
 */
data class ColumnDataTypeInfo(
    val dataType: BodoSQLColumnDataType,
    val isNullable: Boolean,
    val precision: Int,
    val scale: Int,
    val tzInfo: BodoTZInfo?,
    val children: List<ColumnDataTypeInfo>,
    val fieldNames: List<String>,
) {
    // Constructor for most data types without precision or scale
    constructor(
        dataType: BodoSQLColumnDataType,
        isNullable: Boolean,
    ) : this(dataType, isNullable, RelDataType.PRECISION_NOT_SPECIFIED, RelDataType.SCALE_NOT_SPECIFIED, null, listOf(), listOf())

    // Constructor for String/Binary data types
    constructor(
        dataType: BodoSQLColumnDataType,
        isNullable: Boolean,
        precision: Int,
    ) : this(dataType, isNullable, precision, RelDataType.SCALE_NOT_SPECIFIED, null, listOf(), listOf())

    // Constructor for timestamp
    constructor(
        dataType: BodoSQLColumnDataType,
        isNullable: Boolean,
        precision: Int,
        tzInfo: BodoTZInfo?,
    ) : this(dataType, isNullable, precision, RelDataType.SCALE_NOT_SPECIFIED, tzInfo, listOf(), listOf())

    // Constructor for decimal
    constructor(
        dataType: BodoSQLColumnDataType,
        isNullable: Boolean,
        precision: Int,
        scale: Int,
    ) : this(dataType, isNullable, precision, scale, null, listOf(), listOf())

    // Constructor for Array and Categorical
    constructor(
        dataType: BodoSQLColumnDataType,
        isNullable: Boolean,
        child: ColumnDataTypeInfo,
    ) : this(dataType, isNullable, RelDataType.PRECISION_NOT_SPECIFIED, RelDataType.SCALE_NOT_SPECIFIED, null, listOf(child), listOf())

    // Constructor for Struct
    constructor(
        dataType: BodoSQLColumnDataType,
        isNullable: Boolean,
        fields: List<ColumnDataTypeInfo>,
        fieldNames: List<String>,
    ) : this(dataType, isNullable, RelDataType.PRECISION_NOT_SPECIFIED, RelDataType.SCALE_NOT_SPECIFIED, null, fields, fieldNames)

    // Constructor for Map
    constructor(
        dataType: BodoSQLColumnDataType,
        isNullable: Boolean,
        keyType: ColumnDataTypeInfo,
        valueType: ColumnDataTypeInfo,
    ) : this(
        dataType,
        isNullable,
        RelDataType.PRECISION_NOT_SPECIFIED,
        RelDataType.SCALE_NOT_SPECIFIED,
        null,
        listOf(keyType, valueType),
        listOf(),
    )

    fun convertToSqlType(typeFactory: RelDataTypeFactory): RelDataType {
        if (dataType == BodoSQLColumnDataType.CATEGORICAL) {
            val children: List<ColumnDataTypeInfo> = children
            if (children.size != 1) {
                throw BodoSQLCodegenException("Categorical must have exactly 1 child")
            }
            // Categorical code should be treated as its underlying elemType
            return children[0].convertToSqlType(typeFactory)
        }
        // Note: These exceptions are basically asserts.
        if (dataType == BodoSQLColumnDataType.ARRAY) {
            if (children.size != 1) {
                throw BodoSQLCodegenException("Array Column must have exactly 1 child")
            }
        } else if (dataType == BodoSQLColumnDataType.JSON_OBJECT) {
            if (children.size != 2) {
                throw BodoSQLCodegenException("Object Column must have exactly 2 children")
            }
        } else if (dataType == BodoSQLColumnDataType.STRUCT) {
            if (children.size != fieldNames.size) {
                throw BodoSQLCodegenException("Struct Column must have the same number of names and children")
            }
        } else if (children.isNotEmpty()) {
            throw BodoSQLCodegenException("Non-Nested Data Columns should not have any children")
        }
        // Recurse on the children if they exist.
        val mappedChildren = children.map { c -> c.convertToSqlType(typeFactory) }
        return dataType.convertToSqlType(typeFactory, isNullable, tzInfo, precision, scale, mappedChildren, fieldNames)
    }

    companion object {
        @JvmStatic
        fun fromSqlType(relDataType: RelDataType): ColumnDataTypeInfo {
            val isNullable = relDataType.isNullable
            val typeName = relDataType.sqlTypeName
            if (typeName == SqlTypeName.ARRAY) {
                val child = fromSqlType(relDataType.componentType!!)
                return ColumnDataTypeInfo(BodoSQLColumnDataType.ARRAY, isNullable, child)
            } else if (typeName == SqlTypeName.MAP) {
                val key = fromSqlType(relDataType.keyType!!)
                val value = fromSqlType(relDataType.valueType!!)
                return ColumnDataTypeInfo(BodoSQLColumnDataType.JSON_OBJECT, isNullable, key, value)
            } else if (relDataType is VariantSqlType) {
                return ColumnDataTypeInfo(BodoSQLColumnDataType.VARIANT, isNullable)
            } else {
                return when (typeName) {
                    SqlTypeName.TINYINT -> ColumnDataTypeInfo(BodoSQLColumnDataType.INT8, isNullable)
                    SqlTypeName.SMALLINT -> ColumnDataTypeInfo(BodoSQLColumnDataType.INT16, isNullable)
                    SqlTypeName.INTEGER -> ColumnDataTypeInfo(BodoSQLColumnDataType.INT32, isNullable)
                    SqlTypeName.BIGINT -> ColumnDataTypeInfo(BodoSQLColumnDataType.INT64, isNullable)
                    SqlTypeName.FLOAT -> ColumnDataTypeInfo(BodoSQLColumnDataType.FLOAT32, isNullable)
                    // TODO: FIX DECIMAL
                    SqlTypeName.REAL, SqlTypeName.DOUBLE, SqlTypeName.DECIMAL ->
                        ColumnDataTypeInfo(
                            BodoSQLColumnDataType.FLOAT64,
                            isNullable,
                        )

                    SqlTypeName.DATE -> ColumnDataTypeInfo(BodoSQLColumnDataType.DATE, isNullable)
                    SqlTypeName.CHAR, SqlTypeName.VARCHAR ->
                        ColumnDataTypeInfo(
                            BodoSQLColumnDataType.STRING,
                            isNullable,
                            relDataType.precision,
                        )

                    SqlTypeName.TIMESTAMP ->
                        ColumnDataTypeInfo(
                            BodoSQLColumnDataType.DATETIME,
                            isNullable,
                            relDataType.precision,
                        )

                    SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE ->
                        ColumnDataTypeInfo(
                            BodoSQLColumnDataType.TIMESTAMP_LTZ,
                            isNullable,
                            relDataType.precision,
                        )

                    SqlTypeName.TIMESTAMP_TZ ->
                        ColumnDataTypeInfo(
                            BodoSQLColumnDataType.TIMESTAMP_TZ,
                            isNullable,
                            relDataType.precision,
                        )

                    SqlTypeName.TIME ->
                        ColumnDataTypeInfo(
                            BodoSQLColumnDataType.TIME,
                            isNullable,
                            relDataType.precision,
                        )

                    SqlTypeName.BOOLEAN -> ColumnDataTypeInfo(BodoSQLColumnDataType.BOOL8, isNullable)
                    SqlTypeName.INTERVAL_DAY_HOUR, SqlTypeName.INTERVAL_DAY_MINUTE, SqlTypeName.INTERVAL_DAY_SECOND,
                    SqlTypeName.INTERVAL_HOUR_MINUTE, SqlTypeName.INTERVAL_HOUR_SECOND, SqlTypeName.INTERVAL_MINUTE_SECOND,
                    SqlTypeName.INTERVAL_HOUR, SqlTypeName.INTERVAL_MINUTE, SqlTypeName.INTERVAL_SECOND, SqlTypeName.INTERVAL_DAY,
                    SqlTypeName.INTERVAL_YEAR, SqlTypeName.INTERVAL_MONTH, SqlTypeName.INTERVAL_YEAR_MONTH,
                    ->
                        ColumnDataTypeInfo(
                            BodoSQLColumnDataType.TIMEDELTA,
                            isNullable,
                        )

                    else -> throw RuntimeException(
                        "Internal Error: Calcite Plan Produced an Unsupported relDataType" +
                            "for table extension Type",
                    )
                }
            }
        }
    }
}
