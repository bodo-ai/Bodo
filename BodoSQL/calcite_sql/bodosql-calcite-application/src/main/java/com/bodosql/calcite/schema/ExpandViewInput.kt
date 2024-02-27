package com.bodosql.calcite.schema

import org.apache.calcite.rel.type.RelDataType

/**
 * An input class for expanding views that is used as a place-holder
 * for caching.
 */
data class ExpandViewInput(
    val outputType: RelDataType,
    val viewDefinition: String,
    val defaultPath: List<String>,
    val viewPath: List<String>,
)
