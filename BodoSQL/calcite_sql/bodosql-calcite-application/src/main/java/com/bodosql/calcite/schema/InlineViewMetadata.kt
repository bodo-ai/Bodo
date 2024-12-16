package com.bodosql.calcite.schema

class InlineViewMetadata(
    val unsafeToInline: Boolean,
    val isMaterialized: Boolean,
    val viewDefinition: String,
)
