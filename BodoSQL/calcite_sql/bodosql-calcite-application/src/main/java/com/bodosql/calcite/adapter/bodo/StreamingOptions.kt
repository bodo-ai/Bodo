package com.bodosql.calcite.adapter.bodo

data class StreamingOptions(
    val chunkSize: Int,
    val prefetchSFIceberg: Boolean,
)
