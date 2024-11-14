package com.bodosql.calcite.rel.metadata

import org.apache.calcite.rel.metadata.RelMetadataQuery
import java.util.function.Supplier

class BodoRelMetadataQuerySupplier(
    private val query: RelMetadataQuery,
) : Supplier<RelMetadataQuery> {
    override fun get(): RelMetadataQuery = query
}
