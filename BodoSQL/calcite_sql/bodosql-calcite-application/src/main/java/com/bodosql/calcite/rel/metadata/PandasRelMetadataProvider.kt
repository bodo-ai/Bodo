package com.bodosql.calcite.rel.metadata

import org.apache.calcite.rel.metadata.*

class PandasRelMetadataProvider(ranks: Int) : RelMetadataProvider by
ChainedRelMetadataProvider.of(
    listOf(
        ReflectiveRelMetadataProvider.reflectiveSource(
            PandasRelMdRowCount(),
            BuiltInMetadata.RowCount.Handler::class.java
        ),
        // Inject information about the number of ranks
        // for Pandas queries as the parallelism attribute.
        ReflectiveRelMetadataProvider.reflectiveSource(
            // Used to include the number of ranks for planner costs.
            // Not really used yet but planned to be used for computation
            // costs of nodes with highly parallel operations.
            PandasRelMdParallelism(ranks),
            BuiltInMetadata.Parallelism.Handler::class.java
        ),
        ReflectiveRelMetadataProvider.reflectiveSource(
            PandasRelMdSize(),
            BuiltInMetadata.Size.Handler::class.java
        ),
        DefaultRelMetadataProvider.INSTANCE,
    )
) {
    /**
     * Default constructor for this metadata provider.
     *
     * This constructor is temporary until we have a way to
     * inject the number of ranks into the PandasRelMetadataProvider
     * from the Python code.
     *
     * The number of ranks isn't meaningfully used by the planner yet,
     * but placing it at 2 in case it starts to be used that way
     * and to signify that Pandas operations are intended to be
     * parallelized.
     */
    constructor() : this(ranks = 2)
}
