package com.bodosql.calcite.catalog

class TabularCatalog(warehouse: String, token: String? = null, credential: String? = null) : IcebergRESTCatalog(
    "https://api.tabular.io/ws",
    warehouse,
    token,
    credential,
    defaultSchema = "default",
) {
    init {
        if (token == null && credential == null) {
            throw IllegalArgumentException("Either token or credential must be provided.")
        }
    }

    /**
     * Return the db location to which this Catalog refers.
     *
     * @return The source DB location.
     */
    override fun getDBType(): String {
        return "TABULAR"
    }
}
