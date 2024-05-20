package com.bodosql.calcite.catalog

import com.bodosql.calcite.ir.Expr
import com.google.common.collect.ImmutableList
import org.apache.iceberg.CatalogProperties

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

    /**
     * Generate a TabularConnectionType that can generate a
     * Python connection string used to read from or write to a Catalog in Bodo's SQL
     * Python code.
     *
     *
     * TODO(jsternberg): This method is needed for the XXXToPandasConverter nodes, but exposing
     * this is a bad idea and this class likely needs to be refactored in a way that the connection
     * information can be passed around more easily.
     *
     * @param schemaPath The schema component to define the connection not including the table name.
     * @return An Expr representing the connection string
     */
    override fun generatePythonConnStr(schemaPath: ImmutableList<String>): Expr {
        val props = getIcebergConnection().properties()
        val warehouse = props[CatalogProperties.WAREHOUSE_LOCATION]!!
        val uri = props[CatalogProperties.URI]!!.replace(Regex("https?://"), "")
        return Expr.Call("bodosql.get_tabular_connection", Expr.StringLiteral(uri), Expr.StringLiteral(warehouse))
    }
}
