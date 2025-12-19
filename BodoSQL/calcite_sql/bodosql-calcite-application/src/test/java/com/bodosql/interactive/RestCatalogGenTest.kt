package com.bodosql.calcite.interactive

import com.bodosql.calcite.catalog.BodoSQLCatalog
import com.bodosql.calcite.catalog.IcebergRESTCatalog
import com.bodosql.calcite.schema.BodoSqlSchema
import com.bodosql.calcite.schema.LocalSchema

/** Class for locally testing codegen using a REST Iceberg catalog  */
class RestCatalogGenTest : GenTestFixture() {
    override fun isIceberg(): Boolean = true

    override fun supportsTimestampTZ(): Boolean = false

    override fun getCatalog(): BodoSQLCatalog {
        val warehouse = "aws-polaris-warehouse"
        val cat = IcebergRESTCatalog("http://localhost:8181/api/catalog", warehouse, null, "root:s3cr3t", "default", "PRINCIPAL_ROLE:ALL")
        return cat
    }

    override fun getSchema(): BodoSqlSchema = LocalSchema("__BODOLOCAL__")

    companion object {
        @Throws(Exception::class)
        @JvmStatic
        fun main(args: Array<String>) {
            val sql = "SELECT A, B, C FROM CI.BODOSQL_ICEBERG_READ_TEST"
            val generateCode = true
            RestCatalogGenTest().run(sql, generateCode)
        }
    }
}
