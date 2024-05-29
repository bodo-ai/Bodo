package com.bodosql.calcite.application.testing

import com.bodosql.calcite.catalog.BodoGlueCatalog
import com.bodosql.calcite.catalog.BodoSQLCatalog
import com.bodosql.calcite.schema.BodoSqlSchema
import com.bodosql.calcite.schema.LocalSchema

/** Class for locally testing codegen using a AWS Glue Iceberg Catalog
 *  */
class GlueCatalogGenTest : GenTestFixture() {
    override fun isIceberg(): Boolean {
        return true
    }

    override fun supportsTimestampTZ(): Boolean {
        return false
    }

    override fun getCatalog(): BodoSQLCatalog {
        val warehouse = "s3://buckets/aneesh-iceberggluetest"
        return BodoGlueCatalog(warehouse)
    }

    override fun getSchema(): BodoSqlSchema {
        return LocalSchema("__BODOLOCAL__")
    }

    companion object {
        @Throws(Exception::class)
        @JvmStatic
        fun main(args: Array<String>) {
            val sql = "SELECT * FROM \"apple_data_two\".\"destfull\""
            val generateCode = true
            GlueCatalogGenTest().run(sql, generateCode)
        }
    }
}
