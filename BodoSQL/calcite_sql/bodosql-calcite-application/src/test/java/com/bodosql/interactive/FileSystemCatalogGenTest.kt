package com.bodosql.calcite.interactive

import com.bodosql.calcite.application.write.WriteTarget
import com.bodosql.calcite.catalog.BodoSQLCatalog
import com.bodosql.calcite.catalog.FileSystemCatalog
import com.bodosql.calcite.schema.BodoSqlSchema
import com.bodosql.calcite.schema.LocalSchema

/** Class for locally testing codegen using a REST Iceberg catalog  */
class FileSystemCatalogGenTest : GenTestFixture() {
    override fun isIceberg(): Boolean = true

    override fun supportsTimestampTZ(): Boolean = false

    override fun getCatalog(): BodoSQLCatalog {
        val bucket = System.getenv("GCS_BUCKET")
        val con = "gs://$bucket/iceberg_db"
        val wd = WriteTarget.WriteTargetEnum.fromString("iceberg")
        val cat = FileSystemCatalog(con, wd, "MY_NAMESPACE")
        return cat
    }

    override fun getSchema(): BodoSqlSchema = LocalSchema("MY_NAMESPACE")

    companion object {
        @Throws(Exception::class)
        @JvmStatic
        fun main(args: Array<String>) {
            val sql = "SELECT * FROM MY_NAMESPACE.MY_TABLE"
            val generateCode = true
            FileSystemCatalogGenTest().run(sql, generateCode)
        }
    }
}
