package com.bodosql.calcite.interactive

import com.bodosql.calcite.application.write.WriteTarget
import com.bodosql.calcite.catalog.BodoSQLCatalog
import com.bodosql.calcite.catalog.FileSystemCatalog
import com.bodosql.calcite.schema.BodoSqlSchema
import com.bodosql.calcite.schema.LocalSchema

/** Class for locally testing codegen using a REST Iceberg catalog  */
class FileCatalogGenTest : GenTestFixture() {
    override fun isIceberg(): Boolean = true

    override fun supportsTimestampTZ(): Boolean = false

    override fun getCatalog(): BodoSQLCatalog {
        val warehouse = "/Users/scottroutledge/Bodo/iceberg_db"
        val write = WriteTarget.WriteTargetEnum.fromString("iceberg")
        val cat = FileSystemCatalog(warehouse, write, ".")
        return cat
    }

    override fun getSchema(): BodoSqlSchema = LocalSchema(".")

    companion object {
        @Throws(Exception::class)
        @JvmStatic
        fun main(args: Array<String>) {
            val sql = "SELECT 1"
            val generateCode = true
            FileCatalogGenTest().run(sql, generateCode)
        }
    }
}
