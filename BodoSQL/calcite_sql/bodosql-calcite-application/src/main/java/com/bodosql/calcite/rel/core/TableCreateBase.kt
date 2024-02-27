package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.sql.ddl.SnowflakeCreateTableMetadata
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.core.TableCreate
import org.apache.calcite.schema.Schema
import org.apache.calcite.sql.ddl.SqlCreateTable
import org.apache.calcite.sql.ddl.SqlCreateTable.CreateTableType

open class TableCreateBase(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    val schema: Schema,
    val tableName: String,
    val isReplace: Boolean,
    val createTableType: SqlCreateTable.CreateTableType,
    val path: List<String>,
    val meta: SnowflakeCreateTableMetadata,
) : TableCreate(cluster, traitSet, input) {
    override fun explainTerms(pw: RelWriter?): RelWriter? {
        var result =
            super.explainTerms(pw)
                .item("TableName", tableName)
                .item("Target Schema", this.path)
                .item("IsReplace", isReplace)
                .item("CreateTableType", createTableType)
        if (meta.tableComment != null) {
            result = result.item("Table Comment", meta.tableComment)
        }
        if (meta.columnComments != null) {
            result =
                result.item(
                    "Column Comments",
                    meta.columnComments!!.mapIndexed {
                            idx,
                            value,
                        ->
                        value?.let { Pair(this.getRowType().fieldNames[idx], value) }
                    }.filterNotNull(),
                )
        }
        return result
    }

    open fun getSchemaPath(): List<String?>? {
        return path
    }
}
