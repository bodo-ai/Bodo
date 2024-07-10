package com.bodosql.calcite.rel.core

import com.bodosql.calcite.application.write.WriteTarget.IfExistsBehavior
import com.bodosql.calcite.sql.ddl.CreateTableMetadata
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.core.TableCreate
import org.apache.calcite.schema.Schema
import org.apache.calcite.sql.ddl.SqlCreateTable.CreateTableType

open class TableCreateBase(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    private val schema: Schema,
    val tableName: String,
    val isReplace: Boolean,
    val createTableType: CreateTableType,
    val path: List<String>,
    val meta: CreateTableMetadata,
) : TableCreate(cluster, traitSet, input) {
    override fun explainTerms(pw: RelWriter): RelWriter {
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
        if (meta.tableProperties != null) {
            result =
                result.item(
                    "Table Properties", meta.tableProperties!!,
                )
        }
        return result
    }

    open fun getSchema(): Schema {
        return schema
    }

    fun getSchemaPath(): List<String> {
        return path
    }

    fun getIfExistsBehavior(): IfExistsBehavior {
        return if (isReplace) {
            IfExistsBehavior.REPLACE
        } else {
            IfExistsBehavior.FAIL
        }
    }
}
