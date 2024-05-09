package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.TableCreateBase
import com.bodosql.calcite.schema.CatalogSchema
import com.bodosql.calcite.sql.ddl.SnowflakeCreateTableMetadata
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty.Companion.tableCreateProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.schema.Schema
import org.apache.calcite.sql.ddl.SqlCreateTable

class BodoPhysicalTableCreate private constructor(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    private val schema: CatalogSchema,
    tableName: String,
    isReplace: Boolean,
    createTableType: SqlCreateTable.CreateTableType,
    path: List<String>,
    meta: SnowflakeCreateTableMetadata,
) : TableCreateBase(
        cluster,
        traitSet.replace(
            BodoPhysicalRel.CONVENTION,
        ),
        input, schema, tableName, isReplace, createTableType, path, meta,
    ),
    BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): BodoPhysicalTableCreate {
        return BodoPhysicalTableCreate(
            cluster,
            traitSet,
            sole(inputs),
            schema,
            tableName,
            isReplace,
            createTableType,
            path,
            meta,
        )
    }

    // Update getSchema() to always indicate we have a CatalogSchema
    override fun getSchema(): CatalogSchema {
        return schema
    }

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    override fun getTimerType() = SingleBatchRelNodeTimer.OperationType.IO_BATCH

    override fun operationDescriptor() = "writing table"

    override fun loggingTitle() = "IO TIMING"

    override fun nodeDetails() = tableName

    override fun isStreaming() = (input as BodoPhysicalRel).batchingProperty() == BatchingProperty.STREAMING

    override fun expectedInputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        // Note: Types may be lazily computed so use getRowType() instead of rowType
        return tableCreateProperty(input.getRowType())
    }

    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        TODO("Not yet implemented")
    }

    companion object {
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            input: RelNode,
            schema: Schema,
            tableName: String,
            isReplace: Boolean,
            createTableType: SqlCreateTable.CreateTableType,
            path: List<String>,
            meta: SnowflakeCreateTableMetadata,
        ): BodoPhysicalTableCreate {
            if (schema !is CatalogSchema) {
                throw BodoSQLCodegenException("BodoSQL only supports create table with Catalog Schemas.")
            }
            return BodoPhysicalTableCreate(cluster, traitSet, input, schema, tableName, isReplace, createTableType, path, meta)
        }
    }
}
