package com.bodosql.calcite.adapter.pandas

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

class PandasTableCreate private constructor(
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
            PandasRel.CONVENTION,
        ),
        input, schema, tableName, isReplace, createTableType, path, meta,
    ),
    PandasRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): PandasTableCreate {
        return PandasTableCreate(
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

    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    override fun getTimerType() = SingleBatchRelNodeTimer.OperationType.IO_BATCH

    override fun operationDescriptor() = "writing table"

    override fun loggingTitle() = "IO TIMING"

    override fun nodeDetails() = tableName

    override fun isStreaming() = (input as PandasRel).batchingProperty() == BatchingProperty.STREAMING

    override fun expectedInputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        // Note: Types may be lazily computed so use getRowType() instead of rowType
        return tableCreateProperty(input.getRowType())
    }

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(
        ctx: PandasRel.BuildContext,
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
        ): PandasTableCreate {
            if (schema !is CatalogSchema) {
                throw BodoSQLCodegenException("BodoSQL only supports create table with Catalog Schemas.")
            }
            return PandasTableCreate(cluster, traitSet, input, schema, tableName, isReplace, createTableType, path, meta)
        }
    }
}
