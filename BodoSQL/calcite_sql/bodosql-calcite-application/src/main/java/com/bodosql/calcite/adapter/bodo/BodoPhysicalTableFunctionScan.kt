package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.application.operatorTables.TableFunctionOperatorTable
import com.bodosql.calcite.application.operatorTables.TableFunctionOperatorTable.EXTERNAL_TABLE_FILES_NAME
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.TableFunctionScanBase
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelColumnMapping
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.sql.SnowflakeNamedArgumentSqlCatalogTableFunction
import org.apache.calcite.sql.validate.SqlUserDefinedTableFunction
import java.lang.reflect.Type
import java.math.BigDecimal

class BodoPhysicalTableFunctionScan(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    inputs: List<RelNode>,
    call: RexCall,
    elementType: Type?,
    rowType: RelDataType,
    columnMappings: Set<RelColumnMapping>?,
) :
    TableFunctionScanBase(
            cluster,
            traits.replace(BodoPhysicalRel.CONVENTION),
            inputs,
            call,
            elementType,
            rowType,
            columnMappings,
        ),
        BodoPhysicalRel {
    /**
     * Emits the code necessary for implementing this relational operator.
     *
     * @param implementor implementation handler.
     * @return the variable that represents this relational expression.
     */
    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        return implementor::build {
                ctx ->
            if ((call as RexCall).operator.name == TableFunctionOperatorTable.GENERATOR.name) {
                emitGenerator(ctx, call as RexCall)
            } else if ((call as RexCall).operator.name == EXTERNAL_TABLE_FILES_NAME) {
                emitExternalTableFiles(ctx, call as RexCall)
            } else {
                throw BodoSQLCodegenException("Flatten node does not currently support codegen for operation $call")
            }
        }
    }

    /**
     * Emits the code necessary to calculate a call to the GENERATOR function.
     *
     * @param ctx the build context
     * @param generatorCall the function call to GENERATOR
     * @return the variable that represents this relational expression.
     */
    private fun emitGenerator(
        ctx: BodoPhysicalRel.BuildContext,
        generatorCall: RexCall,
    ): BodoEngineTable {
        generatorCall.operands[0]
        val rowCountLiteral = generatorCall.operands[0] as RexLiteral
        val rowCountExpr = Expr.IntegerLiteral(rowCountLiteral.getValueAs(BigDecimal::class.java)!!.toInt())
        return ctx.returns(Expr.Call("bodo.hiframes.table.generate_empty_table_with_rows", listOf(rowCountExpr), listOf()))
    }

    /**
     * Emits the code necessary to calculate a call to the EXTERNAL_TABLE_FILES function.
     *
     * @param ctx the build context
     * @param etfCall the function call to EXTERNAL_TABLE_FILES
     * @return the variable that represents this relational expression.
     */
    private fun emitExternalTableFiles(
        ctx: BodoPhysicalRel.BuildContext,
        etfCall: RexCall,
    ): BodoEngineTable {
        // This is a defensive check since it should always be true by the manner in which
        // a call to EXTERNAL_TABLE_FILES is constructed.
        return if (etfCall.op is SqlUserDefinedTableFunction &&
            etfCall.op.function is
                SnowflakeNamedArgumentSqlCatalogTableFunction
        ) {
            val function = etfCall.op.function as SnowflakeNamedArgumentSqlCatalogTableFunction
            val catalog = function.catalog
            val databaseName = if (function.functionPath.size < 2) catalog.getDefaultSchema(0)[0] else function.functionPath[0]
            val connStr = catalog.generatePythonConnStr(ImmutableList.of(databaseName, ""))
            val tableName = (etfCall.operands[0] as RexLiteral).getValueAs(String::class.java)!!
            val query =
                Expr.StringLiteral(
                    "SELECT * FROM TABLE(\"$databaseName\".\"INFORMATION_SCHEMA\".\"EXTERNAL_TABLE_FILES\"(TABLE_NAME=>$$$tableName$$))",
                )
            val args = listOf(query, connStr)
            val kwargs = listOf("_bodo_read_as_table" to Expr.BooleanLiteral(true))
            ctx.returns(Expr.Call("pd.read_sql", args, kwargs))
        } else {
            throw BodoSQLCodegenException("Cannot call EXTERNAL_TABLE_FILES without an associated SnowflakeCatalog")
        }
    }

    /**
     * Function to create the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    /**
     * Function to delete the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        TODO("Not yet implemented")
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            inputs: List<RelNode>,
            call: RexCall,
            rowType: RelDataType,
        ): BodoPhysicalTableFunctionScan {
            return BodoPhysicalTableFunctionScan(cluster, cluster.traitSet(), inputs, call, null, rowType, null)
        }
    }
}
