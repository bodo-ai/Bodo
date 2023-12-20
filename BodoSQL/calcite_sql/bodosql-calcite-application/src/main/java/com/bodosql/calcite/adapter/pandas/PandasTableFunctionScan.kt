package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.application.operatorTables.TableFunctionOperatorTable
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.TableFunctionScanBase
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelColumnMapping
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexLiteral
import java.lang.reflect.Type
import java.math.BigDecimal

class PandasTableFunctionScan(cluster: RelOptCluster, traits: RelTraitSet, inputs: List<RelNode>, call: RexCall, elementType: Type?, rowType: RelDataType, columnMappings: Set<RelColumnMapping>?) : TableFunctionScanBase(cluster, traits.replace(PandasRel.CONVENTION), inputs, call, elementType, rowType, columnMappings), PandasRel {
    /**
     * Emits the code necessary for implementing this relational operator.
     *
     * @param implementor implementation handler.
     * @return the variable that represents this relational expression.
     */
    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
        return implementor::build {
                ctx ->
            if ((call as RexCall).operator.name == TableFunctionOperatorTable.GENERATOR.name) {
                emitGenerator(ctx, call as RexCall)
            } else {
                throw BodoSQLCodegenException("Flatten node does not currently support codegen for operation $call")
            }
        }
    }

    /**
     * Emits the code necessary to calculate a call to the GENERATOR function.
     *
     * @param ctx the build context
     * @param flattenCall the function call to FLATTEN
     * @return the variable that represents this relational expression.
     */
    fun emitGenerator(ctx: PandasRel.BuildContext, generatorCall: RexCall): BodoEngineTable {
        generatorCall.operands[0]
        val rowCountLiteral = generatorCall.operands[0] as RexLiteral
        val rowCountExpr = Expr.IntegerLiteral(rowCountLiteral.getValueAs(BigDecimal::class.java)!!.toInt())
        return ctx.returns(Expr.Call("bodo.hiframes.table.generate_empty_table_with_rows", listOf(rowCountExpr), listOf()))
    }

    /**
     * Function to create the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    /**
     * Function to delete the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
        TODO("Not yet implemented")
    }
    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            inputs: List<RelNode>,
            call: RexCall,
            rowType: RelDataType,
        ): PandasTableFunctionScan {
            return PandasTableFunctionScan(cluster, cluster.traitSet(), inputs, call, null, rowType, null)
        }
    }
}
