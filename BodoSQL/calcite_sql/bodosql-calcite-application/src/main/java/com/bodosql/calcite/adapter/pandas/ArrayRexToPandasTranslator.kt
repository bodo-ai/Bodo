package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.PandasCodeGenVisitor
import com.bodosql.calcite.application.utils.BodoArrayHelpers
import com.bodosql.calcite.application.utils.IsScalar
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Module
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeSystem
import org.apache.calcite.rex.RexNode

class ArrayRexToPandasTranslator(
    visitor: PandasCodeGenVisitor,
    builder: Module.Builder,
    typeSystem: RelDataTypeSystem,
    nodeId: Int,
    input: BodoEngineTable,
    dynamicParamTypes: List<RelDataType>,
    namedParamTypeMap: Map<String, RelDataType>,
    localRefs: List<Expr>,
) : RexToPandasTranslator(visitor, builder, typeSystem, nodeId, input, dynamicParamTypes, namedParamTypeMap, localRefs) {
    constructor(
        visitor: PandasCodeGenVisitor,
        builder: Module.Builder,
        typeSystem: RelDataTypeSystem,
        nodeId: Int,
        input: BodoEngineTable,
        dynamicParamTypes: List<RelDataType>,
        namedParamTypeMap: Map<String, RelDataType>,
    ) :
        this(visitor, builder, typeSystem, nodeId, input, dynamicParamTypes, namedParamTypeMap, listOf())

    /**
     * Call node.accept on the given RexNode. Is the RexNode is a scalar then it wraps
     * the result in a Scalar -> Array conversion.
     */
    fun apply(node: RexNode): Expr {
        val result = node.accept(this)
        return if (IsScalar.isScalar(node)) {
            scalarToArray(result, node.type)
        } else {
            result
        }
    }

    /**
     * Convert a scalar output to an array.
     */
    private fun scalarToArray(
        scalar: Expr,
        dataType: RelDataType,
    ): Expr.Call {
        val global = visitor.lowerAsGlobal(BodoArrayHelpers.sqlTypeToBodoArrayType(dataType, true, visitor.genDefaultTZ().zoneExpr))
        return Expr.Call(
            "bodo.utils.conversion.coerce_scalar_to_array",
            scalar,
            Expr.Call("len", getInput()),
            global,
        )
    }
}
