package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.application.BodoCodeGenVisitor
import com.bodosql.calcite.application.utils.BodoArrayHelpers
import com.bodosql.calcite.application.utils.IsScalar
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Module
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeSystem
import org.apache.calcite.rex.RexNode

class ArrayRexToBodoTranslator(
    visitor: BodoCodeGenVisitor,
    builder: Module.Builder,
    typeSystem: RelDataTypeSystem,
    input: BodoEngineTable,
    dynamicParamTypes: List<RelDataType>,
    namedParamTypeMap: Map<String, RelDataType>,
    localRefs: List<Expr>,
) : RexToBodoTranslator(visitor, builder, typeSystem, input, dynamicParamTypes, namedParamTypeMap, localRefs) {
    constructor(
        visitor: BodoCodeGenVisitor,
        builder: Module.Builder,
        typeSystem: RelDataTypeSystem,
        input: BodoEngineTable,
        dynamicParamTypes: List<RelDataType>,
        namedParamTypeMap: Map<String, RelDataType>,
    ) :
        this(visitor, builder, typeSystem, input, dynamicParamTypes, namedParamTypeMap, listOf())

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
