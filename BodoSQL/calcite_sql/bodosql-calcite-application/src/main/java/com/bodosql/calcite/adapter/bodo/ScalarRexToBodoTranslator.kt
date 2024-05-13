package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.application.BodoCodeGenVisitor
import com.bodosql.calcite.ir.Module
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeSystem
import org.apache.calcite.rex.RexNode

/**
 * Class for converting RexNodes to Bodo expressions
 * when we know there are only scalar expressions that cannot
 * reference an input column (e.g. no case expressions
 * or length dependency).
 */
class ScalarRexToBodoTranslator(
    visitor: BodoCodeGenVisitor,
    builder: Module.Builder,
    typeSystem: RelDataTypeSystem,
    dynamicParamTypes: List<RelDataType>,
    namedParamTypeMap: Map<String, RelDataType>,
) : RexToBodoTranslator(visitor, builder, typeSystem, null, dynamicParamTypes, namedParamTypeMap) {
    override fun isOperandScalar(operand: RexNode): Boolean {
        return true
    }
}
