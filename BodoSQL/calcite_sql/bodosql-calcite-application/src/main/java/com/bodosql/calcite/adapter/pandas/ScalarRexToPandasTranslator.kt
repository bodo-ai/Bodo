package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.PandasCodeGenVisitor
import com.bodosql.calcite.ir.Module
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeSystem
import org.apache.calcite.rex.RexNode

/**
 * Class for converting RexNodes to Pandas expressions
 * when we know there are only scalar expressions that cannot
 * reference an input column (e.g. no case expressions
 * or length dependency).
 */
class ScalarRexToPandasTranslator(
    visitor: PandasCodeGenVisitor,
    builder: Module.Builder,
    typeSystem: RelDataTypeSystem,
    nodeId: Int,
    dynamicParamTypes: List<RelDataType>,
) : RexToPandasTranslator(visitor, builder, typeSystem, nodeId, null, dynamicParamTypes) {
    override fun isOperandScalar(operand: RexNode): Boolean {
        return true
    }
}
