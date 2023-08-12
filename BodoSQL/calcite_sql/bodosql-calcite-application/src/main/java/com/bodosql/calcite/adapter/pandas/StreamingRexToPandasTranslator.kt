package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.application.PandasCodeGenVisitor
import com.bodosql.calcite.application.utils.IsScalar
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Module
import com.bodosql.calcite.ir.StateVariable
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeSystem
import org.apache.calcite.rex.RexCall

/**
 * A version of the RexToPandasTranslator that is used when the expression is occurring in a
 * streaming and non-scalar context
 */
class StreamingRexToPandasTranslator(
    visitor: PandasCodeGenVisitor,
    builder: Module.Builder,
    typeSystem: RelDataTypeSystem,
    nodeId: Int,
    input: BodoEngineTable,
    localRefs: List<Expr>,
    // State information for the streaming operator that uses
    // this translator. This is used for optimizations with
    // dictionary encoding.
    private var stateVar: StateVariable,
) : RexToPandasTranslator(visitor, builder, typeSystem, nodeId, input, localRefs) {

    /**
     * Generate the additional keyword arguments that are passed to functions that support
     * dictionary encoding caching in a streaming context. This is the state variable for the
     * context and a newly generated function id to unique identify the function.
     *
     * @return The list of pairs to generate dict_encoding_state=stateVar, func_id=NEW_FUNC_ID
     */
    private fun genDictEncodingArgs(): List<Pair<String, Expr>> {
        return listOf(
            Pair<String, Expr>("dict_encoding_state", stateVar),
            Pair<String, Expr>("func_id", builder.symbolTable.genFuncID()),
        )
    }

    override fun visitBinOpScan(operation: RexCall): Expr {
        return this.visitBinOpScan(operation, genDictEncodingArgs())
    }

    override fun visitLikeOp(node: RexCall): Expr {
        return visitLikeOp(node, genDictEncodingArgs())
    }

    override fun visitCastScan(operation: RexCall): Expr {
        return visitCastScan(operation, IsScalar.isScalar(operation), genDictEncodingArgs())
    }

    override fun visitTryCastScan(operation: RexCall): Expr {
        return visitTryCastScan(operation, genDictEncodingArgs())
    }

    override fun visitSubstringScan(node: RexCall): Expr {
        return visitSubstringScan(node, genDictEncodingArgs())
    }

    override fun visitNullIgnoringGenericFunc(fnOperation: RexCall, isSingleRow: Boolean): Expr {
        return visitNullIgnoringGenericFunc(fnOperation, isSingleRow, genDictEncodingArgs())
    }

    override fun visitDynamicCast(
        arg: Expr,
        inputType: RelDataType,
        outputType: RelDataType,
        isScalar: Boolean,
    ): Expr {
        return visitDynamicCast(arg, inputType, outputType, isScalar, genDictEncodingArgs())
    }

    override fun visitTrimFunc(fnName: String, stringToBeTrimmed: Expr, charactersToBeTrimmed: Expr): Expr {
        return visitTrimFunc(fnName, stringToBeTrimmed, charactersToBeTrimmed, genDictEncodingArgs())
    }

    override fun visitLeastGreatest(fnName: String, operands: List<Expr>): Expr {
        return visitLeastGreatest(fnName, operands, genDictEncodingArgs())
    }
    override fun visitNullIfFunc(operands: List<Expr>): Expr {
        return visitNullIfFunc(operands, genDictEncodingArgs())
    }

    override fun visitPosition(operands: List<Expr>): Expr {
        return visitPosition(operands, genDictEncodingArgs())
    }

    override fun visitCastFunc(fnOperation: RexCall, operands: List<Expr>): Expr {
        return visitCastFunc(fnOperation, operands, genDictEncodingArgs())
    }

    override fun visitStringFunc(fnOperation: RexCall, operands: List<Expr>): Expr {
        return visitStringFunc(fnOperation, operands, genDictEncodingArgs())
    }
}
