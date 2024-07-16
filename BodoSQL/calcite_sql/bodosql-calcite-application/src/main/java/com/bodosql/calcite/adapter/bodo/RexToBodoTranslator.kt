package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.application.BodoCodeGenVisitor
import com.bodosql.calcite.application.BodoSQLCodeGen.BinOpCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.CondOpCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.ConversionCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.DateAddCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.DateDiffCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.ExtractCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.JsonCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.LiteralCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.PostfixOpCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.PrefixOpCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.RegexpCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.SinceEpochFnCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen
import com.bodosql.calcite.application.BodoSQLCodeGen.TrigCodeGen
import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem
import com.bodosql.calcite.application.operatorTables.SnowflakeNativeUDF
import com.bodosql.calcite.application.utils.BodoArrayHelpers
import com.bodosql.calcite.application.utils.BodoCtx
import com.bodosql.calcite.application.utils.IsScalar
import com.bodosql.calcite.application.utils.Utils
import com.bodosql.calcite.application.utils.Utils.getConversionName
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Expr.FrameTripleQuotedString
import com.bodosql.calcite.ir.Expr.GetItem
import com.bodosql.calcite.ir.Expr.TripleQuotedString
import com.bodosql.calcite.ir.Frame
import com.bodosql.calcite.ir.Module
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.Op.Assign
import com.bodosql.calcite.ir.Op.Continue
import com.bodosql.calcite.ir.Op.SetItem
import com.bodosql.calcite.ir.Op.Stmt
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.ir.bodoSQLKernel
import com.bodosql.calcite.rex.RexNamedParam
import com.google.common.collect.ImmutableList
import com.google.common.collect.Sets
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeSystem
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexCorrelVariable
import org.apache.calcite.rex.RexDynamicParam
import org.apache.calcite.rex.RexFieldAccess
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexLocalRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.rex.RexPatternFieldRef
import org.apache.calcite.rex.RexRangeRef
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.rex.RexSlot
import org.apache.calcite.rex.RexSubQuery
import org.apache.calcite.rex.RexTableInputRef
import org.apache.calcite.rex.RexVisitor
import org.apache.calcite.sql.SqlBinaryOperator
import org.apache.calcite.sql.SqlFunction
import org.apache.calcite.sql.SqlInternalOperator
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlNullTreatmentOperator
import org.apache.calcite.sql.SqlPostfixOperator
import org.apache.calcite.sql.SqlPrefixOperator
import org.apache.calcite.sql.SqlSpecialOperator
import org.apache.calcite.sql.`fun`.SqlCaseOperator
import org.apache.calcite.sql.`fun`.SqlCastFunction
import org.apache.calcite.sql.`fun`.SqlDatetimePlusOperator
import org.apache.calcite.sql.`fun`.SqlDatetimeSubtractionOperator
import org.apache.calcite.sql.`fun`.SqlExtractFunction
import org.apache.calcite.sql.`fun`.SqlLikeOperator
import org.apache.calcite.sql.`fun`.SqlSubstringFunction
import org.apache.calcite.sql.type.BodoSqlTypeUtil
import org.apache.calcite.sql.type.BodoTZInfo
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.sql.type.SqlTypeName
import java.math.BigDecimal
import java.util.Locale

/**
 * Base class for RexToBodoTranslator that contains
 * most of the functionality but doesn't require
 * an input, so it can be reused for translators
 * that don't support columns
 */
open class RexToBodoTranslator(
    // Don't really want this here, but it's easier than trying to move all
    // of its functionality into the builder immediately.
    protected val visitor: BodoCodeGenVisitor,
    protected val builder: Module.Builder,
    protected val typeSystem: RelDataTypeSystem,
    private val input: BodoEngineTable?,
    private val dynamicParamTypes: List<RelDataType>,
    private val namedParamTypeMap: Map<String, RelDataType>,
    private val localRefs: List<Expr>,
) :
    RexVisitor<Expr> {
    constructor(
        visitor: BodoCodeGenVisitor,
        builder: Module.Builder,
        typeSystem: RelDataTypeSystem,
        input: BodoEngineTable?,
        dynamicParamTypes: List<RelDataType>,
        namedParamTypeMap: Map<String, RelDataType>,
    ) : this(visitor, builder, typeSystem, input, dynamicParamTypes, namedParamTypeMap, listOf())

    private val ctx: BodoCtx = BodoCtx()
    private var weekStart: Int? = null
    private var weekOfYearPolicy: Int? = null
    private var currentDatabase: String? = null
    private var currentAccount: String? = null

    init {
        if (typeSystem is BodoSQLRelDataTypeSystem) {
            weekStart = typeSystem.weekStart
            weekOfYearPolicy = typeSystem.weekOfYearPolicy
            currentDatabase = typeSystem.catalogContext?.currentDatabase
            currentAccount = typeSystem.catalogContext?.currentAccount
        } else {
            weekStart = 0
            weekOfYearPolicy = 0
            currentDatabase = null
            currentAccount = null
        }
    }

    fun getInput(): BodoEngineTable {
        return input ?: throw BodoSQLCodegenException("Illegal use of Input in a context that doesn't support table references.")
    }

    override fun visitInputRef(inputRef: RexInputRef): Expr {
        return Expr.Call(
            "bodo.hiframes.table.get_table_data",
            listOf(getInput(), Expr.IntegerLiteral(inputRef.index)),
        )
    }

    override fun visitLocalRef(localRef: RexLocalRef): Expr {
        return localRefs[localRef.index]
    }

    override fun visitLiteral(literal: RexLiteral): Expr {
        return LiteralCodeGen.generateLiteralCode(literal, visitor)
    }

    override fun visitCall(call: RexCall): Expr {
        // TODO(jsternberg): Using instanceof here is problematic.
        // It would be better to use getKind(). Revisit this later.
        if (call.operator is SqlNullTreatmentOperator) {
            return visitNullTreatmentOp(call)
        } else if (call.operator is SqlBinaryOperator ||
            call.operator is SqlDatetimePlusOperator ||
            call.operator is SqlDatetimeSubtractionOperator
        ) {
            return visitBinOpScan(call)
        } else if (call.operator is SqlPostfixOperator) {
            return visitPostfixOpScan(call)
        } else if (call.operator is SqlPrefixOperator) {
            return visitPrefixOpScan(call)
        } else if (call.operator is SqlInternalOperator) {
            return visitInternalOp(call)
        } else if (call.operator is SqlLikeOperator) {
            return visitLikeOp(call)
        } else if (call.operator is SqlCaseOperator) {
            return visitCaseOp(call)
        } else if (call.operator is SqlCastFunction) {
            return visitCastScan(call, call.kind == SqlKind.SAFE_CAST)
        } else if (call.operator is SqlExtractFunction) {
            return visitExtractScan(call)
        } else if (call.operator is SqlSubstringFunction) {
            return visitSubstringScan(call)
        } else if (call.operator is SqlFunction) {
            return visitGenericFuncOp(call)
        } else {
            return if (call.operator is SqlSpecialOperator) {
                visitSpecialOp(call)
            } else {
                throw BodoSQLCodegenException(
                    "Internal Error: Calcite Plan Produced an Unsupported RexCall:" + call.operator,
                )
            }
        }
    }

    private fun visitSpecialOp(node: RexCall): Expr {
        val operands = visitList(node.operands)
        when (node.kind) {
            SqlKind.ITEM -> {
                assert(operands.size == 2)
                return JsonCodeGen.visitGetOp(
                    isOperandScalar(node.operands[0]),
                    isOperandScalar(node.operands[1]),
                    operands,
                )
            }
            SqlKind.ROW -> {
                val keys = node.getType().fieldNames.map { Expr.StringLiteral(it) }
                val scalars = node.operands.map { Expr.BooleanLiteral(isOperandScalar(it)) }
                return JsonCodeGen.getObjectConstructKeepNullCode("OBJECT_CONSTRUCT_KEEP_NULL", keys, operands, scalars, visitor)
            }
            else -> throw BodoSQLCodegenException(
                "Internal Error: Calcite Plan Produced an Unsupported special operand call: " +
                    node.operator,
            )
        }
    }

    /**
     * Visitor for RexCalls IGNORE NULLS and RESPECT NULLS This function is only called if IGNORE
     * NULLS and RESPECT NULLS is called without an associated window. Otherwise, it is included as a
     * field in the REX OVER node.
     *
     *
     * Currently, we always throw an error when entering this call. Frankly, based on my reading of
     * calcite's syntax, we only reach this node through invalid syntax in Calcite (LEAD/LAG
     * RESPECT/IGNORE NULL's without a window)
     *
     * @param node RexCall being visited
     * @return Expr containing the new column name and the code generated for the relational
     * expression.
     */
    private fun visitNullTreatmentOp(node: RexCall): Expr {
        when (val innerCallKind = node.getOperands()[0].kind) {
            SqlKind.LEAD, SqlKind.LAG, SqlKind.NTH_VALUE, SqlKind.FIRST_VALUE, SqlKind.LAST_VALUE -> throw BodoSQLCodegenException(
                "Error during codegen: $innerCallKind requires OVER clause.",
            )

            else -> throw BodoSQLCodegenException(
                (
                    "Error during codegen: Unreachable code entered while evaluating the following rex" +
                        " node in visitNullTreatmentOp: " +
                        node
                ),
            )
        }
    }

    protected open fun visitBinOpScan(operation: RexCall): Expr {
        return this.visitBinOpScan(operation, listOf())
    }

    /**
     * @param operand
     * @return True if the operand is a scalar
     */
    protected open fun isOperandScalar(operand: RexNode): Boolean {
        return IsScalar.isScalar(operand)
    }

    /**
     * Generate the code for a Binary operation.
     *
     * @param operation The operation from which to generate the expression.
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitBinOpScan(
        operation: RexCall,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        val args: MutableList<Expr> = ArrayList()
        val binOp = operation.operator
        // Store the argument types for TZ-Aware data
        val argDataTypes: MutableList<RelDataType> = ArrayList()
        // Store whether the arguments were scalars vs columns
        val argScalars: MutableList<Boolean> = ArrayList()
        for (operand: RexNode in operation.operands) {
            val exprCode = operand.accept(this)
            args.add(exprCode)
            argDataTypes.add(operand.type)
            argScalars.add(isOperandScalar(operand))
        }
        return if (binOp.getKind() == SqlKind.OTHER && (binOp.name == "||")) {
            // Support the concat operator by using the concat array kernel.
            StringFnCodeGen.generateConcatCode(args, streamingNamedArgs, operation.getType())
        } else {
            BinOpCodeGen.generateBinOpCode(
                args,
                binOp,
                argDataTypes,
                builder,
                streamingNamedArgs,
                argScalars,
            )
        }
    }

    private fun visitPostfixOpScan(operation: RexCall): Expr {
        val args = visitList(operation.operands)
        val seriesOp = args[0]
        return PostfixOpCodeGen.generatePostfixOpCode(seriesOp, operation.operator)
    }

    private fun visitPrefixOpScan(operation: RexCall): Expr {
        val args = visitList(operation.operands)
        val seriesOp = args[0]
        return PrefixOpCodeGen.generatePrefixOpCode(seriesOp, operation.operator)
    }

    protected open fun visitInternalOp(node: RexCall): Expr {
        val sqlOp = node.operator.getKind()
        when (sqlOp) {
            SqlKind.SEARCH -> {
                // Note the valid use of Search args are enforced by the
                // SearchArgExpandProgram.
                val args = visitList(node.operands)
                return bodoSQLKernel("is_in", args)
            }

            else -> throw BodoSQLCodegenException(
                "Internal Error: Calcite Plan Produced an Internal Operator",
            )
        }
    }

    protected open fun visitLikeOp(node: RexCall): Expr {
        return visitLikeOp(node, listOf())
    }

    /**
     * Generate the code for a like operation.
     *
     * @param node The node from which to generate the expression.
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitLikeOp(
        node: RexCall,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        // The input node has ${index} as its first operand, where
        // ${index} is something like $3, and a SQL regular expression
        // as its second operand. If there is an escape value it will
        // be the third value, although it is not required and only supported
        // for LIKE and ILIKE
        val op: SqlLikeOperator = node.operator as SqlLikeOperator
        val operands = node.getOperands()
        val patternNode = operands[1]

        // The regular expression functions only support literal patterns
        val patternRegex = false
        val arg = operands[0].accept(this)
        val pattern = patternNode.accept(this)
        val escape: Expr
        if (operands.size == 3) {
            escape = operands[2].accept(this)
        } else {
            escape = Expr.StringLiteral("")
        }
        return if (patternRegex) {
            bodoSQLKernel(
                "regexp_like",
                listOf(arg, pattern, Expr.StringLiteral("")),
                streamingNamedArgs,
            )
        } else {
            bodoSQLKernel(
                "like_kernel",
                listOf(
                    arg,
                    pattern,
                    escape,
                    // Use the opposite. The python call is for case insensitivity while
                    // our boolean is for case sensitivity, so they are opposites.
                    Expr.BooleanLiteral(!op.isCaseSensitive),
                ),
                streamingNamedArgs,
            )
        }
    }

    /**
     * Overview of code generation for case:
     * https://bodo.atlassian.net/wiki/spaces/B/pages/1368752135/WIP+BodoSQL+Case+Implementation
     *
     * @param node The case node to visit
     * @return The resulting expression from generating case code.
     */
    protected open fun visitCaseOp(node: RexCall): Expr {
        // Extract the inputs as we need to know what columns to pass into
        // the case placeholder, and we need to rewrite the refs to the ones
        // we will generate.
        val inputFinder = CaseInputFinder()
        val operands = inputFinder.visitList(node.getOperands())

        // Generate the initialization of the local variables and also
        // fill in the access names for those local variables.
        builder.startCodegenFrame()
        val localRefsBuilder = ImmutableList.builder<Expr>()
        val arrs = Variable("arrs")
        val indexingVar = builder.symbolTable.genGenericTempVar()
        val closureVars: MutableList<Variable> = java.util.ArrayList()
        for (i in 0 until inputFinder.size()) {
            val localVar = Variable(getInput().name + "_" + i)
            closureVars.add(localVar)
            localRefsBuilder.add(
                Expr.Call("bodo.utils.indexing.scalar_optional_getitem", localVar, indexingVar),
            )
            val initLine = Assign(localVar, GetItem(arrs, Expr.IntegerLiteral(i)))
            builder.add(initLine)
        }
        closureVars.add(indexingVar)
        val localRefs: List<Expr> = localRefsBuilder.build()

        // Create a local translator for the case operands and initialize it with the
        // local variables we initialized above.
        val localTranslator =
            ScalarContext(
                visitor,
                builder,
                typeSystem,
                // Note: This is not a usage of the input so we access the raw value.
                input,
                dynamicParamTypes,
                namedParamTypeMap,
                localRefs,
                closureVars,
            )

        // Start a new codegen frame as we will perform our processing there.
        val arrVar = builder.symbolTable.genArrayVar()
        val outputFrame = visitCaseOperands(localTranslator, operands, listOf(arrVar, indexingVar), false)
        val caseBodyGlobal = visitor.lowerAsMetaType(FrameTripleQuotedString(outputFrame, 2))

        // Append all the closures generated to the init frame
        val closures = localTranslator.getClosures()
        for (closure in closures) {
            builder.add(closure)
        }
        val caseBodyInit = visitor.lowerAsMetaType(FrameTripleQuotedString(builder.endFrame(), 1))
        val caseArgs = visitList(inputFinder.getRefs())

        // Organize any named parameters if they exist.
        val namedArgs: MutableList<Pair<String, Expr>> = java.util.ArrayList()
        for (param in inputFinder.getDynamicParams()) {
            namedArgs.add(Pair<String, Expr>(param, Expr.Raw(param)))
        }

        // Generate the call to bodosql_case_placeholder and assign the results
        // to a temporary value that we return as the output.
        val tempVar = builder.symbolTable.genGenericTempVar()
        // Bodo needs to infer output type if not known by BodoSQL
        val outputArrayTypeGlobal: Variable =
            if (Utils.hasVariantType(node.getType())) {
                visitor.lowerAsGlobal(Expr.Raw("numba.core.types.unknown"))
            } else {
                visitor
                    .lowerAsGlobal(
                        BodoArrayHelpers.sqlTypeToBodoArrayType(
                            node.getType(),
                            false,
                            visitor.genDefaultTZ().zoneExpr,
                        ),
                    )
            }
        val casePlaceholder: Expr =
            Expr.Call(
                "bodo.utils.typing.bodosql_case_placeholder",
                listOf(
                    Expr.Tuple(caseArgs),
                    Expr.Call("len", getInput()),
                    caseBodyInit,
                    caseBodyGlobal,
                    // Note: The variable name must be a string literal here.
                    Expr.StringLiteral(arrVar.emit()),
                    // Note: The variable name must be a string literal here.
                    Expr.StringLiteral(indexingVar.emit()),
                    outputArrayTypeGlobal,
                ),
                namedArgs,
            )
        builder.add(Assign(tempVar, casePlaceholder))
        return tempVar
    }

    /** Utility to find variable uses inside a case statement.  */
    private class CaseInputFinder : RexShuttle() {
        private val refs: MutableList<RexSlot> = java.util.ArrayList()
        private val dynamicParams: MutableSet<String> = HashSet()

        fun getRefs(): List<RexSlot> {
            return ImmutableList.copyOf(refs)
        }

        fun getDynamicParams(): List<String> {
            return ImmutableList.copyOf(dynamicParams)
        }

        fun size(): Int {
            return refs.size
        }

        override fun visitInputRef(inputRef: RexInputRef): RexNode {
            return visitGenericRef(inputRef)
        }

        override fun visitLocalRef(localRef: RexLocalRef): RexNode {
            return visitGenericRef(localRef)
        }

        override fun visitCall(call: RexCall): RexNode {
            return super.visitCall(call)
        }

        override fun visitDynamicParam(dynamicParam: RexDynamicParam): RexNode {
            val paramName =
                if (dynamicParam is RexNamedParam) {
                    val paramBase = "_NAMED_PARAM_"
                    "$paramBase${dynamicParam.paramName}"
                } else {
                    val paramBase = "_DYNAMIC_PARAM_"
                    "$paramBase${dynamicParam.index}"
                }
            dynamicParams.add(paramName)
            return dynamicParam
        }

        private fun visitGenericRef(ref: RexSlot): RexNode {
            // Identify if there's an identical RexSlot that was
            // already found.
            for (i in refs.indices) {
                val it = refs[i]
                if (it == ref) {
                    // Return a RexLocalRef that refers to this index position.
                    return RexLocalRef(i, ref.type)
                }
            }

            // Record this as a new RexSlot and return a RexLocalRef
            // that refers to it.
            val next = refs.size
            refs.add(ref)
            return RexLocalRef(next, ref.type)
        }
    }

    /**
     * Visit the operands to a call to case using the given translator. Each operand generates its
     * code in a unique frame to define a unique scope.
     *
     *
     * The operands to case are of the form: [cond1, truepath1, cond2, truepath2, ..., elsepath]
     * All of these components are present, so the operands is always of Length 2n + 1, where n is the
     * number of conditions and n > 0. The else is always present, even if not explicit in the
     * original SQL query.
     *
     *
     * For code generation purposes we cannot produce an arbitrary set of if/else statements
     * because this can potentially trigger a max indentation depth issue in Python. As a result we
     * generate each if statement as its own scope and either continue or return directly from that
     * block.
     *
     *
     * For example, if we have 5 inputs with no intermediate variables, the generated code might
     * look like this:
     *
     *
     * `
     * if bodo.libs.bodosql_array_kernels.is_true(cond1):
     * out_arr[i] = truepath1
     * continue
     * if bodo.libs.bodosql_array_kernels.is_true(cond2):
     * out_arr[i] = truepath2
     * continue
     * out_arr[i] = elsepath
     ` *  If instead this is called from a scalar context, then we will be generating a closure
     * so each out_arr[i] should be a return instead
     *
     *
     * `
     * if bodo.libs.bodosql_array_kernels.is_true(cond1):
     * var = truepath1
     * return var
     * if bodo.libs.bodosql_array_kernels.is_true(cond2):
     * var = truepath2
     * return var
     * var = elsepath
     * return var
     ` *
     *
     * @param translator The translator used to visit each operand.
     * @param operands The list of RexNodes to visit to capture the proper computation.
     * @param outputVars The variables used in generating the output.
     * @param isScalarContext Is this code generated in a scalar context?
     * @return A single frame containing all the generated code.
     */
    protected open fun visitCaseOperands(
        translator: RexToBodoTranslator,
        operands: List<RexNode>,
        outputVars: List<Variable>,
        isScalarContext: Boolean,
    ): Frame {
        // Generate the target Frame for the output
        builder.startCodegenFrame()
        var i = 0
        while (i < operands.size - 1) {
            // Visit the cond code
            val cond = operands[i].accept(translator)
            // Visit the if code
            builder.startCodegenFrame()
            val ifPath = operands[i + 1].accept(translator)
            assignCasePathOutput(ifPath, outputVars, isScalarContext)
            // Pop the frame
            val ifFrame = builder.endFrame()
            // Generate the if statement
            val condCall = bodoSQLKernel("is_true", listOf(cond))
            val ifStatement = Op.If(condCall, ifFrame, null)
            builder.add(ifStatement)
            i += 2
        }
        // Process the else.
        val elsePath = operands[operands.size - 1].accept(translator)
        assignCasePathOutput(elsePath, outputVars, isScalarContext)
        return builder.endFrame()
    }

    /**
     * Assign the output value from a singular case path.
     *
     * @param outputExpr The expression from one of the then/else paths that needs to be assigned to
     * the final output.
     * @param outputVars The variables used in generating the output.
     * @param isScalarContext Is this code generated in a scalar context?
     */
    private fun assignCasePathOutput(
        outputExpr: Expr,
        outputVars: List<Variable>,
        isScalarContext: Boolean,
    ) {
        if (isScalarContext) {
            // Scalar path. Assign and return the variable.
            val outputVar = outputVars[0]
            builder.add(Assign(outputVar, outputExpr))
            builder.add(Op.ReturnStatement(outputVar))
        } else {
            // Unwrap the code
            val arrVar = outputVars[0]
            val indexVar = outputVars[1]
            val unwrappedExpr: Expr =
                Expr.Call("bodo.utils.conversion.unbox_if_tz_naive_timestamp", listOf(outputExpr))
            builder.add(SetItem(arrVar, indexVar, unwrappedExpr))
            builder.add(Continue)
        }
    }

    protected open fun visitCastScan(
        operation: RexCall,
        isSafe: Boolean,
    ): Expr {
        return visitCastScan(operation, isSafe, IsScalar.isScalar(operation), listOf())
    }

    /**
     * Generate the code for a cast operation.
     *
     * @param operation The operation from which to generate the expression.
     * @param outputScalar Is the output a scalar or an array?
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitCastScan(
        operation: RexCall,
        isSafe: Boolean,
        outputScalar: Boolean,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        val inputType = operation.operands[0].type
        val outputType = operation.getType()
        val operands = this.visitList(operation.getOperands())
        val fnName = getConversionName(outputType, isSafe)
        val (precision, scale) =
            if (SqlTypeFamily.EXACT_NUMERIC.contains(outputType)) {
                Pair(outputType.precision, outputType.scale)
            } else {
                Pair(RelDataType.PRECISION_NOT_SPECIFIED, RelDataType.SCALE_NOT_SPECIFIED)
            }
        return visitCastFunc(
            fnName,
            precision,
            scale,
            inputType,
            outputType,
            operands,
            listOf(outputScalar),
            streamingNamedArgs,
        )
    }

    private fun visitExtractScan(node: RexCall): Expr {
        val args = visitList(node.operands)
        val isTime = (node.operands[1].type.sqlTypeName == SqlTypeName.TIME)
        val isDate = (node.operands[1].type.sqlTypeName == SqlTypeName.DATE)
        val dateVal = args[0]
        val column = args[1]
        return ExtractCodeGen.generateExtractCode(
            dateVal.emit(),
            column,
            isTime,
            isDate,
            weekStart,
            weekOfYearPolicy,
        )
    }

    protected open fun visitSubstringScan(node: RexCall): Expr {
        return visitSubstringScan(node, listOf())
    }

    /**
     * Generate the code for a substring operation.
     *
     * @param node The operation from which to generate the expression.
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitSubstringScan(
        node: RexCall,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        // node.operands contains
        //  * String to perform the substring operation on
        //  * start index
        //  * substring length (optional)
        //  All of these values can be both scalars and columns
        // NOTE: check on number of arguments happen in generateSubstringCode
        val operands = visitList(node.operands)
        return StringFnCodeGen.generateSubstringCode(operands, streamingNamedArgs)
    }

    protected open fun visitGenericFuncOp(fnOperation: RexCall): Expr {
        return visitGenericFuncOp(fnOperation, false)
    }

    protected open fun visitNullIgnoringGenericFunc(
        fnOperation: RexCall,
        isSingleRow: Boolean,
        argScalars: List<Boolean>,
    ): Expr {
        return visitNullIgnoringGenericFunc(fnOperation, isSingleRow, listOf(), argScalars)
    }

    /**
     * Generate the code for generic functions that have special handling for null values.
     *
     * @param fnOperation The RexCall operation
     * @param isSingleRow Does the data operate on/output a single row?
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @param argScalars Whether each argument is a scalar or a column
     * @return The generated expression.
     */
    protected fun visitNullIgnoringGenericFunc(
        fnOperation: RexCall,
        isSingleRow: Boolean,
        streamingNamedArgs: List<Pair<String, Expr>>,
        argScalars: List<Boolean>,
    ): Expr {
        val fnName = fnOperation.operator.name
        val codeExprs: MutableList<Expr> = ArrayList()
        if ((fnName == "OBJECT_CONSTRUCT") || (fnName == "OBJECT_CONSTRUCT_KEEP_NULL")) {
            val keys: MutableList<Expr.StringLiteral> = ArrayList()
            val values: MutableList<Expr> = ArrayList()
            val scalars: MutableList<Expr.BooleanLiteral> = ArrayList()
            for (i in 0 until fnOperation.operands.size step 2) {
                keys.add(Expr.StringLiteral((fnOperation.operands[i] as RexLiteral).getValueAs(String::class.java)!!))
            }
            for (i in 1 until fnOperation.operands.size step 2) {
                values.add(fnOperation.operands[i].accept(this))
                scalars.add(Expr.BooleanLiteral(argScalars[i]))
            }
            return JsonCodeGen.getObjectConstructKeepNullCode(fnName, keys, values, scalars, visitor)
        }
        for (operand: RexNode in fnOperation.operands) {
            var operandInfo = operand.accept(this)
            // Need to unbox scalar timestamp values.
            if (isSingleRow || IsScalar.isScalar(operand)) {
                operandInfo =
                    Expr.Call(
                        "bodo.utils.conversion.unbox_if_tz_naive_timestamp", listOf(operandInfo),
                    )
            }
            codeExprs.add(operandInfo)
        }
        val result: Expr
        when (fnName) {
            "IFF", "BOOLNOT", "BOOLAND", "BOOLOR", "BOOLXOR", "NVL2" ->
                result =
                    CondOpCodeGen.getCondFuncCode(fnName, codeExprs)

            "EQUAL_NULL" ->
                result =
                    CondOpCodeGen.getCondFuncCodeOptimized(fnName, codeExprs, streamingNamedArgs, argScalars)

            "COALESCE", "ZEROIFNULL", "DECODE" ->
                result =
                    CondOpCodeGen.visitVariadic(fnName, codeExprs, streamingNamedArgs)

            "ARRAY_CONSTRUCT" -> result = visitArrayConstruct(codeExprs, argScalars)
            "ARRAY_CONSTRUCT_COMPACT" -> {
                result = visitArrayConstruct(codeExprs, argScalars)
                var isScalar = true
                for (i: Boolean in argScalars) {
                    if (!i) {
                        isScalar = false
                        break
                    }
                }
                return visitNestedArrayFunc("ARRAY_COMPACT", java.util.List.of(result), java.util.List.of(isScalar))
            }

            "HASH" -> result = CondOpCodeGen.visitHash(codeExprs, argScalars, visitor)
            else -> throw BodoSQLCodegenException("Internal Error: reached unreachable code")
        }
        return result
    }

    /**
     * Represents a cast operation that isn't generated by the planner. This is necessary
     * for operators like DynamicParams that aren't supported directly by the planner.
     */
    protected open fun visitDynamicCast(
        arg: Expr,
        inputType: RelDataType,
        outputType: RelDataType,
        isScalar: Boolean,
    ): Expr {
        return visitDynamicCast(arg, inputType, outputType, isScalar, listOf())
    }

    /**
     * Generate the code for a cast operation that isn't generated by the planner. TODO(njriasan):
     * Remove and update the planner to insert these casts.
     *
     * @param arg The arg being cast.
     * @param inputType The input type.
     * @param outputType The output type.
     * @param isScalar Is the input/output a scalar value.
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitDynamicCast(
        arg: Expr,
        inputType: RelDataType,
        outputType: RelDataType,
        isScalar: Boolean,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        val fnName = getConversionName(outputType, false)
        val (precision, scale) =
            if (SqlTypeFamily.EXACT_NUMERIC.contains(outputType)) {
                Pair(outputType.precision, outputType.scale)
            } else {
                Pair(RelDataType.PRECISION_NOT_SPECIFIED, RelDataType.SCALE_NOT_SPECIFIED)
            }
        return visitCastFunc(
            fnName,
            precision,
            scale,
            inputType,
            outputType,
            listOf(arg),
            listOf(isScalar),
            streamingNamedArgs,
        )
    }

    protected open fun visitTrimFunc(
        fnName: String,
        stringToBeTrimmed: Expr,
        charactersToBeTrimmed: Expr,
    ): Expr {
        return visitTrimFunc(fnName, stringToBeTrimmed, charactersToBeTrimmed, listOf())
    }

    /**
     * Generate the code for the TRIM functions.
     *
     * @param fnName The name of the TRIM function.
     * @param stringToBeTrimmed Expr for the string to be trim.
     * @param charactersToBeTrimmed Expr for identifying the characters to trim.
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitTrimFunc(
        fnName: String,
        stringToBeTrimmed: Expr,
        charactersToBeTrimmed: Expr,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        return StringFnCodeGen.generateTrimFnCode(fnName, stringToBeTrimmed, charactersToBeTrimmed, streamingNamedArgs)
    }

    /**
     * Generate the code for a NULLIF function.
     *
     * @param operands The arguments to the function.
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitNullIfFunc(
        operands: List<Expr>,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        return bodoSQLKernel("nullif", operands, streamingNamedArgs)
    }

    protected open fun visitNullIfFunc(operands: List<Expr>): Expr {
        return visitNullIfFunc(operands, listOf())
    }

    /**
     * Generate the code for the Least/Greatest.
     *
     * @param fnName The name of the function.
     * @param operands The arguments to the function.
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitLeastGreatest(
        fnName: String,
        operands: List<Expr>,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        return NumericCodeGen.generateLeastGreatestCode(fnName, operands, streamingNamedArgs)
    }

    protected open fun visitLeastGreatest(
        fnName: String,
        operands: List<Expr>,
    ): Expr {
        return visitLeastGreatest(fnName, operands, listOf())
    }

    /**
     * Generate the code for Position.
     *
     * @param operands The arguments to the function.
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitPosition(
        operands: List<Expr>,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        return StringFnCodeGen.generatePosition(operands, streamingNamedArgs)
    }

    protected open fun visitPosition(operands: List<Expr>): Expr {
        return StringFnCodeGen.generatePosition(operands, listOf())
    }

    /** Wrapper to unpack the RexCall so the implementation can be reused for cast/try_cast.  */
    protected open fun visitCastFunc(
        fnOperation: RexCall,
        operands: List<Expr>,
        argScalars: List<Boolean>,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        var precision = RelDataType.PRECISION_NOT_SPECIFIED
        var scale = RelDataType.SCALE_NOT_SPECIFIED
        val fnName = fnOperation.operator.name
        // TODO: Update when we add format string support.
        if (fnName == "TO_NUMBER" || fnName == "TRY_TO_NUMBER") {
            val rexOperands = fnOperation.getOperands()
            if (rexOperands.size > 1) {
                precision = (rexOperands[1] as RexLiteral).getValueAs(BigDecimal::class.java)!!.intValueExact()
            }
            if (rexOperands.size > 2) {
                scale = (rexOperands[2] as RexLiteral).getValueAs(BigDecimal::class.java)!!.intValueExact()
            }
        }
        val inputType = fnOperation.operands[0].type
        val outputType = fnOperation.getType()
        return visitCastFunc(
            fnOperation.operator.name,
            precision,
            scale,
            inputType,
            outputType,
            operands,
            argScalars,
            streamingNamedArgs,
        )
    }

    /**
     * Determine if a cast function can omit any cast operations.
     * @param inputType The input type.
     * @param outputType The output type.
     * @param scale The scale of the output type.
     */
    private fun canOmitCast(
        inputType: RelDataType,
        outputType: RelDataType,
        precision: Int,
        scale: Int,
    ): Boolean {
        return if (inputType == outputType) {
            true
        } else if (inputType.sqlTypeName == outputType.sqlTypeName && inputType.sqlTypeName != SqlTypeName.DECIMAL) {
            // Can omit cast if the input and output types are the same, and it's not a type where precision matters
            // to Bodo.
            true
        } else if (SqlTypeFamily.CHARACTER.contains(inputType) && SqlTypeFamily.CHARACTER.contains(outputType)) {
            // Can omit cast if the input and output types are both character types.
            true
        } else if (SqlTypeFamily.INTEGER.contains(inputType) && SqlTypeFamily.INTEGER.contains(outputType)) {
            inputType.precision <= precision && scale == 0
        } else {
            false
        }
    }

    /**
     * Generate the code for Cast function calls.
     *
     * @param fnName The name of the output function code.
     * @param precision The output precision. This is needed because TO_NUMBER doesn't output a
     *     decimal yet.
     * @param scale The output scale. This is needed because TO_NUMBER doesn't output a decimal yet.
     * @param operands The arguments to the function.
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     *     we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitCastFunc(
        fnName: String,
        precision: Int,
        scale: Int,
        inputType: RelDataType,
        outputType: RelDataType,
        operands: List<Expr>,
        argScalars: List<Boolean>,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        if (canOmitCast(inputType, outputType, precision, scale)) {
            return operands[0]
        }
        when (fnName) {
            "TIMESTAMP" -> return ConversionCodeGen.generateTimestampFnCode(operands, streamingNamedArgs)
            "TO_DATE", "TRY_TO_DATE" -> return ConversionCodeGen.generateToDateFnCode(
                operands,
                fnName,
                streamingNamedArgs,
            )

            "TO_TIMESTAMP", "TO_TIMESTAMP_NTZ",
            "TRY_TO_TIMESTAMP", "TRY_TO_TIMESTAMP_NTZ",
            -> return ConversionCodeGen.generateToTimestampFnCode(
                operands,
                Expr.None,
                fnName,
                streamingNamedArgs,
            )

            "TRY_TO_TIMESTAMP_LTZ", "TO_TIMESTAMP_LTZ" -> return ConversionCodeGen.generateToTimestampFnCode(
                operands,
                visitor.genDefaultTZ().zoneExpr,
                fnName,
                streamingNamedArgs,
            )

            "TO_TIMESTAMPTZ", "TRY_TO_TIMESTAMPTZ" -> return ConversionCodeGen.generateToTimestampTzFnCode(
                operands,
                visitor.genDefaultTZ().zoneExpr,
                fnName,
                streamingNamedArgs,
            )

            "TRY_TO_BOOLEAN", "TO_BOOLEAN" -> return ConversionCodeGen.generateToBooleanFnCode(
                operands,
                fnName,
                streamingNamedArgs,
            )

            "TRY_TO_BINARY", "TO_BINARY" -> return ConversionCodeGen.generateToBinaryFnCode(
                operands,
                fnName,
                streamingNamedArgs,
            )

            "TO_VARCHAR" -> return ConversionCodeGen.generateToCharFnCode(operands, argScalars)
            "TO_NUMBER" -> return NumericCodeGen.generateToNumberCode(
                operands[0],
                precision,
                scale,
                false,
                // Differentiate between integer and decimal casts since
                // both could use the same precision and scale. In the future
                // we may want to separate these, but currently that would
                // lead to a lot of duplicated code.
                outputType.sqlTypeName == SqlTypeName.DECIMAL,
                streamingNamedArgs,
            )

            "TRY_TO_NUMBER" -> return NumericCodeGen.generateToNumberCode(
                operands[0],
                precision,
                scale,
                true,
                // Differentiate between integer and decimal casts since
                // both could use the same precision and scale. In the future
                // we may want to separate these, but currently that would
                // lead to a lot of duplicated code.
                outputType.sqlTypeName == SqlTypeName.DECIMAL,
                streamingNamedArgs,
            )

            "TO_DOUBLE", "TRY_TO_DOUBLE" -> return ConversionCodeGen.generateToDoubleFnCode(
                operands,
                fnName,
                streamingNamedArgs,
            )

            "TO_TIME", "TRY_TO_TIME" -> return DatetimeFnCodeGen.generateToTimeCode(
                operands,
                fnName,
                streamingNamedArgs,
            )

            "TO_ARRAY" -> return ConversionCodeGen.generateToArrayCode(operands, argScalars)
            "TO_VARIANT" -> {
                assert(operands.size == 1)
                return operands[0]
            }

            "TO_OBJECT" -> {
                assert(operands.size == 1)
                return bodoSQLKernel("to_object", operands, java.util.List.of())
            }

            else -> throw BodoSQLCodegenException(String.format("Unexpected Cast function: %s", fnName))
        }
    }

    protected open fun visitCastFunc(
        fnOperation: RexCall,
        operands: List<Expr>,
        argScalars: List<Boolean>,
    ): Expr {
        return visitCastFunc(fnOperation, operands, argScalars, listOf())
    }

    /**
     * Generate the code for Regex function calls.
     *
     * @param fnOperation The RexNode for the function call.
     * @param operands The arguments to the function.
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitRegexFunc(
        fnOperation: RexCall,
        operands: List<Expr>,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        when (val fnName = fnOperation.operator.name) {
            "REGEXP_LIKE" -> {
                if (!(operands.size in 2..3)) {
                    throw BodoSQLCodegenException(
                        "Error, invalid number of arguments passed to REGEXP_LIKE",
                    )
                }
                if (operands.size == 3 && !IsScalar.isScalar(fnOperation.operands[2])) {
                    throw BodoSQLCodegenException(
                        "Error, FLAG argument for REGEXP functions must be a scalar",
                    )
                }
                return RegexpCodeGen.generateRegexpLikeInfo(operands, streamingNamedArgs)
            }

            "REGEXP_COUNT" -> {
                if (!(operands.size in 2..4)) {
                    throw BodoSQLCodegenException(
                        "Error, invalid number of arguments passed to REGEXP_COUNT",
                    )
                }
                if (operands.size == 4 && !IsScalar.isScalar(fnOperation.operands[3])) {
                    throw BodoSQLCodegenException(
                        "Error, FLAG argument for REGEXP functions must be a scalar",
                    )
                }
                return RegexpCodeGen.generateRegexpCountInfo(operands, streamingNamedArgs)
            }

            "REGEXP_REPLACE" -> {
                if (!(operands.size in 2..6)) {
                    throw BodoSQLCodegenException(
                        "Error, invalid number of arguments passed to REGEXP_REPLACE",
                    )
                }
                if (operands.size == 6 && !IsScalar.isScalar(fnOperation.operands[5])) {
                    throw BodoSQLCodegenException(
                        "Error, FLAG argument for REGEXP functions must be a scalar",
                    )
                }
                return RegexpCodeGen.generateRegexpReplaceInfo(operands, streamingNamedArgs)
            }

            "REGEXP_SUBSTR" -> {
                if (!(operands.size in 2..6)) {
                    throw BodoSQLCodegenException(
                        "Error, invalid number of arguments passed to REGEXP_SUBSTR",
                    )
                }
                if ((
                        !IsScalar.isScalar(fnOperation.operands[1]) ||
                            (operands.size > 4 && !IsScalar.isScalar(fnOperation.operands[4]))
                    )
                ) {
                    throw BodoSQLCodegenException(
                        "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar",
                    )
                }
                return RegexpCodeGen.generateRegexpSubstrInfo(operands, streamingNamedArgs)
            }

            "REGEXP_INSTR" -> {
                if (!(operands.size in 2..7)) {
                    throw BodoSQLCodegenException(
                        "Error, invalid number of arguments passed to REGEXP_INSTR",
                    )
                }
                if ((
                        !IsScalar.isScalar(fnOperation.operands[1]) ||
                            (operands.size > 5 && !IsScalar.isScalar(fnOperation.operands[5]))
                    )
                ) {
                    throw BodoSQLCodegenException(
                        "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar",
                    )
                }
                return RegexpCodeGen.generateRegexpInstrInfo(operands, streamingNamedArgs)
            }

            else -> throw BodoSQLCodegenException(String.format("Unexpected Regex function: %s", fnName))
        }
    }

    protected open fun visitStringFunc(
        fnOperation: RexCall,
        operands: List<Expr>,
        isSingleRow: Boolean,
    ): Expr {
        return visitStringFunc(fnOperation, operands, listOf(), isSingleRow)
    }

    /**
     * Generate the code for String function calls.
     *
     * @param fnOperation The RexNode for the function call.
     * @param operands The arguments to the function.
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitStringFunc(
        fnOperation: RexCall,
        operands: List<Expr>,
        streamingNamedArgs: List<Pair<String, Expr>>,
        isSingleRow: Boolean,
    ): Expr {
        when (val fnName = fnOperation.operator.name) {
            "REGEXP_LIKE", "REGEXP_COUNT", "REGEXP_REPLACE", "REGEXP_SUBSTR", "REGEXP_INSTR" -> return visitRegexFunc(
                fnOperation,
                operands,
                streamingNamedArgs,
            )

            "CHAR", "FORMAT" -> return StringFnCodeGen.getStringFnCode(fnName, operands)
            "ASCII", "CHAR_LENGTH", "LENGTH", "REVERSE",
            "LOWER", "UPPER", "SPACE", "RTRIMMED_LENGTH",
            "JAROWINKLER_SIMILARITY", "REPEAT", "STRCMP",
            "RIGHT", "LEFT", "CONTAINS", "INSTR", "INSERT",
            "STARTSWITH", "ENDSWITH", "SPLIT_PART",
            "SUBSTRING_INDEX", "TRANSLATE3", "SPLIT",
            -> return StringFnCodeGen.getOptimizedStringFnCode(
                fnName,
                operands,
                streamingNamedArgs,
            )

            "RPAD", "LPAD" -> return StringFnCodeGen.generatePadCode(fnOperation, operands, streamingNamedArgs)
            "SUBSTRING" -> return StringFnCodeGen.generateSubstringCode(operands, streamingNamedArgs)
            "CHARINDEX" -> return visitPosition(operands, streamingNamedArgs)
            "STRTOK" -> return StringFnCodeGen.generateStrtok(operands, streamingNamedArgs)
            "STRTOK_TO_ARRAY" -> return StringFnCodeGen.generateStrtokToArray(operands, streamingNamedArgs)
            "EDITDISTANCE" -> return StringFnCodeGen.generateEditdistance(operands, streamingNamedArgs)
            "INITCAP" -> return StringFnCodeGen.generateInitcapInfo(operands, streamingNamedArgs)
            "REPLACE" -> return StringFnCodeGen.generateReplace(operands, streamingNamedArgs)
            "SHA2" -> return StringFnCodeGen.generateSHA2(operands, streamingNamedArgs)
            "MD5" -> return bodoSQLKernel("md5", operands, streamingNamedArgs)
            "HEX_ENCODE" -> return StringFnCodeGen.generateHexEncode(operands, streamingNamedArgs)
            "HEX_DECODE_STRING", "HEX_DECODE_BINARY",
            "TRY_HEX_DECODE_STRING", "TRY_HEX_DECODE_BINARY",
            -> return StringFnCodeGen.generateHexDecodeFn(
                fnName,
                operands,
                streamingNamedArgs,
            )

            "BASE64_ENCODE" -> return StringFnCodeGen.generateBase64Encode(operands, streamingNamedArgs)
            "BASE64_DECODE_STRING", "TRY_BASE64_DECODE_STRING",
            "BASE64_DECODE_BINARY", "TRY_BASE64_DECODE_BINARY",
            -> return StringFnCodeGen.generateBase64DecodeFn(
                fnName,
                operands,
                streamingNamedArgs,
            )

            "UUID_STRING" -> return StringFnCodeGen.generateUUIDString(getInput(), isSingleRow, operands, streamingNamedArgs)
            else -> throw BodoSQLCodegenException(String.format("Unexpected String function: %s", fnName))
        }
    }

    /**
     * Implementation for functions that match or resemble Snowflake General context functions.
     *
     *
     * https://docs.snowflake.com/en/sql-reference/functions-context
     *
     *
     * These function are typically non-deterministic, so they must be called outside any loop to
     * give consistent results and should be required to hold the same value on all ranks. If called
     * inside a Case statement then we won't make the results consistent.
     *
     * @param fnOperation The RexCall that is producing a system operation.
     * @param makeConsistent Should the function be made consistent. This influences the generated
     * function call.
     * @return A variable holding the result. This function always writes its result to an
     * intermediate variable because it needs to insert the code into the Builder without being
     * caught in the body of a loop for streaming.
     */
    protected fun visitGeneralContextFunction(
        fnOperation: RexCall,
        makeConsistent: Boolean,
    ): Variable {
        val fnName = fnOperation.operator.name.uppercase()
        val systemCall: Expr
        when (fnName) {
            "GETDATE" ->
                systemCall =
                    DatetimeFnCodeGen.generateCurrTimestampCode(visitor.genDefaultTZ().zoneExpr, makeConsistent)

            "CURRENT_TIME" -> {
                val tzTimeInfo = BodoTZInfo.getDefaultTZInfo(typeSystem)
                systemCall = DatetimeFnCodeGen.generateCurrTimeCode(tzTimeInfo, makeConsistent)
            }

            "UTC_TIMESTAMP" -> systemCall = DatetimeFnCodeGen.generateUTCTimestampCode(makeConsistent)
            "UTC_DATE" -> systemCall = DatetimeFnCodeGen.generateUTCDateCode(makeConsistent)
            "CURRENT_DATE" ->
                systemCall =
                    DatetimeFnCodeGen.generateCurrentDateCode(
                        BodoTZInfo.getDefaultTZInfo(
                            typeSystem,
                        ),
                        makeConsistent,
                    )

            "CURRENT_ACCOUNT", "CURRENT_ACCOUNT_NAME" ->
                if (currentAccount != null) {
                    var acct = currentAccount!!
                    val idx = acct.lastIndexOf('.')
                    if (idx != -1) {
                        acct = acct.substring(0, idx)
                    }
                    acct = acct.uppercase()
                    systemCall = Expr.StringLiteral(acct)
                } else {
                    throw BodoSQLCodegenException("No information about current account is found.")
                }
            "CURRENT_DATABASE" ->
                if (currentDatabase != null) {
                    systemCall = Expr.StringLiteral(currentDatabase!!)
                } else {
                    throw BodoSQLCodegenException("No information about current database is found.")
                }

            else -> throw BodoSQLCodegenException(String.format(Locale.ROOT, "Unsupported System function: %s", fnName))
        }
        val finalVar = builder.symbolTable.genGenericTempVar()
        val assign = Assign(finalVar, systemCall)
        builder.addPureScalarAssign(assign)
        return finalVar
    }

    protected open fun visitGeneralContextFunction(fnOperation: RexCall): Variable {
        return visitGeneralContextFunction(fnOperation, true)
    }

    /**
     * Implementation for functions that use nested arrays.
     *
     * @param fnName The name of the function.
     * @param fnOperands The arguments to the function.
     * @param argScalars Indicates which arguments are scalars
     * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
     * we aren't in a streaming context.
     * @return The generated expression.
     */
    protected fun visitNestedArrayFunc(
        fnName: String,
        fnOperands: List<Expr>,
        argScalars: List<Boolean>,
        streamingNamedArgs: List<Pair<String, Expr>>,
    ): Expr {
        val kwargs = ArrayList<Pair<String, Expr>>()
        when (fnName) {
            "ARRAY_COMPACT", "ARRAY_REMOVE_AT", "ARRAY_SIZE", "ARRAY_SLICE", "ARRAY_TO_STRING", "TO_ARRAY" ->
                kwargs.add(
                    Pair<String, Expr>(
                        "is_scalar",
                        Expr.BooleanLiteral(
                            (argScalars[0]),
                        ),
                    ),
                )

            "ARRAYS_OVERLAP", "ARRAY_CAT", "ARRAY_CONTAINS", "ARRAY_EXCEPT", "ARRAY_INTERSECTION", "ARRAY_POSITION", "ARRAY_REMOVE" -> {
                kwargs.add(
                    Pair<String, Expr>(
                        "is_scalar_0",
                        Expr.BooleanLiteral(
                            (argScalars[0]),
                        ),
                    ),
                )
                kwargs.add(
                    Pair<String, Expr>(
                        "is_scalar_1",
                        Expr.BooleanLiteral(
                            (argScalars[1]),
                        ),
                    ),
                )
            }

            else -> throw BodoSQLCodegenException(
                String.format(
                    Locale.ROOT,
                    "Unsupported nested Array function: %s",
                    fnName,
                ),
            )
        }
        return bodoSQLKernel(fnName.lowercase(), fnOperands, kwargs)
    }

    /**
     * Code generation for Snowflake UDFs that are not being inlined. These generate 3 stages: -
     * Create the UDF for the source language - Execute the UDF for each batch of data - Delete the
     * UDF This is generic enough to handle all UDFs that are not inlined by providing a different
     * function for each supported language with the same structure.
     *
     *
     * The design can be found here:
     * https://bodo.atlassian.net/wiki/spaces/B/pages/1615200311/Javascript+UDFs#Code-Generation
     *
     * @param udf Operator that holds the code generation information for the UDF.
     * @param operands The operands to pass to the kernel when executing the UDF.
     * @param returnType The return type of the UDF.
     * @return The variable holding the result of executing the UDF.
     */
    private fun visitSnowflakeUDF(
        udf: SnowflakeNativeUDF,
        operands: List<Expr>,
        returnType: RelDataType,
    ): Variable {
        // Verify the language is supported
        val createFunctionName: String
        val executeFunctionName: String
        val deleteFunctionName: String
        when (udf.functionLanguage) {
            "JAVASCRIPT" -> {
                createFunctionName = "create_javascript_udf"
                executeFunctionName = "execute_javascript_udf"
                deleteFunctionName = "delete_javascript_udf"
            }

            else -> throw BodoSQLCodegenException(
                String.format(
                    Locale.ROOT,
                    "Unsupported language for Snowflake UDF: %s",
                    udf.functionLanguage,
                ),
            )
        }
        // Lower the globals needed to create the UDF
        val udfBody = udf.functionBody
        val textVar = visitor.lowerAsMetaType(TripleQuotedString(udfBody))
        val paramNamesExpr = udf.parameterNames.map { Expr.StringLiteral(it) }
        val paramNamesVar = visitor.lowerAsMetaType(Expr.Tuple(paramNamesExpr))
        val returnTypeExpr = BodoArrayHelpers.sqlTypeToBodoArrayType(returnType, false, visitor.genDefaultTZ().zoneExpr)
        val returnTypeVar = visitor.lowerAsGlobal(returnTypeExpr)
        // Construct the create function and place it before the pipeline.
        val createCall = bodoSQLKernel(createFunctionName, listOf(textVar, paramNamesVar, returnTypeVar))
        val functionVar = visitor.genGenericTempVar()
        val createAssign = Assign(functionVar, createCall)
        // TODO: Determine a cleaner API.
        if (builder.isStreamingFrame()) {
            builder.getCurrentStreamingPipeline().addInitialization(createAssign)
        } else {
            builder.add(createAssign)
        }
        // Construct the execute function and place it in the pipeline body.
        val argsTuple: Expr = Expr.Tuple(operands)
        val executeCall = bodoSQLKernel(executeFunctionName, listOf(functionVar, argsTuple))
        val executeVar = visitor.genGenericTempVar()
        val executeAssign = Assign(executeVar, executeCall)
        builder.add(executeAssign)
        // Add the delete function to the end of the pipeline.
        val deleteCall = bodoSQLKernel(deleteFunctionName, listOf(functionVar))
        val deleteStmt = Stmt(deleteCall)
        // TODO: Determine a cleaner API.
        if (builder.isStreamingFrame()) {
            builder.getCurrentStreamingPipeline().addTermination(deleteStmt)
        } else {
            builder.add(deleteStmt)
        }
        return executeVar
    }

    /**
     * Constructs the Expression to make a call to the variadic functions OBJECT_DELETE or
     * OBJECT_PICK.
     *
     * @param codeExprs the Python expressions to calculate the arguments.
     * @param keep True for OBJECT_PICK and false for OBJECT_DELETE.
     * @param argScalars List of booleans indicating which arguments are scalars.
     * @return Expr containing the code generated for the relational expression.
     */
    fun visitObjectPickDelete(
        codeExprs: List<Expr>,
        keep: Boolean,
        argScalars: List<Boolean>,
    ): Expr {
        val scalarExprs: MutableList<Expr> = ArrayList()
        for (isScalar: Boolean in argScalars) {
            scalarExprs.add(Expr.BooleanLiteral((isScalar)))
        }
        val scalarGlobal: Expr = visitor.lowerAsMetaType(Expr.Tuple(scalarExprs))
        return bodoSQLKernel(
            "object_filter_keys",
            listOf(
                Expr.Tuple(codeExprs),
                Expr.BooleanLiteral(
                    (keep),
                ),
                scalarGlobal,
            ),
        )
    }

    protected fun visitNestedArrayFunc(
        fnName: String,
        operands: List<Expr>,
        argScalars: List<Boolean>,
    ): Expr {
        return visitNestedArrayFunc(fnName, operands, argScalars, listOf())
    }

    protected fun visitGenericFuncOp(
        fnOperation: RexCall,
        isSingleRow: Boolean,
    ): Expr {
        var fnName = fnOperation.operator.toString()
        val argScalars: ArrayList<Boolean> = ArrayList()
        for (operand: RexNode in fnOperation.operands) {
            argScalars.add(isOperandScalar(operand))
        }
        // Handle functions that do not care about nulls separately
        if ((
                fnName === "ARRAY_CONSTRUCT"
            ) || (
                fnName === "ARRAY_CONSTRUCT_COMPACT"
            ) || (
                fnName === "BOOLAND"
            ) || (
                fnName === "BOOLNOT"
            ) || (
                fnName === "BOOLOR"
            ) || (
                fnName === "BOOLXOR"
            ) || (
                fnName === "COALESCE"
            ) || (
                fnName === "DECODE"
            ) || (
                fnName === "EQUAL_NULL"
            ) || (
                fnName === "HASH"
            ) || (
                fnName === "IF"
            ) || (
                fnName === "IFF"
            ) || (
                fnName === "NVL"
            ) || (
                fnName === "NVL2"
            ) || (
                fnName === "OBJECT_CONSTRUCT"
            ) || (
                fnName === "OBJECT_CONSTRUCT_KEEP_NULL"
            ) || (fnName === "ZEROIFNULL")
        ) {
            return visitNullIgnoringGenericFunc(fnOperation, isSingleRow, argScalars)
        }

        // Extract all inputs to the current function.
        val operands = visitList(fnOperation.operands)

        // Handle UDFs separately since they each have their own name
        if (fnOperation.op is SnowflakeNativeUDF) {
            return visitSnowflakeUDF(
                fnOperation.operator as SnowflakeNativeUDF,
                operands,
                fnOperation.getType(),
            )
        }

        val dateTimeExprType1: DatetimeFnCodeGen.DateTimeType
        val dateTimeExprType2: DatetimeFnCodeGen.DateTimeType
        val isTime: Boolean
        val isDate: Boolean
        var unit: String
        when (fnOperation.operator.kind) {
            SqlKind.MOD -> return NumericCodeGen.getNumericFnCode(fnName, operands)
            SqlKind.GREATEST, SqlKind.LEAST -> return visitLeastGreatest(fnOperation.operator.toString(), operands)
            SqlKind.TRIM -> {
                assert(operands.size == 3)
                assert(fnOperation.operands[0] is RexLiteral)
                val literal = fnOperation.operands[0] as RexLiteral
                val argValue = literal.value2.toString().uppercase()
                if ((argValue == "BOTH")) {
                    fnName = "trim"
                } else if ((argValue == "LEADING")) {
                    fnName = "ltrim"
                } else {
                    assert((argValue == "TRAILING"))
                    fnName = "rtrim"
                }
                // Calcite expects: TRIM(<chars> FROM <expr>>) or TRIM(<chars>, <expr>)
                // However, Snowflake/BodoSQL expects: TRIM(<expr>, <chars>)
                // So we just need to swap the arguments here.
                return visitTrimFunc(fnName, operands[2], operands[1])
            }

            SqlKind.POSITION -> return visitPosition(operands)
            SqlKind.RANDOM -> return NumericCodeGen.generateRandomFnInfo(getInput(), isSingleRow)
            SqlKind.ITEM -> {
                assert(operands.size == 2)
                return JsonCodeGen.visitGetOp(
                    isOperandScalar(fnOperation.operands[0]),
                    isOperandScalar(fnOperation.operands[1]),
                    operands,
                )
            }

            SqlKind.OTHER, SqlKind.OTHER_FUNCTION -> {
                when (fnName) {
                    "CEIL", "FLOOR" -> return NumericCodeGen.genFloorCeilCode(fnName, operands)
                    "LTRIM", "RTRIM" ->
                        if (operands.size == 1) { // no optional characters to be trimmed
                            return visitTrimFunc(
                                fnName,
                                operands[0],
                                Expr.StringLiteral(" "),
                            ) // remove spaces by default
                        } else {
                            return if (operands.size == 2) {
                                visitTrimFunc(fnName, operands[0], operands[1])
                            } else {
                                throw BodoSQLCodegenException(
                                    "Invalid number of arguments to TRIM: must be either 1 or 2.",
                                )
                            }
                        }

                    "WIDTH_BUCKET" -> {
                        val numOps = operands.size
                        assert(numOps == 4) { "WIDTH_BUCKET takes 4 arguments, but found $numOps" }
                        return bodoSQLKernel("width_bucket", operands)
                    }

                    "HAVERSINE" -> {
                        assert(operands.size == 4)
                        return bodoSQLKernel("haversine", operands, java.util.List.of())
                    }

                    "DIV0" -> {
                        assert(operands.size == 2 && fnOperation.operands.size == 2)
                        return bodoSQLKernel("div0", operands, java.util.List.of())
                    }

                    "NULLIF" -> {
                        assert(operands.size == 2)
                        return visitNullIfFunc(operands)
                    }

                    "DATEADD" -> {
                        // If DATEADD receives 3 arguments, use the Snowflake DATEADD.
                        // Otherwise, fall back to the normal DATEADD. TIMEADD and TIMESTAMPADD are aliases.
                        if (operands.size == 3) {
                            dateTimeExprType1 = DatetimeFnCodeGen.getDateTimeDataType(fnOperation.getOperands()[2])
                            unit =
                                DatetimeFnCodeGen.standardizeTimeUnit(fnName, operands[0].emit(), dateTimeExprType1)
                            assert(IsScalar.isScalar(fnOperation.operands[0]))
                            return DateAddCodeGen.generateSnowflakeDateAddCode(operands.subList(1, operands.size), unit)
                        }
                        run {
                            assert(operands.size == 2)
                            // If the second argument is a timedelta, switch to manual addition
                            val manualAddition: Boolean =
                                SqlTypeName.INTERVAL_TYPES.contains(
                                    fnOperation.getOperands()[1].type.sqlTypeName,
                                )
                            // Cannot use dateadd/datesub functions on TIME data unless the
                            // amount being added to them is a timedelta
                            if ((
                                    !manualAddition &&
                                        (
                                            DatetimeFnCodeGen.getDateTimeDataType(fnOperation.getOperands()[0])
                                                == DatetimeFnCodeGen.DateTimeType.TIME
                                        )
                                )
                            ) {
                                throw BodoSQLCodegenException("Cannot add/subtract days from TIME")
                            }
                            val dateIntervalTypes: Set<SqlTypeName> =
                                Sets.immutableEnumSet(
                                    SqlTypeName.INTERVAL_YEAR_MONTH,
                                    SqlTypeName.INTERVAL_YEAR,
                                    SqlTypeName.INTERVAL_MONTH,
                                    SqlTypeName.INTERVAL_DAY,
                                )
                            val isDateInterval: Boolean =
                                dateIntervalTypes.contains(
                                    fnOperation.getOperands()[1].type.sqlTypeName,
                                )
                            var arg0: Expr = operands[0]
                            var arg1: Expr = operands[1]
                            // Cast arg0 to from string to timestamp, if needed
                            if (SqlTypeName.STRING_TYPES.contains(
                                    fnOperation.getOperands()[0].type.sqlTypeName,
                                )
                            ) {
                                val inputType: RelDataType = fnOperation.getOperands()[0].type
                                // The output type will always be the timestamp the string is being cast to.
                                val outputType: RelDataType = fnOperation.getType()
                                arg0 =
                                    visitDynamicCast(
                                        operands[0],
                                        inputType,
                                        outputType,
                                        isSingleRow || IsScalar.isScalar(fnOperation.operands[0]),
                                    )
                            }
                            // add/minus a date interval to a date object should return a date object
                            if ((
                                    isDateInterval &&
                                        DatetimeFnCodeGen.getDateTimeDataType(
                                            fnOperation.getOperands()[0],
                                        ) == DatetimeFnCodeGen.DateTimeType.DATE
                                )
                            ) {
                                if ((fnName == "SUBDATE") || (fnName == "DATE_SUB")) {
                                    arg1 =
                                        bodoSQLKernel(
                                            "negate",
                                            listOf(arg1),
                                        )
                                }
                                return bodoSQLKernel(
                                    "add_date_interval_to_date",
                                    listOf(arg0, arg1),
                                )
                            }
                            return DateAddCodeGen.generateMySQLDateAddCode(arg0, arg1, manualAddition, fnName)
                        }
                    }

                    "ADDDATE", "SUBDATE", "DATE_SUB" -> {
                        assert(operands.size == 2)
                        val manualAddition =
                            SqlTypeName.INTERVAL_TYPES.contains(
                                fnOperation.getOperands()[1].type.sqlTypeName,
                            )
                        if ((
                                !manualAddition &&
                                    (
                                        DatetimeFnCodeGen.getDateTimeDataType(fnOperation.getOperands()[0])
                                            == DatetimeFnCodeGen.DateTimeType.TIME
                                    )
                            )
                        ) {
                            throw BodoSQLCodegenException("Cannot add/subtract days from TIME")
                        }
                        val dateIntervalTypes: Set<SqlTypeName> =
                            Sets.immutableEnumSet(
                                SqlTypeName.INTERVAL_YEAR_MONTH,
                                SqlTypeName.INTERVAL_YEAR,
                                SqlTypeName.INTERVAL_MONTH,
                                SqlTypeName.INTERVAL_DAY,
                            )
                        val isDateInterval =
                            dateIntervalTypes.contains(
                                fnOperation.getOperands()[1].type.sqlTypeName,
                            )
                        var arg0 = operands[0]
                        var arg1 = operands[1]
                        if (SqlTypeName.STRING_TYPES.contains(
                                fnOperation.getOperands()[0].type.sqlTypeName,
                            )
                        ) {
                            val inputType = fnOperation.getOperands()[0].type
                            val outputType = fnOperation.getType()
                            arg0 =
                                visitDynamicCast(
                                    operands[0],
                                    inputType,
                                    outputType,
                                    isSingleRow || IsScalar.isScalar(fnOperation.operands[0]),
                                )
                        }
                        if ((
                                isDateInterval &&
                                    DatetimeFnCodeGen.getDateTimeDataType(fnOperation.getOperands()[0])
                                    == DatetimeFnCodeGen.DateTimeType.DATE
                            )
                        ) {
                            if ((fnName == "SUBDATE") || (fnName == "DATE_SUB")) {
                                arg1 = bodoSQLKernel("negate", listOf(arg1))
                            }
                            return bodoSQLKernel(
                                "add_date_interval_to_date",
                                listOf(arg0, arg1),
                            )
                        }
                        return DateAddCodeGen.generateMySQLDateAddCode(arg0, arg1, manualAddition, fnName)
                    }

                    "DATEDIFF" -> {
                        val arg1: Expr
                        val arg2: Expr
                        unit = "DAY"
                        if (operands.size == 2) {
                            arg1 = operands[1]
                            arg2 = operands[0]
                            dateTimeExprType1 = DatetimeFnCodeGen.getDateTimeDataType(fnOperation.getOperands()[0])
                            dateTimeExprType2 = DatetimeFnCodeGen.getDateTimeDataType(fnOperation.getOperands()[1])
                        } else if (operands.size == 3) { // this is the Snowflake option
                            unit = operands[0].emit()
                            arg1 = operands[1]
                            arg2 = operands[2]
                            dateTimeExprType1 = DatetimeFnCodeGen.getDateTimeDataType(fnOperation.getOperands()[1])
                            dateTimeExprType2 = DatetimeFnCodeGen.getDateTimeDataType(fnOperation.getOperands()[2])
                        } else {
                            throw BodoSQLCodegenException(
                                "Invalid number of arguments to DATEDIFF: must be 2 or 3.",
                            )
                        }
                        if ((
                                (dateTimeExprType1 == DatetimeFnCodeGen.DateTimeType.TIME)
                                    != (dateTimeExprType2 == DatetimeFnCodeGen.DateTimeType.TIME)
                            )
                        ) {
                            throw BodoSQLCodegenException(
                                "Invalid type of arguments to DATEDIFF: cannot mix date/timestamp with time.",
                            )
                        }
                        unit = DatetimeFnCodeGen.standardizeTimeUnit(fnName, unit, dateTimeExprType1)
                        return DateDiffCodeGen.generateDateDiffFnInfo(unit, arg1, arg2)
                    }

                    "STR_TO_DATE" -> {
                        assert(operands.size == 2)
                        // Format string should be a string literal.
                        // This is required by the function definition.
                        if (fnOperation.operands[1] !is RexLiteral) {
                            throw BodoSQLCodegenException(
                                "Error STR_TO_DATE(): 'Format' must be a string literal",
                            )
                        }
                        return ConversionCodeGen.generateStrToDateCode(
                            operands[0],
                            IsScalar.isScalar(fnOperation.operands[0]),
                            operands[1].emit(),
                        )
                    }

                    "TIME_SLICE" -> return DatetimeFnCodeGen.generateTimeSliceFnCode(operands, 0)
                    "TIMESTAMP", "TO_ARRAY", "TO_BINARY", "TO_BOOLEAN",
                    "TO_DATE", "TO_DOUBLE", "TO_NUMBER", "TO_OBJECT",
                    "TO_TIME", "TO_TIMESTAMP", "TO_TIMESTAMP_LTZ",
                    "TO_TIMESTAMP_NTZ", "TO_TIMESTAMP_TZ", "TO_VARCHAR",
                    "TO_VARIANT", "TRY_TO_BINARY", "TRY_TO_BOOLEAN",
                    "TRY_TO_DATE", "TRY_TO_DOUBLE", "TRY_TO_NUMBER",
                    "TRY_TO_TIME", "TRY_TO_TIMESTAMP", "TRY_TO_TIMESTAMP_LTZ",
                    "TRY_TO_TIMESTAMP_NTZ", "TRY_TO_TIMESTAMP_TZ",
                    -> return visitCastFunc(
                        fnOperation,
                        operands,
                        argScalars,
                    )

                    "ACOS", "ACOSH", "ASIN", "ASINH", "ATAN", "ATAN2",
                    "ATANH", "COS", "COSH", "COT", "DEGREES", "RADIANS",
                    "SIN", "SINH", "TAN", "TANH",
                    -> return TrigCodeGen.getTrigFnCode(
                        fnName,
                        operands,
                    )

                    "ABS", "BITAND", "BITNOT", "BITOR", "BITSHIFTLEFT",
                    "BITSHIFTRIGHT", "BITXOR", "CBRT", "EXP", "FACTORIAL",
                    "GETBIT", "LN", "LOG10", "LOG2", "POW", "POWER",
                    "SIGN", "SQRT", "SQUARE",
                    -> return NumericCodeGen.getNumericFnCode(
                        fnName,
                        operands,
                    )

                    "TRUNC", "TRUNCATE" -> {
                        val args: MutableList<Expr> = ArrayList()
                        args.addAll(operands)
                        if (operands.size == 1) {
                            // If no value is specified by, default to 0
                            args.add(Expr.IntegerLiteral(0))
                        }
                        assert(args.size == 2)
                        return NumericCodeGen.getNumericFnCode(fnName, args)
                    }

                    "ROUND" -> {
                        val args: MutableList<Expr> = ArrayList()
                        args.addAll(operands)
                        if (operands.size == 1) {
                            // If no value is specified by, default to 0
                            args.add(Expr.IntegerLiteral(0))
                        }
                        assert(args.size == 2)
                        return NumericCodeGen.generateRoundCode(args)
                    }

                    "LOG" -> return NumericCodeGen.generateLogFnInfo(operands)
                    "CONV" -> {
                        assert(operands.size == 3)
                        return bodoSQLKernel("conv", operands, java.util.List.of())
                    }

                    "RAND" -> return Expr.Call("np.random.rand")
                    "PI" -> return Expr.Raw("np.pi")
                    "UNIFORM" -> {
                        assert(operands.size == 3)
                        Utils.expectScalarArgument(fnOperation.operands[0], "UNIFORM", "lo")
                        Utils.expectScalarArgument(fnOperation.operands[1], "UNIFORM", "hi")
                        return NumericCodeGen.generateUniformFnInfo(operands)
                    }

                    "CONCAT" -> return StringFnCodeGen.generateConcatCode(
                        operands,
                        listOf(),
                        fnOperation.operands[0].type,
                    )

                    "CONCAT_WS" -> {
                        assert(operands.size >= 2)
                        return StringFnCodeGen.generateConcatWSCode(
                            operands[0],
                            operands.subList(1, operands.size),
                            listOf(),
                        )
                    }

                    "GETDATE",
                    "CURRENT_TIME",
                    "UTC_TIMESTAMP",
                    "UTC_DATE",
                    "CURRENT_DATE",
                    "CURRENT_DATABASE",
                    "CURRENT_ACCOUNT",
                    "CURRENT_ACCOUNT_NAME",
                    -> {
                        assert(operands.isEmpty())
                        return visitGeneralContextFunction(fnOperation)
                    }

                    "MAKEDATE" -> {
                        assert(operands.size == 2)
                        return DatetimeFnCodeGen.generateMakeDateInfo(operands[0], operands[1])
                    }

                    "DATE_FORMAT" -> {
                        if (operands.size != 2 && IsScalar.isScalar(fnOperation.operands[1])) {
                            throw BodoSQLCodegenException(
                                "Error, invalid argument types passed to DATE_FORMAT",
                            )
                        }
                        if (fnOperation.operands[1] !is RexLiteral) {
                            throw BodoSQLCodegenException(
                                "Error DATE_FORMAT(): 'Format' must be a string literal",
                            )
                        }
                        return DatetimeFnCodeGen.generateDateFormatCode(operands[0], operands[1])
                    }

                    "COMBINE_INTERVALS" -> {
                        assert(operands.size == 2)
                        return DatetimeFnCodeGen.generateCombineIntervalsCode(operands)
                    }

                    "CONVERT_TIMEZONE" -> {
                        assert(operands.size == 2 || operands.size == 3)
                        return DatetimeFnCodeGen.generateConvertTimezoneCode(
                            operands,
                            fnOperation.getOperands(),
                            visitor.genDefaultTZ(),
                        )
                    }

                    "YEARWEEK" -> {
                        assert(operands.size == 1)
                        return DatetimeFnCodeGen.getYearWeekFnInfo(operands[0])
                    }

                    "MONTHS_BETWEEN" -> {
                        assert(operands.size == 2)
                        return bodoSQLKernel("months_between", operands)
                    }

                    "ADD_MONTHS" -> {
                        assert(operands.size == 2)
                        return bodoSQLKernel("add_months", operands)
                    }

                    "MONTHNAME", "MONTH_NAME", "DAYNAME", "WEEKDAY", "YEAROFWEEKISO" -> {
                        assert(operands.size == 1)
                        if (DatetimeFnCodeGen.getDateTimeDataType(fnOperation.getOperands()[0])
                            == DatetimeFnCodeGen.DateTimeType.TIME
                        ) {
                            throw BodoSQLCodegenException(
                                "Time object is not supported by $fnName",
                            )
                        }
                        return DatetimeFnCodeGen.getSingleArgDatetimeFnInfo(fnName, operands[0])
                    }

                    "YEAROFWEEK" -> {
                        assert(operands.size == 1)
                        val args = ArrayList<Expr>(operands)
                        args.add(Expr.IntegerLiteral((weekStart)!!))
                        args.add(Expr.IntegerLiteral((weekOfYearPolicy)!!))
                        return bodoSQLKernel("yearofweek", args)
                    }

                    "LAST_DAY" -> {
                        dateTimeExprType1 = DatetimeFnCodeGen.getDateTimeDataType(fnOperation.getOperands()[0])
                        if (dateTimeExprType1 == DatetimeFnCodeGen.DateTimeType.TIME) {
                            throw BodoSQLCodegenException(
                                "Time object is not supported by $fnName",
                            )
                        }
                        if (operands.size == 2) {
                            unit =
                                DatetimeFnCodeGen.standardizeTimeUnit(fnName, operands[1].emit(), dateTimeExprType1)
                            if ((unit == "day") || DatetimeFnCodeGen.TIME_PART_UNITS.contains(unit)) {
                                throw BodoSQLCodegenException(
                                    operands[1].emit() + " is not a valid time unit for " + fnName,
                                )
                            }
                            return DatetimeFnCodeGen.generateLastDayCode(operands[0], unit)
                        }
                        assert(operands.size == 1)
                        // the default time unit is month
                        return DatetimeFnCodeGen.generateLastDayCode(operands[0], "month")
                    }

                    "NEXT_DAY", "PREVIOUS_DAY" -> {
                        assert(operands.size == 2)
                        if (DatetimeFnCodeGen.getDateTimeDataType(fnOperation.getOperands()[0])
                            == DatetimeFnCodeGen.DateTimeType.TIME
                        ) {
                            throw BodoSQLCodegenException(
                                "Time object is not supported by $fnName",
                            )
                        }
                        return DatetimeFnCodeGen.getDoubleArgDatetimeFnInfo(fnName, operands[0], operands[1])
                    }

                    "TO_DAYS" -> return SinceEpochFnCodeGen.generateToDaysCode(operands[0])
                    "TO_SECONDS" -> return SinceEpochFnCodeGen.generateToSecondsCode(operands[0])
                    "FROM_DAYS" -> return SinceEpochFnCodeGen.generateFromDaysCode(operands[0])
                    "DATE_FROM_PARTS" -> {
                        assert(operands.size == 3)
                        return DatetimeFnCodeGen.generateDateTimeTypeFromPartsCode(
                            fnName,
                            operands,
                            Expr.None,
                        )
                    }

                    "TIME_FROM_PARTS",
                    "TIMESTAMP_FROM_PARTS",
                    "TIMESTAMP_NTZ_FROM_PARTS",
                    -> return DatetimeFnCodeGen.generateDateTimeTypeFromPartsCode(
                        fnName,
                        operands,
                        Expr.None,
                    )

                    "TIMESTAMP_TZ_FROM_PARTS",
                    "TIMESTAMP_LTZ_FROM_PARTS",
                    -> return DatetimeFnCodeGen.generateDateTimeTypeFromPartsCode(
                        fnName,
                        operands,
                        visitor.genDefaultTZ().zoneExpr,
                    )

                    "UNIX_TIMESTAMP" -> return SinceEpochFnCodeGen.generateUnixTimestamp()
                    "FROM_UNIXTIME" -> return SinceEpochFnCodeGen.generateFromUnixTimeCode(operands[0])
                    "GET_PATH", "JSON_EXTRACT_PATH_TEXT", "OBJECT_KEYS" -> return JsonCodeGen.visitJsonFunc(
                        fnName,
                        operands,
                        argScalars,
                    )

                    "OBJECT_DELETE" -> return visitObjectPickDelete(operands, false, argScalars)
                    "OBJECT_PICK" -> return visitObjectPickDelete(operands, true, argScalars)
                    "OBJECT_INSERT" -> return visitObjectInsert(operands, argScalars)
                    "IS_ARRAY", "IS_OBJECT" -> return visitVariantFunc(fnName, operands)
                    "REGEXP_LIKE", "REGEXP_COUNT", "REGEXP_REPLACE", "REGEXP_SUBSTR", "REGEXP_INSTR",
                    "ASCII", "CHAR", "CHAR_LENGTH", "LENGTH", "REVERSE", "LOWER", "UPPER", "SPACE",
                    "RTRIMMED_LENGTH", "FORMAT", "REPEAT", "STRCMP", "RIGHT", "LEFT", "CONTAINS", "INSTR",
                    "STARTSWITH", "ENDSWITH", "RPAD", "LPAD", "SPLIT_PART", "SUBSTRING_INDEX", "TRANSLATE3",
                    "REPLACE", "SUBSTRING", "INSERT", "CHARINDEX", "STRTOK", "STRTOK_TO_ARRAY", "SPLIT",
                    "EDITDISTANCE", "JAROWINKLER_SIMILARITY", "INITCAP", "SHA2", "MD5", "HEX_ENCODE",
                    "HEX_DECODE_STRING", "HEX_DECODE_BINARY", "TRY_HEX_DECODE_STRING", "TRY_HEX_DECODE_BINARY",
                    "BASE64_ENCODE", "BASE64_DECODE_STRING", "TRY_BASE64_DECODE_STRING", "BASE64_DECODE_BINARY",
                    "TRY_BASE64_DECODE_BINARY", "UUID_STRING",
                    -> return visitStringFunc(
                        fnOperation,
                        operands,
                        isSingleRow,
                    )

                    "DATE_TRUNC" -> {
                        dateTimeExprType1 = DatetimeFnCodeGen.getDateTimeDataType(fnOperation.getOperands()[1])
                        unit = DatetimeFnCodeGen.standardizeTimeUnit(fnName, operands[0].emit(), dateTimeExprType1)
                        return DatetimeFnCodeGen.generateDateTruncCode(unit, operands[1])
                    }

                    "MICROSECOND", "NANOSECOND", "SECOND", "MINUTE", "DAY", "DAYOFYEAR",
                    "DAYOFWEEK", "DAYOFWEEKISO", "DAYOFMONTH", "HOUR", "MONTH", "QUARTER",
                    "YEAR", "WEEK", "WEEKOFYEAR", "WEEKISO", "EPOCH_SECOND", "EPOCH_MILLISECOND",
                    "EPOCH_MICROSECOND", "EPOCH_NANOSECOND", "TIMEZONE_HOUR", "TIMEZONE_MINUTE",
                    -> {
                        isTime = fnOperation.getOperands()[0].type.sqlTypeName == SqlTypeName.TIME
                        isDate = fnOperation.getOperands()[0].type.sqlTypeName == SqlTypeName.DATE
                        return ExtractCodeGen.generateExtractCode(
                            fnName,
                            operands[0],
                            isTime,
                            isDate,
                            weekStart,
                            weekOfYearPolicy,
                        )
                    }

                    "REGR_VALX", "REGR_VALY" -> return CondOpCodeGen.getCondFuncCode(fnName, operands)
                    "ARRAYS_OVERLAP", "ARRAY_CAT", "ARRAY_COMPACT", "ARRAY_CONTAINS",
                    "ARRAY_EXCEPT", "ARRAY_INTERSECTION", "ARRAY_POSITION", "ARRAY_REMOVE",
                    "ARRAY_REMOVE_AT", "ARRAY_SIZE", "ARRAY_SLICE", "ARRAY_TO_STRING",
                    -> return visitNestedArrayFunc(
                        fnName,
                        operands,
                        argScalars,
                    )

                    "GET_IGNORE_CASE" -> return JsonCodeGen.visitGetIgnoreCaseOp(operands, argScalars)

                    "PARSE_JSON" -> throw BodoSQLCodegenException(
                        (
                            "Internal Error: PARSE_JSON currently only supported when it can be rewritten as" +
                                " ParseExtractCast sequence."
                        ),
                    )

                    "PARSE_URL" -> return bodoSQLKernel("parse_url", operands)
                }
                throw BodoSQLCodegenException(
                    "Internal Error: Function: " + fnOperation.operator.toString() + " not supported",
                )
            }

            else -> throw BodoSQLCodegenException(
                "Internal Error: Function: " + fnOperation.operator.toString() + " not supported",
            )
        }
    }

    override fun visitOver(over: RexOver): Expr {
        throw BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Unsupported RexOver: " + over.operator,
        )
    }

    override fun visitCorrelVariable(correlVariable: RexCorrelVariable): Expr {
        throw unsupportedNode("RexCorrelVariable")
    }

    override fun visitDynamicParam(dynamicParam: RexDynamicParam): Expr {
        val (paramName, actualType) =
            if (dynamicParam is RexNamedParam) {
                val name = dynamicParam.paramName
                val actualType =
                    if (namedParamTypeMap.contains(name)) {
                        namedParamTypeMap[name]!!
                    } else {
                        throw RuntimeException(
                            "Internal Error: Named parameter ${dynamicParam.paramName} is referenced in a query but not provided.",
                        )
                    }
                val paramBase = "_NAMED_PARAM_"
                val paramName = "$paramBase$name"
                Pair(paramName, actualType)
            } else {
                val index = dynamicParam.index
                val actualType =
                    if (index >= dynamicParamTypes.size) {
                        throw RuntimeException(
                            "Internal Error: At least ${index + 1} dynamic parameters are referenced in the query, " +
                                "but only ${dynamicParamTypes.size} were provided.",
                        )
                    } else {
                        dynamicParamTypes[index]
                    }
                val paramBase = "_DYNAMIC_PARAM_"
                val paramName = "$paramBase$index"
                Pair(paramName, actualType)
            }
        ctx.dynamicParams.add(paramName)
        val paramVariable = Variable(paramName)
        // Generate a cast if necessary
        return if (BodoSqlTypeUtil.literalEqualSansNullability(dynamicParam.type, actualType)) {
            paramVariable
        } else {
            visitDynamicCast(paramVariable, actualType, dynamicParam.type, true)
        }
    }

    override fun visitRangeRef(rangeRef: RexRangeRef): Expr {
        throw unsupportedNode("RexRangeRef")
    }

    override fun visitFieldAccess(fieldAccess: RexFieldAccess): Expr {
        throw unsupportedNode("RexFieldAccess")
    }

    override fun visitSubQuery(subQuery: RexSubQuery): Expr {
        throw unsupportedNode("RexSubQuery")
    }

    override fun visitTableInputRef(fieldRef: RexTableInputRef): Expr {
        return visitInputRef(fieldRef)
    }

    override fun visitPatternFieldRef(fieldRef: RexPatternFieldRef): Expr {
        return visitInputRef(fieldRef)
    }

    private fun unsupportedNode(nodeType: String): BodoSQLCodegenException {
        return BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Unsupported RexNode: $nodeType",
        )
    }

    /**
     * A version of the RexToBodoTranslator that is used when the expression is occurring in a
     * scalar context.
     */
    private class ScalarContext(
        visitor: BodoCodeGenVisitor,
        builder: Module.Builder,
        typeSystem: RelDataTypeSystem,
        input: BodoEngineTable?,
        dynamicParamTypes: List<RelDataType>,
        namedParamTypeMap: Map<String, RelDataType>,
        localRefs: List<Expr>,
        /** Variable names used in generated closures.  */
        private val closureVars: List<Variable>,
    ) :
        RexToBodoTranslator(visitor, builder, typeSystem, input, dynamicParamTypes, namedParamTypeMap, localRefs) {
        /**
         * List of functions generated by this scalar context. This is needed for nested case statements
         * to maintain control flow.
         */
        private val closures: MutableList<Op.Function> = java.util.ArrayList()

        override fun visitCastScan(
            operation: RexCall,
            isSafe: Boolean,
        ): Expr {
            return visitCastScan(operation, isSafe, true, listOf())
        }

        override fun isOperandScalar(operand: RexNode): Boolean {
            return true
        }

        /**
         * Generating code in a scalar context requires building a closure that will be "bubbled up" to
         * the original RexToBodoTranslator and then generating an expression that looks like var =
         * func(...).
         *
         *
         * This updates closures in and the builder to generate the function call.
         *
         * @return The final Variable for the output of the closure.
         */
        override fun visitCaseOp(node: RexCall): Expr {
            // Generate the frame for the closure.
            val closureFrame =
                visitCaseOperands(
                    this,
                    node.getOperands(),
                    listOf(builder.symbolTable.genGenericTempVar()),
                    true,
                )
            val funcVar = builder.symbolTable.genClosureVar()
            val closure = Op.Function(funcVar.emit(), closureVars, closureFrame)
            closures.add(closure)
            return Expr.Call(funcVar, closureVars)
        }

        override fun visitGeneralContextFunction(fnOperation: RexCall): Variable {
            // Case statements are not called consistently on all ranks, so we cannot
            // generate code that tries to make all ranks generate a consistent output.
            return visitGeneralContextFunction(fnOperation, false)
        }

        override fun visitGenericFuncOp(fnOperation: RexCall): Expr {
            return visitGenericFuncOp(fnOperation, true)
        }

        override fun visitOver(over: RexOver): Expr {
            throw BodoSQLCodegenException(
                "Internal Error: Calcite Plan Produced an Unsupported RexOver: " + over.operator,
            )
        }

        /** @return The closures generated by this scalar context.
         */
        fun getClosures(): List<Op.Function> {
            return closures
        }
    }

    companion object {
        /**
         * Constructs the Expression to make a call to the variadic function ARRAY_CONSTRUCT.
         *
         * @param codeExprs the Python expressions to calculate the arguments
         * @param argScalars Whether each argument is a scalar or a column
         * @return Expr containing the code generated for the relational expression.
         */
        @JvmStatic
        private fun visitArrayConstruct(
            codeExprs: List<Expr>,
            argScalars: List<Boolean>,
        ): Expr {
            val scalarExprs: ArrayList<Expr> = ArrayList()
            for (isScalar: Boolean in argScalars) {
                scalarExprs.add(Expr.BooleanLiteral((isScalar)))
            }
            return bodoSQLKernel(
                "array_construct",
                listOf(Expr.Tuple((codeExprs)), Expr.Tuple(scalarExprs)),
            )
        }

        @JvmStatic
        private fun visitVariantFunc(
            fnName: String,
            operands: List<Expr>,
        ): Expr {
            when (fnName) {
                "IS_ARRAY", "IS_OBJECT" -> return bodoSQLKernel(fnName.lowercase(), operands)
                else -> throw BodoSQLCodegenException("Internal Error: Function: " + fnName + "not supported")
            }
        }

        @JvmStatic
        private fun visitObjectInsert(
            codeExprs: List<Expr>,
            argScalars: List<Boolean>,
        ): Expr {
            // args: object, key, value[, update]

            // Convert codeExprs to a mutable type
            val args = ArrayList(codeExprs)
            // If update argument is missing, then default to false
            if (codeExprs.size != 4) {
                args.add(Expr.BooleanLiteral(false))
            }

            // Indicate whether the value is a scalar to distinguish between array types and vector.
            val kwargs = listOf(Pair("is_scalar", Expr.BooleanLiteral(argScalars[2])))
            return bodoSQLKernel("object_insert", args, kwargs)
        }
    }
}
