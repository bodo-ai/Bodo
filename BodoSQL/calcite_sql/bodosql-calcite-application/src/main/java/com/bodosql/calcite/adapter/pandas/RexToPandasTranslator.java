package com.bodosql.calcite.adapter.pandas;

import static com.bodosql.calcite.application.BodoSQLCodeGen.BinOpCodeGen.generateBinOpCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CastCodeGen.generateCastCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CastCodeGen.generateTryCastCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CondOpCodeGen.getDoubleArgCondFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CondOpCodeGen.getSingleArgCondFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CondOpCodeGen.visitIf;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CondOpCodeGen.visitVariadic;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ConversionCodeGen.generateStrToDateCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ConversionCodeGen.generateTimestampFnCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ConversionCodeGen.generateToBinaryFnCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ConversionCodeGen.generateToBooleanFnCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ConversionCodeGen.generateToCharFnCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ConversionCodeGen.generateToDateFnCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ConversionCodeGen.generateToDoubleFnCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ConversionCodeGen.generateToTimestampFnCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DateAddCodeGen.generateMySQLDateAddCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DateAddCodeGen.generateSnowflakeDateAddCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DateDiffCodeGen.generateDateDiffFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.DateTimeType;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.TIME_PART_UNITS;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateCurdateCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateCurrTimeCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateCurrTimestampCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateDateFormatCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateDateTimeTypeFromPartsCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateDateTruncCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateLastDayCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateMakeDateInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateToTimeCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateUTCDateCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateUTCTimestampCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.getDateTimeDataType;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.getDoubleArgDatetimeFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.getSingleArgDatetimeFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.getYearWeekFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.standardizeTimeUnit;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ExtractCodeGen.generateDatePart;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ExtractCodeGen.generateExtractCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.JsonCodeGen.generateJsonTwoArgsInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.generateConvCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.generateLeastGreatestCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.generateLogFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.generateToNumberCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.generateTryToNumberCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.getDoubleArgNumericFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.getSingleArgNumericFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.PostfixOpCodeGen.generatePostfixOpCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.PrefixOpCodeGen.generatePrefixOpCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.RegexpCodeGen.generateRegexpCountInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.RegexpCodeGen.generateRegexpInstrInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.RegexpCodeGen.generateRegexpLikeInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.RegexpCodeGen.generateRegexpReplaceInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.RegexpCodeGen.generateRegexpSubstrInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SinceEpochFnCodeGen.generateFromDaysCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SinceEpochFnCodeGen.generateFromUnixTimeCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SinceEpochFnCodeGen.generateToDaysCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SinceEpochFnCodeGen.generateToSecondsCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SinceEpochFnCodeGen.generateUnixTimestamp;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateConcatFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateConcatWSFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateEditdistance;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateInitcapInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateInsert;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generatePosition;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateStrtok;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateSubstringInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateTrimFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.getSingleArgStringFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.getThreeArgStringFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.getTwoArgStringFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.TrigCodeGen.getDoubleArgTrigFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.TrigCodeGen.getSingleArgTrigFnInfo;
import static com.bodosql.calcite.application.Utils.BodoArrayHelpers.sqlTypeToBodoArrayType;
import static com.bodosql.calcite.application.Utils.Utils.generateCombinedDf;
import static com.bodosql.calcite.application.Utils.Utils.isWindowedAggFn;
import static com.bodosql.calcite.application.Utils.Utils.renameTableRef;

import com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen;
import com.bodosql.calcite.application.BodoSQLCodeGen.LiteralCodeGen;
import com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen;
import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.application.ExprTypeVisitor;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.application.Utils.BodoCtx;
import com.bodosql.calcite.ir.Dataframe;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.FrameTripleQuotedString;
import com.bodosql.calcite.ir.Frame;
import com.bodosql.calcite.ir.Module;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.Op.Assign;
import com.bodosql.calcite.ir.Op.If;
import com.bodosql.calcite.ir.Variable;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexCorrelVariable;
import org.apache.calcite.rex.RexDynamicParam;
import org.apache.calcite.rex.RexFieldAccess;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexLocalRef;
import org.apache.calcite.rex.RexNamedParam;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexOver;
import org.apache.calcite.rex.RexPatternFieldRef;
import org.apache.calcite.rex.RexRangeRef;
import org.apache.calcite.rex.RexSubQuery;
import org.apache.calcite.rex.RexTableInputRef;
import org.apache.calcite.rex.RexVisitor;
import org.apache.calcite.sql.SqlBinaryOperator;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlInternalOperator;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNullTreatmentOperator;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlPostfixOperator;
import org.apache.calcite.sql.SqlPrefixOperator;
import org.apache.calcite.sql.fun.BodoSqlTryCastFunction;
import org.apache.calcite.sql.fun.SqlCaseOperator;
import org.apache.calcite.sql.fun.SqlCastFunction;
import org.apache.calcite.sql.fun.SqlDatetimePlusOperator;
import org.apache.calcite.sql.fun.SqlDatetimeSubtractionOperator;
import org.apache.calcite.sql.fun.SqlExtractFunction;
import org.apache.calcite.sql.fun.SqlLikeOperator;
import org.apache.calcite.sql.fun.SqlSubstringFunction;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.TZAwareSqlType;
import org.apache.calcite.util.Sarg;
import org.jetbrains.annotations.NotNull;

/** Translates a RexNode into a Pandas expression. */
public class RexToPandasTranslator implements RexVisitor<Expr> {
  // Don't really want this here, but it's easier than trying to move all
  // of its functionality into the builder immediately.
  @NotNull protected final PandasCodeGenVisitor visitor;
  @NotNull protected final Module.Builder builder;
  @NotNull protected final RelDataTypeSystem typeSystem;
  protected final int nodeId;
  @NotNull protected final Dataframe input;

  @NotNull public final BodoCtx ctx;

  public RexToPandasTranslator(
      @NotNull PandasCodeGenVisitor visitor,
      @NotNull Module.Builder builder,
      @NotNull RelDataTypeSystem typeSystem,
      int nodeId,
      @NotNull Dataframe input) {
    this.visitor = visitor;
    this.builder = builder;
    this.typeSystem = typeSystem;
    this.nodeId = nodeId;
    this.input = input;
    this.ctx = new BodoCtx();
  }

  public @NotNull Dataframe getInput() {
    return input;
  }

  @Override
  public Expr visitInputRef(RexInputRef inputRef) {
    return new Expr.Call(
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data",
        List.of(input.getVariable(), new Expr.IntegerLiteral(inputRef.getIndex())));
  }

  @Override
  public Expr visitLocalRef(RexLocalRef localRef) {
    throw unsupportedNode();
  }

  @Override
  public Expr visitLiteral(RexLiteral literal) {
    return LiteralCodeGen.generateLiteralCode(literal, false, visitor);
  }

  @Override
  public Expr visitCall(RexCall call) {
    // TODO(jsternberg): Using instanceof here is problematic.
    // It would be better to use getKind(). Revisit this later.
    if (call.getOperator() instanceof SqlNullTreatmentOperator) {
      return visitNullTreatmentOp(call);
    } else if (call.getOperator() instanceof SqlBinaryOperator
        || call.getOperator() instanceof SqlDatetimePlusOperator
        || call.getOperator() instanceof SqlDatetimeSubtractionOperator) {
      return visitBinOpScan(call);
    } else if (call.getOperator() instanceof SqlPostfixOperator) {
      return visitPostfixOpScan(call);
    } else if (call.getOperator() instanceof SqlPrefixOperator) {
      return visitPrefixOpScan(call);
    } else if (call.getOperator() instanceof SqlInternalOperator) {
      return visitInternalOp(call);
    } else if (call.getOperator() instanceof SqlLikeOperator) {
      return visitLikeOp(call);
    } else if (call.getOperator() instanceof SqlCaseOperator) {
      return visitCaseOp(call);
    } else if (call.getOperator() instanceof SqlCastFunction) {
      return visitCastScan(call);
    } else if (call.getOperator() instanceof BodoSqlTryCastFunction) {
      return visitTryCastScan(call);
    } else if (call.getOperator() instanceof SqlExtractFunction) {
      return visitExtractScan(call);
    } else if (call.getOperator() instanceof SqlSubstringFunction) {
      return visitSubstringScan(call);
    } else if (call.getOperator() instanceof SqlFunction) {
      return visitGenericFuncOp(call);
    } else {
      throw new BodoSQLCodegenException(
          "Internal Error: Calcite Plan Produced an Unsupported RexCall:" + call.getOperator());
    }
  }

  /**
   * Visitor for RexCalls IGNORE NULLS and RESPECT NULLS This function is only called if IGNORE
   * NULLS and RESPECT NULLS is called without an associated window. Otherwise, it is included as a
   * field in the REX OVER node.
   *
   * <p>Currently, we always throw an error when entering this call. Frankly, based on my reading of
   * calcite's syntax, we only reach this node through invalid syntax in Calcite (LEAD/LAG
   * RESPECT/IGNORE NULL's without a window)
   *
   * @param node RexCall being visited
   * @return Expr containing the new column name and the code generated for the relational
   *     expression.
   */
  private Expr visitNullTreatmentOp(RexCall node) {
    SqlKind innerCallKind = node.getOperands().get(0).getKind();
    switch (innerCallKind) {
      case LEAD:
      case LAG:
      case NTH_VALUE:
      case FIRST_VALUE:
      case LAST_VALUE:
        throw new BodoSQLCodegenException(
            "Error during codegen: " + innerCallKind.toString() + " requires OVER clause.");
      default:
        throw new BodoSQLCodegenException(
            "Error during codegen: Unreachable code entered while evaluating the following rex"
                + " node in visitNullTreatmentOp: "
                + node.toString());
    }
  }

  private Expr visitBinOpScan(RexCall operation) {
    List<Expr> args = new ArrayList<>();
    List<BodoSQLExprType.ExprType> exprTypes = new ArrayList<>();
    SqlOperator binOp = operation.getOperator();
    // Store the argument types for TZ-Aware data
    List<RelDataType> argDataTypes = new ArrayList<>();
    for (RexNode operand : operation.operands) {
      Expr exprCode = operand.accept(this);
      args.add(exprCode);
      exprTypes.add(visitor.exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(operand, nodeId)));
      argDataTypes.add(operand.getType());
    }
    if (binOp.getKind() == SqlKind.OTHER && binOp.getName().equals("||")) {
      // Support the concat operator by using the concat array kernel.
      return StringFnCodeGen.generateConcatFnInfo(args);
    }
    return generateBinOpCode(args, binOp, argDataTypes);
  }

  private Expr visitPostfixOpScan(RexCall operation) {
    List<Expr> args = visitList(operation.operands);
    Expr seriesOp = args.get(0);
    String codeExpr = generatePostfixOpCode(seriesOp.emit(), operation.getOperator());
    return new Expr.Raw(codeExpr);
  }

  private Expr visitPrefixOpScan(RexCall operation) {
    List<Expr> args = visitList(operation.operands);
    Expr seriesOp = args.get(0);
    String codeExpr = generatePrefixOpCode(seriesOp.emit(), operation.getOperator());
    return new Expr.Raw(codeExpr);
  }

  protected Expr visitInternalOp(RexCall node) {
    return visitInternalOp(node, false);
  }

  protected Expr visitInternalOp(RexCall node, boolean isSingleRow) {
    SqlKind sqlOp = node.getOperator().getKind();
    switch (sqlOp) {
        /* TODO(Ritwika): investigate more possible internal operations as result of optimization rules*/
      case SEARCH:

        // Determine if we can use the optimized is_in codepath. We can take this codepath if
        // the second argument (the elements to search for) consists of exclusively discrete values,
        // and
        // we are not inside a case statement.
        // We can't use the optimized implementation within a case statement due to the Sarg array
        // being
        // lowered as a global, and we can't lower globals within a case statement, as
        // bodosql_case_placeholder
        // doesn't have the doesn't have the same global state as
        // the rest of the main generated code
        SqlTypeName search_val_type = node.getOperands().get(1).getType().getSqlTypeName();
        // TODO: add testing/support for sql types other than string/int:
        // https://bodo.atlassian.net/browse/BE-4046
        // TODO: allow lowering globals within case statements in BodoSQL:
        //
        boolean can_use_isin_codegen =
            !isSingleRow
                && (SqlTypeName.STRING_TYPES.contains(search_val_type)
                    || SqlTypeName.INT_TYPES.contains(search_val_type));

        RexLiteral sargNode = (RexLiteral) node.getOperands().get(1);
        Sarg sargVal = (Sarg) sargNode.getValue();
        Iterator<Range> iter = sargVal.rangeSet.asRanges().iterator();
        // We expect the range to have at least one value,
        // otherwise, this search should have been optimized out
        assert iter.hasNext() : "Internal Error: search Sarg literal had no elements";
        while (iter.hasNext() && can_use_isin_codegen) {
          Range curRange = iter.next();
          // Assert that each element of the range is scalar.
          if (!(curRange.hasLowerBound()
              && curRange.hasUpperBound()
              && curRange.upperEndpoint() == curRange.lowerEndpoint())) {
            can_use_isin_codegen = false;
          }
        }

        if (can_use_isin_codegen) {
          // use the isin array kernel in the case
          // that the second argument does consists of
          // exclusively discrete values
          List<Expr> args = visitList(node.operands);
          return new Expr.Raw(
              "bodo.libs.bodosql_array_kernels.is_in("
                  + args.get(0).emit()
                  + ", "
                  + args.get(1).emit()
                  + ")");
        } else {
          // Fallback to generating individual checks
          // in the case that the second argument does not consist of
          // exclusively discrete values

          // Lookup the expanded nodes previously generated
          RexNode searchNode =
              visitor.searchMap.get(ExprTypeVisitor.generateRexNodeKey(node, nodeId));
          return searchNode.accept(this);
        }
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Internal Operator");
    }
  }

  private Expr visitLikeOp(RexCall node) {
    // The input node has ${index} as its first operand, where
    // ${index} is something like $3, and a SQL regular expression
    // as its second operand. If there is an escape value it will
    // be the third value, although it is not required and only supported
    // for LIKE and ILIKE
    SqlLikeOperator op = (SqlLikeOperator) node.getOperator();
    List<RexNode> operands = node.getOperands();
    RexNode patternNode = operands.get(1);

    // The regular expression functions only support literal patterns
    boolean patternRegex = false;
    if (op.getKind() == SqlKind.REGEXP || op.getKind() == SqlKind.RLIKE) {
      if (!(patternNode instanceof RexLiteral)) {
        throw new BodoSQLCodegenException(
            String.format("%s Error: Pattern must be a string literal", op.getName()));
      }
      patternRegex = true;
    }

    Expr arg = operands.get(0).accept(this);
    Expr pattern = patternNode.accept(this);
    Expr escape;
    if (operands.size() == 3) {
      escape = operands.get(2).accept(this);
    } else {
      escape = new Expr.StringLiteral("");
    }

    if (patternRegex) {
      return new Expr.Call(
          "bodo.libs.bodosql_array_kernels.regexp_like", arg, pattern, new Expr.StringLiteral(""));
    } else {
      return new Expr.Call(
          "bodo.libs.bodosql_array_kernels.like_kernel",
          arg,
          pattern,
          escape,
          // Use the opposite. The python call is for case insensitivity while
          // our boolean is for case sensitivity so they are opposites.
          new Expr.BooleanLiteral(!op.isCaseSensitive()));
    }
  }

  protected Expr visitCaseOp(RexCall node) {
    return visitCaseOp(node, false);
  }

  /**
   * Overview of code generation for case:
   * https://bodo.atlassian.net/wiki/spaces/B/pages/1368752135/WIP+BodoSQL+Case+Implementation
   *
   * @param node The case node to visit
   * @param isSingleRow Is case a nested inside another case statement?
   * @return The resulting expression from generating case code.
   */
  protected Expr visitCaseOp(RexCall node, boolean isSingleRow) {
    // TODO: Technical debt, this should be done in our fork of calcite
    // Calcite optimizes a large number of windowed aggregation functions into case statements,
    // which check if the window is valid. This can be during the parsing step by setting the
    // "allowPartial" variable to be true.

    if (isWindowedAggFn(node)) {
      return node.getOperands().get(1).accept(this);
    }

    List<RexNode> operands = node.getOperands();
    // Even if the contents are scalars, we will always have a dataframe unless we
    // are inside another case statement. We opt to use the case kernel to avoid
    // the compilation overhead for very large case statements (the kernel
    // can make the inlining decision).
    boolean callCaseKernel = !isSingleRow;

    if (callCaseKernel) {
      // If we call the case kernel we need to generate a new frame for
      // writing the values that will be inserted into the loop body.
      // We create the loop body as a global to allow the kernel to make
      // decisions about inlining.
      builder.startFrame();
    }
    /** Create the translator for visiting the operands. */
    RexToPandasTranslator localTranslator =
        new RexToPandasTranslator.ScalarContext(visitor, builder, typeSystem, nodeId, input);
    /** Generate the code for visiting the operands to case. */
    Variable outputVar = visitCaseOperands(localTranslator, operands);
    if (callCaseKernel) {
      RelDataType outputType = node.getType();
      return generateCaseKernelCall(localTranslator.ctx, outputVar, outputType);
    } else {
      // If we're not the top level kernel, so we need to pass back the information so that it is
      // properly handled by the actual kernel.
      ctx.unionContext(localTranslator.ctx);
      // Just return the output variable because there is no kernel.
      return outputVar;
    }
  }

  /**
   * Visit the operands to a call to case using the given translator. These operands are traversed
   * in reverse order because each operand should generate its code into a different Python frame
   * and its simplest to generate the code this way.
   *
   * <p>The operands to case are of the form: [cond1, truepath1, cond2, truepath2, ..., elsepath]
   * All of these components are present, so the operands is always of Length 2n + 1, where n is the
   * number of conditions and n > 0. The else is always present, even if not explicit in the
   * original SQL query.
   *
   * <p>For code generation purposes we first process the else and then we process each pair of
   * condition + truepath. The final code will place each truepath in the `if` block when evaluating
   * each condition and the next condition in its else block.
   *
   * <p>All "result" paths are defined to use the same output Variable, which is returned by this
   * function.
   *
   * <p>For example, if we have 5 inputs with no intermediate variables, the generated code might
   * look like this:
   *
   * <p><code>
   *     if bodo.libs.bodosql_array_kernels.is_true(cond1):
   *        output_var = truepath1
   *     else:
   *        if bodo.libs.bodosql_array_kernels.is_true(cond2):
   *          output_var = truepath2
   *        else:
   *          output_var = elsepath
   * </code>
   *
   * @param translator The translator used to visit each operand.
   * @param operands The list of RexNodes to visit to capture the proper computation.
   * @return The output variable written to in all paths.
   */
  private Variable visitCaseOperands(RexToPandasTranslator translator, List<RexNode> operands) {
    // Create the output variable shared by all nodes
    Variable outputVar = visitor.genGenericTempVar();
    // Generate the frames we will use. Each operand is it a different, frame,
    // including the current frame for the 0th operand.
    for (int i = 0; i < operands.size() - 1; i++) {
      builder.startFrame();
    }
    // Generate the else code
    Expr elsePath = operands.get(operands.size() - 1).accept(translator);
    // Assign to the output variable
    builder.add(new Op.Assign(outputVar, elsePath));
    for (int i = operands.size() - 2; i > 0; i -= 2) {
      // Pop the else Frame
      Frame elseFrame = builder.endFrame();
      // Generate the if path frame + code
      Expr ifPath = operands.get(i).accept(translator);
      // Assign to the output variable
      builder.add(new Op.Assign(outputVar, ifPath));
      // Pop the two frames for the condition
      Frame ifFrame = builder.endFrame();
      // Visit the cond code
      Expr cond = operands.get(i - 1).accept(translator);
      // Wrap the cond expr in a is_true call
      Expr.Call condCall = new Expr.Call("bodo.libs.bodosql_array_kernels.is_true", List.of(cond));
      // Generate the if statement
      Op.If ifStatement = new If(condCall, ifFrame, elseFrame);
      builder.add(ifStatement);
    }
    return outputVar;
  }

  /**
   * Prepare the arguments and generate the call to `bodo.utils.typing.bodosql_case_placeholder` for
   * the main case kernel. There are additional complexities associated with window functions, so
   * some arguments make modifications to existing code, which are highlighted in the helper
   * functions.
   *
   * <p>This function returns a Variable which holds the result of kernel call and appends the call
   * directly to the generated Code for the active Frame.
   *
   * @param translatorContext The BodoContext generated from visiting the operands. This tracks
   *     information like which columns where used.
   * @param loopOutputVar The output variable for the result in the loop body.
   * @param outputType The RelDataType of the final output.
   * @return The variable that is assigned to the output of the kernel.
   */
  private Variable generateCaseKernelCall(
      BodoCtx translatorContext, Variable loopOutputVar, RelDataType outputType) {
    // Pop off the active Frame for generating a triple quote
    // string global.
    Frame loopBodyFrame = builder.endFrame();
    Variable inputVar = getCaseInputVar(translatorContext);

    // Relevant steps for generating the kernel call:
    //
    // 1) a tuple of necessary input arrays
    // 2) (global constant) initialization code for unpacking the input array tuple with the correct
    // array
    // 3) (global constant) body of the CASE loop
    // 4) Generated named parameters. This is only used when Named parameters (BodoSQL's
    // SQL reference to Python variables) are used. This is very uncommon!
    // 5) number of output rows (same as input rows, needed for allocation)
    // 6) loop variable name
    // 7) output array type
    // 8) Generate the final call

    // Sort used columns so we always get the same code. This is used by steps
    // 1 and 2.
    TreeSet<Integer> sortedUsedColumns = new TreeSet<>(translatorContext.getUsedColumns());
    // Step 1: Create a tuple of arrays
    Expr.Tuple inputData = generateCaseArrayTuple(inputVar, sortedUsedColumns);
    // Step 2: Create the initialization code + constant
    Variable initGlobal = generateCaseInitializationCode(inputVar, sortedUsedColumns);
    // Step 3: Generate the loop body constant
    Variable bodyGlobal = generateCaseLoopBody(inputVar, loopBodyFrame);
    // Step 4: Generate the namedArguments to the bodosql_case_placeholder call.
    // This is only used if we have named Parameters, so in most cases it will be empty.
    List<kotlin.Pair<String, Expr>> namedArgs = generateCaseNamedArgs(translatorContext);
    // Step 5: Generate the output size
    Expr.Call lenCall = new Expr.Call("len", List.of(inputVar));
    // Step 6: Generate the array variable
    Variable arrVar = visitor.genGenericTempVar();
    // Step 7: Create the type
    Variable outputArrayTypeGlobal =
        visitor.lowerAsGlobal(sqlTypeToBodoArrayType(outputType, false));

    // Step 8: Generate the function call
    Expr.Call functionCall =
        new Expr.Call(
            "bodo.utils.typing.bodosql_case_placeholder",
            List.of(
                inputData,
                lenCall,
                initGlobal,
                bodyGlobal,
                // Note: The variable name must be a string literal here.
                new Expr.StringLiteral(loopOutputVar.getName()),
                outputArrayTypeGlobal),
            namedArgs);
    // Add the call to the generated code.
    builder.add(new Op.Assign(arrVar, functionCall));
    // Update the output variable
    return arrVar;
  }

  /**
   * Determine the input variable for function calls. If we need to generate additional columns
   * (which can occur if a RexOver is used in the body of a case statement), then we need to
   * generate a new variable that will be added to the code with additional columns.
   *
   * @param translatorContext The BodoContext holding what columns need to be added.
   * @return The input variable being selected.
   */
  private Variable getCaseInputVar(BodoCtx translatorContext) {
    if (translatorContext.getColsToAddList().size() > 0) {
      Variable newInputVar = builder.getSymbolTable().genDfVar();
      // Generate the new variable in the code
      List<String> colNames = input.getRel().getRowType().getFieldNames();
      Variable prevInputVar = input.getVariable();
      Op.Assign newDataFrame =
          new Assign(
              newInputVar,
              generateCombinedDf(prevInputVar, colNames, translatorContext.getColsToAddList()));
      // Add the new variable to the generated code.
      this.builder.add(newDataFrame);
      return newInputVar;
    } else {
      return input.getVariable();
    }
  }

  /**
   * Generate the tuple of arrays to feed as input to `bodo.utils.typing.bodosql_case_placeholder`
   *
   * @param inputVar The input DataFrame variable from which to extract the arrays.
   * @param sortedUsedColumns A sorted Set of columns indicated which columns to select.
   * @return The Expr.Tuple of arguments.
   */
  private Expr.Tuple generateCaseArrayTuple(Variable inputVar, TreeSet<Integer> sortedUsedColumns) {
    List<Expr.Call> inputDataArgs = new ArrayList<>();
    // Create the tuples variable
    for (int colNo : sortedUsedColumns) {
      inputDataArgs.add(
          new Expr.Call(
              "bodo.hiframes.pd_dataframe_ext.get_dataframe_data",
              List.of(inputVar, new Expr.IntegerLiteral(colNo))));
    }
    return new Expr.Tuple(inputDataArgs);
  }

  /**
   * Generates the initial to be used inside `bodo.utils.typing.bodosql_case_placeholder` before
   * loop generation. This unpacks the tuple of arrays into variable names that we assumed when we
   * visited the operands.
   *
   * <p>This data is then converted to a triple quoted string and passed to the globals via the
   * MetaType (for reduced compliation time). That global variable is returned.
   *
   * @param inputVar The input variable which is needed for generating the variable names.
   * @param sortedUsedColumns A sorted Set of columns indicated which columns to select.
   * @return The generated global variable.
   */
  private Variable generateCaseInitializationCode(
      Variable inputVar, TreeSet<Integer> sortedUsedColumns) {
    // Create a new frame for appending the initialization variable
    builder.startFrame();
    // Create the tuples variable
    Variable arrs = new Variable("arrs");
    Iterator<Integer> usedCols = sortedUsedColumns.iterator();
    for (int i = 0; i < sortedUsedColumns.size(); i++) {
      // Note: This exact variable name is required/assumed in the code generation.
      Op.Assign initLine =
          new Assign(
              new Variable(inputVar.getName() + "_" + usedCols.next()),
              new Expr.Getitem(arrs, new Expr.IntegerLiteral(i)));
      builder.add(initLine);
    }
    // Pop the frame and generate the triple quoted string
    Frame initFrame = builder.endFrame();
    Expr.FrameTripleQuotedString initBody = new FrameTripleQuotedString(initFrame, 1);
    // Create the global variable.
    return visitor.lowerAsMetaType(initBody);
  }

  /**
   * Generate the loop body code as a global variable from the frame used when visiting the
   * operands. If we needed to generate a new input variable (which can occur if a RexOver is used
   * in the body of a case statement), we will need to update this loop body with the new input
   * variable name. This is "hacky" and should be removed once we can guarentee not having RexOver
   * inside case at a plan level.
   *
   * @param inputVar The input variable that should be used in the loopBody. If this isn't used we
   *     need to update the contents of the body.
   * @param loopBodyFrame The Frame where the operands were initially visited. This contains the
   *     code for the loop body.
   * @return The generated global variable.
   */
  private Variable generateCaseLoopBody(Variable inputVar, Frame loopBodyFrame) {
    // Generate the triple quoted string
    // Note: We use indent level 2 because this is found inside a for loop
    Expr loopBody = new FrameTripleQuotedString(loopBodyFrame, 2);
    Variable prevInputVar = input.getVariable();
    if (!prevInputVar.equals(inputVar)) {
      // Update the loop body to replace any uses of the prevInputVar with the new inputVar.
      // TODO: Remove
      String loopBodyStr = loopBody.emit();
      loopBody = new Expr.Raw(renameTableRef(loopBodyStr, prevInputVar, inputVar));
    }
    return visitor.lowerAsGlobal(loopBody);
  }

  /**
   * Generate a list suitable to pass as the named arguments to an Expr.Call expression for the
   * named Parameters that were used inside the body of the loop. Named parameters are not standard
   * SQL, equating to SQL variables in other languages, so in most cases this will return an empty
   * list.
   *
   * @param translatorContext The context that tracks any used named parameters
   * @return Thes list suitable to pass to the Expr.Call
   */
  private List<kotlin.Pair<String, Expr>> generateCaseNamedArgs(BodoCtx translatorContext) {
    TreeSet<String> sortedParamSet = new TreeSet<>(translatorContext.getNamedParams());
    List<kotlin.Pair<String, Expr>> namedArgs = new ArrayList<>();
    for (String param : sortedParamSet) {
      namedArgs.add(new kotlin.Pair<>(param, new Expr.Raw(param)));
    }
    return namedArgs;
  }

  protected Expr visitCastScan(RexCall operation) {
    RelDataType inputType = operation.operands.get(0).getType();
    RelDataType outputType = operation.getType();

    boolean outputScalar =
        visitor.exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(operation, nodeId))
            == BodoSQLExprType.ExprType.SCALAR;
    List<Expr> args = visitList(operation.operands);
    Expr child = args.get(0);
    String exprCode = generateCastCode(child.emit(), inputType, outputType, outputScalar);
    return new Expr.Raw(exprCode);
  }

  protected Expr visitTryCastScan(RexCall operation) {
    RelDataType inputType = operation.operands.get(0).getType();
    if (!SqlTypeName.CHAR_TYPES.contains(inputType.getSqlTypeName()))
      throw new BodoSQLCodegenException("TRY_CAST only supports casting from strings.");
    RelDataType outputType = operation.getType();

    List<Expr> args = visitList(operation.operands);
    Expr child = args.get(0);
    String exprCode = generateTryCastCode(child.emit(), outputType);
    return new Expr.Raw(exprCode);
  }

  private Expr visitExtractScan(RexCall node) {
    List<Expr> args = visitList(node.operands);
    boolean isTime = node.operands.get(1).getType().getSqlTypeName().toString().equals("TIME");
    boolean isDate = node.operands.get(1).getType().getSqlTypeName().toString().equals("DATE");
    Expr dateVal = args.get(0);
    Expr column = args.get(1);
    String codeExpr = generateExtractCode(dateVal.emit(), column.emit(), isTime, isDate);
    return new Expr.Raw(codeExpr);
  }

  private Expr visitSubstringScan(RexCall node) {
    // node.operands contains
    //  * String to perform the substring operation on
    //  * start index
    //  * substring length (optional)
    //  All of these values can be both scalars and columns
    // NOTE: check on number of arguments happen in generateSubstringInfo
    List<Expr> operands = visitList(node.operands);
    return generateSubstringInfo(operands);
  }

  protected Expr visitGenericFuncOp(RexCall fnOperation) {
    return visitGenericFuncOp(fnOperation, false);
  }

  protected Expr visitGenericFuncOp(RexCall fnOperation, boolean isSingleRow) {
    String fnName = fnOperation.getOperator().toString();

    List<BodoSQLExprType.ExprType> exprTypes = new ArrayList<>();
    for (RexNode node : fnOperation.getOperands()) {
      exprTypes.add(visitor.exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(node, nodeId)));
    }

    // Handle IF, COALESCE, DECODE and their variants separately
    if (fnName == "COALESCE"
        || fnName == "NVL"
        || fnName == "NVL2"
        || fnName == "BOOLAND"
        || fnName == "BOOLOR"
        || fnName == "BOOLXOR"
        || fnName == "BOOLNOT"
        || fnName == "EQUAL_NULL"
        || fnName == "ZEROIFNULL"
        || fnName == "IFNULL"
        || fnName == "IF"
        || fnName == "IFF"
        || fnName == "DECODE") {
      List<String> codeExprs = new ArrayList<>();
      BodoCtx localCtx = new BodoCtx();
      int j = 0;
      for (RexNode operand : fnOperation.operands) {
        Expr operandInfo = operand.accept(this);
        String expr = operandInfo.emit();
        // Need to unbox scalar timestamp values.
        if (isSingleRow || (exprTypes.get(j) == BodoSQLExprType.ExprType.SCALAR)) {
          expr = "bodo.utils.conversion.unbox_if_tz_naive_timestamp(" + expr + ")";
        }
        codeExprs.add(expr);
        j++;
      }

      Expr result;
      switch (fnName) {
        case "IF":
        case "IFF":
          result = visitIf(fnOperation, codeExprs);
          break;
        case "BOOLNOT":
          result = getSingleArgCondFnInfo(fnName, codeExprs.get(0));
          break;
        case "BOOLAND":
        case "BOOLOR":
        case "BOOLXOR":
        case "EQUAL_NULL":
          result = getDoubleArgCondFnInfo(fnName, codeExprs.get(0), codeExprs.get(1));
          break;
        case "COALESCE":
        case "ZEROIFNULL":
        case "IFNULL":
        case "NVL":
        case "NVL2":
        case "DECODE":
          result = visitVariadic(fnOperation, codeExprs);
          break;
        default:
          throw new BodoSQLCodegenException("Internal Error: reached unreachable code");
      }

      // If we're not the top level apply, we need to pass back the information so that it is
      // properly handled by the actual top level apply
      ctx.unionContext(localCtx);
      return result;
    }

    // Extract all inputs to the current function.
    List<Expr> operands = visitList(fnOperation.operands);

    String expr;
    String strExpr;
    DatetimeFnCodeGen.DateTimeType dateTimeExprType1;
    DatetimeFnCodeGen.DateTimeType dateTimeExprType2;
    boolean isTime;
    boolean isDate;
    String unit;
    String tzStr;
    switch (fnOperation.getOperator().kind) {
      case CEIL:
      case FLOOR:
        return new Expr.Raw(
            getSingleArgNumericFnInfo(
                fnOperation.getOperator().toString(), operands.get(0).emit()));
      case GREATEST:
      case LEAST:
        return generateLeastGreatestCode(fnOperation.getOperator().toString(), operands);

      case MOD:
        return new Expr.Raw(
            getDoubleArgNumericFnInfo(
                fnOperation.getOperator().toString(),
                operands.get(0).emit(),
                operands.get(1).emit()));
      case TIMESTAMP_ADD:
        // Uses Calcite parser, accepts both quoted and unquoted time units
        dateTimeExprType1 = getDateTimeDataType(fnOperation.getOperands().get(2));
        unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType1);
        assert exprTypes.get(0) == BodoSQLExprType.ExprType.SCALAR;
        return new Expr.Raw(generateSnowflakeDateAddCode(operands, unit));
      case TIMESTAMP_DIFF:
        assert operands.size() == 3;
        dateTimeExprType1 = getDateTimeDataType(fnOperation.getOperands().get(1));
        dateTimeExprType2 = getDateTimeDataType(fnOperation.getOperands().get(2));
        if ((dateTimeExprType1 == DatetimeFnCodeGen.DateTimeType.TIME)
            != (dateTimeExprType2 == DatetimeFnCodeGen.DateTimeType.TIME)) {
          throw new BodoSQLCodegenException(
              "Invalid type of arguments to TIMESTAMPDIFF: cannot mix date/timestamp with time.");
        }
        unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType1);
        return generateDateDiffFnInfo(unit, operands.get(1), operands.get(2));
      case TRIM:
        assert operands.size() == 3;
        // Calcite expects: TRIM(<chars> FROM <expr>>) or TRIM(<chars>, <expr>)
        // However, Snowflake/BodoSQL expects: TRIM(<expr>, <chars>)
        // So we just need to swap the arguments here.
        return generateTrimFnInfo(
            fnOperation.getOperator().toString(), operands.get(2), operands.get(1));
      case NULLIF:
        assert operands.size() == 2;
        expr =
            "bodo.libs.bodosql_array_kernels.nullif("
                + operands.get(0).emit()
                + ", "
                + operands.get(1).emit()
                + ")";
        return new Expr.Raw(expr);

      case POSITION:
        return generatePosition(operands);
      case OTHER:
      case OTHER_FUNCTION:
        /* If sqlKind = other function, the only recourse is to match on the name of the function. */
        switch (fnName) {
            // TODO (allai5): update this in a future PR for clean-up so it re-uses the
            // SQLLibraryOperator definition.
          case "LTRIM":
          case "RTRIM":
            if (operands.size() == 1) { // no optional characters to be trimmed
              return generateTrimFnInfo(
                  fnName, operands.get(0), new Expr.Raw("' '")); // remove spaces by default
            } else if (operands.size() == 2) {
              return generateTrimFnInfo(fnName, operands.get(0), operands.get(1));
            } else {
              throw new BodoSQLCodegenException(
                  "Invalid number of arguments to TRIM: must be either 1 or 2.");
            }
          case "WIDTH_BUCKET":
            {
              int numOps = operands.size();
              assert numOps == 4 : "WIDTH_BUCKET takes 4 arguments, but found " + numOps;
              StringBuilder exprCode =
                  new StringBuilder("bodo.libs.bodosql_array_kernels.width_bucket(");
              for (int i = 0; i < numOps; i++) {
                exprCode.append(operands.get(i).emit());
                if (i != (numOps - 1)) {
                  exprCode.append(", ");
                }
              }
              exprCode.append(")");
              return new Expr.Raw(exprCode.toString());
            }
          case "HAVERSINE":
            {
              assert operands.size() == 4;
              StringBuilder exprCode =
                  new StringBuilder("bodo.libs.bodosql_array_kernels.haversine(");
              int numOps = fnOperation.operands.size();
              for (int i = 0; i < numOps; i++) {
                exprCode.append(operands.get(i).emit());
                if (i != (numOps - 1)) {
                  exprCode.append(", ");
                }
              }
              exprCode.append(")");
              return new Expr.Raw(exprCode.toString());
            }
          case "DIV0":
            {
              assert operands.size() == 2 && fnOperation.operands.size() == 2;
              StringBuilder exprCode = new StringBuilder("bodo.libs.bodosql_array_kernels.div0(");
              exprCode.append(operands.get(0).emit());
              exprCode.append(", ");
              exprCode.append(operands.get(1).emit());
              exprCode.append(")");
              return new Expr.Raw(exprCode.toString());
            }
          case "NULLIFZERO":
            assert operands.size() == 1;
            String exprCode =
                "bodo.libs.bodosql_array_kernels.nullif(" + operands.get(0).emit() + ", 0)";
            return new Expr.Raw(exprCode);
          case "DATEADD":
          case "TIMEADD":
            // If DATEADD receives 3 arguments, use the Snowflake DATEADD.
            // Otherwise, fall back to the normal DATEADD. TIMEADD and TIMESTAMPADD are aliases.
            if (operands.size() == 3) {
              dateTimeExprType1 = getDateTimeDataType(fnOperation.getOperands().get(2));
              unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType1);
              assert exprTypes.get(0) == BodoSQLExprType.ExprType.SCALAR;
              return new Expr.Raw(generateSnowflakeDateAddCode(operands, unit));
            }
          case "DATE_ADD":
          case "ADDDATE":
          case "SUBDATE":
          case "DATE_SUB":
            {
              assert operands.size() == 2;
              // If the second argument is a timedelta, switch to manual addition
              boolean manual_addition =
                  SqlTypeName.INTERVAL_TYPES.contains(
                      fnOperation.getOperands().get(1).getType().getSqlTypeName());
              // Cannot use dateadd/datesub functions on TIME data unless the
              // amount being added to them is a timedelta
              if (!manual_addition
                  && getDateTimeDataType(fnOperation.getOperands().get(0))
                      .equals(DateTimeType.TIME)) {
                throw new BodoSQLCodegenException("Cannot add/subtract days from TIME");
              }
              Set<SqlTypeName> DATE_INTERVAL_TYPES =
                  Sets.immutableEnumSet(
                      SqlTypeName.INTERVAL_YEAR_MONTH,
                      SqlTypeName.INTERVAL_YEAR,
                      SqlTypeName.INTERVAL_MONTH,
                      SqlTypeName.INTERVAL_WEEK,
                      SqlTypeName.INTERVAL_DAY);
              boolean is_date_interval =
                  DATE_INTERVAL_TYPES.contains(
                      fnOperation.getOperands().get(1).getType().getSqlTypeName());
              Expr arg0 = operands.get(0);
              Expr arg1 = operands.get(1);
              // Cast arg0 to from string to timestamp, if needed
              if (SqlTypeName.STRING_TYPES.contains(
                  fnOperation.getOperands().get(0).getType().getSqlTypeName())) {
                RelDataType inputType = fnOperation.getOperands().get(0).getType();
                // The output type will always be the timestamp the string is being cast to.
                RelDataType outputType = fnOperation.getType();
                String casted_expr =
                    generateCastCode(
                        operands.get(0).emit(),
                        inputType,
                        outputType,
                        exprTypes.get(0) == BodoSQLExprType.ExprType.SCALAR || isSingleRow);
                arg0 = new Expr.Raw(casted_expr);
              }
              // add/minus a date interval to a date object should return a date object
              if (is_date_interval
                  && getDateTimeDataType(fnOperation.getOperands().get(0)) == DateTimeType.DATE) {
                if (fnName.equals("SUBDATE") || fnName.equals("DATE_SUB")) {
                  arg1 = new Expr.Call("bodo.libs.bodosql_array_kernels.negate", arg1);
                }
                return new Expr.Call(
                    "bodo.libs.bodosql_array_kernels.add_date_interval_to_date", arg0, arg1);
              }
              return generateMySQLDateAddCode(arg0, arg1, manual_addition, fnName);
            }

          case "DATEDIFF":
            Expr arg1;
            Expr arg2;
            unit = "DAY";

            if (operands.size() == 2) {
              arg1 = operands.get(1);
              arg2 = operands.get(0);
              dateTimeExprType1 = getDateTimeDataType(fnOperation.getOperands().get(0));
              dateTimeExprType2 = getDateTimeDataType(fnOperation.getOperands().get(1));
            } else if (operands.size() == 3) { // this is the Snowflake option
              unit = operands.get(0).emit();
              arg1 = operands.get(1);
              arg2 = operands.get(2);
              dateTimeExprType1 = getDateTimeDataType(fnOperation.getOperands().get(1));
              dateTimeExprType2 = getDateTimeDataType(fnOperation.getOperands().get(2));
            } else {
              throw new BodoSQLCodegenException(
                  "Invalid number of arguments to DATEDIFF: must be 2 or 3.");
            }
            if ((dateTimeExprType1 == DateTimeType.TIME)
                != (dateTimeExprType2 == DateTimeType.TIME)) {
              throw new BodoSQLCodegenException(
                  "Invalid type of arguments to DATEDIFF: cannot mix date/timestamp with time.");
            }
            unit = standardizeTimeUnit(fnName, unit, dateTimeExprType1);
            return generateDateDiffFnInfo(unit, arg1, arg2);
          case "TIMEDIFF":
            assert operands.size() == 3;
            dateTimeExprType1 = getDateTimeDataType(fnOperation.getOperands().get(1));
            dateTimeExprType2 = getDateTimeDataType(fnOperation.getOperands().get(2));
            if ((dateTimeExprType1 == DateTimeType.TIME)
                != (dateTimeExprType2 == DateTimeType.TIME)) {
              throw new BodoSQLCodegenException(
                  "Invalid type of arguments to TIMEDIFF: cannot mix date/timestamp with time.");
            }
            unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType1);
            arg1 = operands.get(1);
            arg2 = operands.get(2);
            return generateDateDiffFnInfo(unit, arg1, arg2);
          case "STR_TO_DATE":
            assert operands.size() == 2;
            // Format string should be a string literal.
            // This is required by the function definition.
            if (!(fnOperation.operands.get(1) instanceof RexLiteral)) {
              throw new BodoSQLCodegenException(
                  "Error STR_TO_DATE(): 'Format' must be a string literal");
            }
            strExpr =
                generateStrToDateCode(
                    operands.get(0).emit(), exprTypes.get(0), operands.get(1).emit());
            return new Expr.Raw(strExpr);
          case "TIMESTAMP":
            return generateTimestampFnCode(operands.get(0).emit());
          case "DATE":
          case "TO_DATE":
          case "TRY_TO_DATE":
            return generateToDateFnCode(operands, fnName);
          case "TO_TIMESTAMP":
          case "TO_TIMESTAMP_NTZ":
          case "TO_TIMESTAMP_LTZ":
          case "TO_TIMESTAMP_TZ":
          case "TRY_TO_TIMESTAMP":
          case "TRY_TO_TIMESTAMP_NTZ":
          case "TRY_TO_TIMESTAMP_LTZ":
          case "TRY_TO_TIMESTAMP_TZ":
            tzStr = "None";
            if (fnOperation.getType() instanceof TZAwareSqlType) {
              tzStr = ((TZAwareSqlType) fnOperation.getType()).getTZInfo().getPyZone();
            }
            return generateToTimestampFnCode(operands, fnOperation.getOperands(), tzStr, fnName);
          case "TRY_TO_BOOLEAN":
          case "TO_BOOLEAN":
            return generateToBooleanFnCode(operands, fnName);
          case "TRY_TO_BINARY":
          case "TO_BINARY":
            return generateToBinaryFnCode(operands, fnName);
          case "TO_CHAR":
          case "TO_VARCHAR":
            assert fnOperation.getOperands().size() == 1
                : "Error: TO_CHAR supplied improper number of arguments, Bodo only supports one"
                    + " argument.";
            return generateToCharFnCode(operands, fnName);
          case "TO_DOUBLE":
          case "TRY_TO_DOUBLE":
            return generateToDoubleFnCode(operands, fnName);
          case "ASINH":
          case "ACOSH":
          case "ATANH":
          case "SINH":
          case "COSH":
          case "TANH":
          case "COS":
          case "SIN":
          case "TAN":
          case "COT":
          case "ACOS":
          case "ASIN":
          case "ATAN":
          case "DEGREES":
          case "RADIANS":
            return getSingleArgTrigFnInfo(fnName, operands.get(0).emit());
          case "ATAN2":
            return getDoubleArgTrigFnInfo(fnName, operands.get(0).emit(), operands.get(1).emit());
          case "ABS":
          case "CBRT":
          case "EXP":
          case "FACTORIAL":
          case "LOG2":
          case "LOG10":
          case "LN":
          case "SIGN":
          case "SQUARE":
          case "SQRT":
          case "BITNOT":
            return new Expr.Raw(getSingleArgNumericFnInfo(fnName, operands.get(0).emit()));
          case "POWER":
          case "POW":
          case "BITAND":
          case "BITOR":
          case "BITXOR":
          case "BITSHIFTLEFT":
          case "BITSHIFTRIGHT":
          case "GETBIT":
            return new Expr.Raw(
                getDoubleArgNumericFnInfo(fnName, operands.get(0).emit(), operands.get(1).emit()));
          case "TRUNC":
          case "TRUNCATE":
          case "ROUND":
            String arg1_expr_code;
            if (operands.size() == 1) {
              // If no value is specified by, default to 0
              arg1_expr_code = "0";
            } else {
              assert operands.size() == 2;
              arg1_expr_code = operands.get(1).emit();
            }
            return new Expr.Raw(
                getDoubleArgNumericFnInfo(fnName, operands.get(0).emit(), arg1_expr_code));

          case "LOG":
            return generateLogFnInfo(operands);
          case "CONV":
            assert operands.size() == 3;
            strExpr =
                generateConvCode(
                    operands.get(0).emit(),
                    operands.get(1).emit(),
                    operands.get(2).emit(),
                    isSingleRow
                        || (visitor.exprTypesMap.get(
                                ExprTypeVisitor.generateRexNodeKey(fnOperation, nodeId))
                            == BodoSQLExprType.ExprType.SCALAR));
            return new Expr.Raw(strExpr);
          case "RAND":
            return new Expr.Raw("np.random.rand()");
          case "PI":
            return new Expr.Raw("np.pi");
          case "CONCAT":
            return generateConcatFnInfo(operands);
          case "CONCAT_WS":
            assert operands.size() >= 2;
            return generateConcatWSFnInfo(operands.get(0), operands.subList(1, operands.size()));
          case "GETDATE":
          case "CURRENT_TIMESTAMP":
          case "NOW":
          case "LOCALTIMESTAMP":
          case "SYSTIMESTAMP":
            assert operands.size() == 0;
            assert fnOperation.getType() instanceof TZAwareSqlType;
            BodoTZInfo tzTimestampInfo = ((TZAwareSqlType) fnOperation.getType()).getTZInfo();
            return generateCurrTimestampCode(fnName, tzTimestampInfo);
          case "CURRENT_TIME":
          case "LOCALTIME":
            assert operands.size() == 0;
            BodoTZInfo tzTimeInfo = this.typeSystem.getDefaultTZInfo();
            return generateCurrTimeCode(fnName, tzTimeInfo);
          case "SYSDATE":
          case "UTC_TIMESTAMP":
            assert operands.size() == 0;
            return generateUTCTimestampCode(fnName);
          case "UTC_DATE":
            assert operands.size() == 0;
            return generateUTCDateCode();
          case "MAKEDATE":
            assert operands.size() == 2;
            return generateMakeDateInfo(operands.get(0), operands.get(1));
          case "DATE_FORMAT":
            if (!(operands.size() == 2 && exprTypes.get(1) == BodoSQLExprType.ExprType.SCALAR)) {
              throw new BodoSQLCodegenException(
                  "Error, invalid argument types passed to DATE_FORMAT");
            }
            if (!(fnOperation.operands.get(1) instanceof RexLiteral)) {
              throw new BodoSQLCodegenException(
                  "Error DATE_FORMAT(): 'Format' must be a string literal");
            }
            return generateDateFormatCode(operands.get(0), operands.get(1));
          case "CURRENT_DATE":
          case "CURDATE":
            assert operands.size() == 0;
            return generateCurdateCode();
          case "YEARWEEK":
            assert operands.size() == 1;
            return getYearWeekFnInfo(operands.get(0));
          case "MONTHNAME":
          case "MONTH_NAME":
          case "DAYNAME":
          case "WEEKDAY":
          case "YEAROFWEEK":
          case "YEAROFWEEKISO":
            assert operands.size() == 1;
            if (getDateTimeDataType(fnOperation.getOperands().get(0)) == DateTimeType.TIME)
              throw new BodoSQLCodegenException("Time object is not supported by " + fnName);
            return getSingleArgDatetimeFnInfo(fnName, operands.get(0).emit());
          case "LAST_DAY":
            dateTimeExprType1 = getDateTimeDataType(fnOperation.getOperands().get(0));
            if (dateTimeExprType1 == DateTimeType.TIME)
              throw new BodoSQLCodegenException("Time object is not supported by " + fnName);
            if (operands.size() == 2) {
              unit = standardizeTimeUnit(fnName, operands.get(1).emit(), dateTimeExprType1);
              if (unit.equals("day") || TIME_PART_UNITS.contains(unit))
                throw new BodoSQLCodegenException(
                    operands.get(1).emit() + " is not a valid time unit for " + fnName);
              return generateLastDayCode(operands.get(0).emit(), unit);
            }
            assert operands.size() == 1;
            // the default time unit is month
            return generateLastDayCode(operands.get(0).emit(), "month");
          case "NEXT_DAY":
          case "PREVIOUS_DAY":
            assert operands.size() == 2;
            if (getDateTimeDataType(fnOperation.getOperands().get(0)) == DateTimeType.TIME)
              throw new BodoSQLCodegenException("Time object is not supported by " + fnName);
            return getDoubleArgDatetimeFnInfo(
                fnName, operands.get(0).emit(), operands.get(1).emit());
          case "DATE_PART":
            assert operands.size() == 2;
            assert exprTypes.get(0) == BodoSQLExprType.ExprType.SCALAR;
            isTime =
                fnOperation
                    .getOperands()
                    .get(1)
                    .getType()
                    .getSqlTypeName()
                    .toString()
                    .equals("TIME");
            isDate =
                fnOperation
                    .getOperands()
                    .get(1)
                    .getType()
                    .getSqlTypeName()
                    .toString()
                    .equals("DATE");
            return generateDatePart(operands, isTime, isDate);
          case "TO_DAYS":
            return generateToDaysCode(operands.get(0));
          case "TO_SECONDS":
            return generateToSecondsCode(operands.get(0));
          case "FROM_DAYS":
            return generateFromDaysCode(operands.get(0));
          case "TIME":
          case "TO_TIME":
          case "TRY_TO_TIME":
            return generateToTimeCode(
                fnOperation.getOperands().get(0).getType().getSqlTypeName(),
                operands.get(0),
                fnName);
          case "DATE_FROM_PARTS":
          case "DATEFROMPARTS":
            tzStr = "None";
            assert operands.size() == 3;
            return generateDateTimeTypeFromPartsCode(fnName, operands, tzStr);
          case "TIMEFROMPARTS":
          case "TIME_FROM_PARTS":
          case "TIMESTAMP_FROM_PARTS":
          case "TIMESTAMPFROMPARTS":
          case "TIMESTAMP_NTZ_FROM_PARTS":
          case "TIMESTAMPNTZFROMPARTS":
          case "TIMESTAMP_LTZ_FROM_PARTS":
          case "TIMESTAMPLTZFROMPARTS":
          case "TIMESTAMP_TZ_FROM_PARTS":
          case "TIMESTAMPTZFROMPARTS":
            tzStr = "None";
            if (fnOperation.getType() instanceof TZAwareSqlType) {
              tzStr = ((TZAwareSqlType) fnOperation.getType()).getTZInfo().getPyZone();
            }
            return generateDateTimeTypeFromPartsCode(fnName, operands, tzStr);
          case "TO_NUMBER":
          case "TO_NUMERIC":
          case "TO_DECIMAL":
            return generateToNumberCode(operands.get(0), fnName);
          case "TRY_TO_NUMBER":
          case "TRY_TO_NUMERIC":
          case "TRY_TO_DECIMAL":
            return generateTryToNumberCode(operands.get(0), fnName);
          case "UNIX_TIMESTAMP":
            return generateUnixTimestamp();
          case "FROM_UNIXTIME":
            return generateFromUnixTimeCode(operands.get(0));
          case "JSON_EXTRACT_PATH_TEXT":
            return generateJsonTwoArgsInfo(fnName, operands.get(0), operands.get(1));
          case "RLIKE":
          case "REGEXP_LIKE":
            if (!(2 <= operands.size() && operands.size() <= 3)) {
              throw new BodoSQLCodegenException(
                  "Error, invalid number of arguments passed to REGEXP_LIKE");
            }
            if (exprTypes.get(1) != BodoSQLExprType.ExprType.SCALAR
                || (operands.size() == 3 && exprTypes.get(2) != BodoSQLExprType.ExprType.SCALAR)) {
              throw new BodoSQLCodegenException(
                  "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
            }
            return generateRegexpLikeInfo(operands);
          case "REGEXP_COUNT":
            if (!(2 <= operands.size() && operands.size() <= 4)) {
              throw new BodoSQLCodegenException(
                  "Error, invalid number of arguments passed to REGEXP_COUNT");
            }
            if (exprTypes.get(1) != BodoSQLExprType.ExprType.SCALAR
                || (operands.size() == 4 && exprTypes.get(3) != BodoSQLExprType.ExprType.SCALAR)) {
              throw new BodoSQLCodegenException(
                  "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
            }
            return generateRegexpCountInfo(operands);
          case "REGEXP_REPLACE":
            if (!(2 <= operands.size() && operands.size() <= 6)) {
              throw new BodoSQLCodegenException(
                  "Error, invalid number of arguments passed to REGEXP_REPLACE");
            }
            if (exprTypes.get(1) != BodoSQLExprType.ExprType.SCALAR
                || (operands.size() == 6 && exprTypes.get(5) != BodoSQLExprType.ExprType.SCALAR)) {
              throw new BodoSQLCodegenException(
                  "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
            }
            return generateRegexpReplaceInfo(operands);
          case "REGEXP_SUBSTR":
            if (!(2 <= operands.size() && operands.size() <= 6)) {
              throw new BodoSQLCodegenException(
                  "Error, invalid number of arguments passed to REGEXP_SUBSTR");
            }
            if (exprTypes.get(1) != BodoSQLExprType.ExprType.SCALAR
                || (operands.size() > 4 && exprTypes.get(4) != BodoSQLExprType.ExprType.SCALAR)) {
              throw new BodoSQLCodegenException(
                  "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
            }
            return generateRegexpSubstrInfo(operands);
          case "REGEXP_INSTR":
            if (!(2 <= operands.size() && operands.size() <= 7)) {
              throw new BodoSQLCodegenException(
                  "Error, invalid number of arguments passed to REGEXP_INSTR");
            }
            if (exprTypes.get(1) != BodoSQLExprType.ExprType.SCALAR
                || (operands.size() > 5 && exprTypes.get(5) != BodoSQLExprType.ExprType.SCALAR)) {
              throw new BodoSQLCodegenException(
                  "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
            }
            return generateRegexpInstrInfo(operands);
          case "ORD":
          case "ASCII":
          case "CHAR":
          case "CHR":
          case "CHAR_LENGTH":
          case "CHARACTER_LENGTH":
          case "LEN":
          case "LENGTH":
          case "REVERSE":
          case "LCASE":
          case "LOWER":
          case "UCASE":
          case "UPPER":
          case "SPACE":
          case "RTRIMMED_LENGTH":
            if (operands.size() != 1) {
              throw new BodoSQLCodegenException(fnName + " requires providing only 1 argument");
            }
            return getSingleArgStringFnInfo(fnName, operands.get(0).emit());
          case "FORMAT":
          case "REPEAT":
          case "STRCMP":
          case "RIGHT":
          case "LEFT":
          case "CONTAINS":
          case "INSTR":
          case "STARTSWITH":
          case "ENDSWITH":
            if (operands.size() != 2) {
              throw new BodoSQLCodegenException(fnName + " requires providing only 2 arguments");
            }
            return getTwoArgStringFnInfo(fnName, operands.get(0), operands.get(1));
          case "RPAD":
          case "LPAD":
          case "SPLIT_PART":
          case "REPLACE":
          case "MID":
          case "SUBSTRING_INDEX":
          case "TRANSLATE3":
            if (operands.size() != 3) {
              throw new BodoSQLCodegenException(fnName + " requires providing only 3 argument");
            }
            return new Expr.Raw(
                getThreeArgStringFnInfo(
                    fnName,
                    operands.get(0).emit(),
                    operands.get(1).emit(),
                    operands.get(2).emit()));
          case "SUBSTR":
            return generateSubstringInfo(operands);
          case "INSERT":
            return generateInsert(operands);
          case "POSITION":
          case "CHARINDEX":
            return generatePosition(operands);
          case "STRTOK":
            return generateStrtok(operands);
          case "EDITDISTANCE":
            return generateEditdistance(operands);
          case "INITCAP":
            return generateInitcapInfo(operands);
          case "DATE_TRUNC":
            dateTimeExprType1 = getDateTimeDataType(fnOperation.getOperands().get(1));
            unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType1);
            return generateDateTruncCode(unit, operands.get(1));
          case "MICROSECOND":
          case "SECOND":
          case "MINUTE":
          case "DAY":
          case "DAYOFYEAR":
          case "DAYOFWEEK":
          case "DAYOFWEEKISO":
          case "DAYOFMONTH":
          case "HOUR":
          case "MONTH":
          case "QUARTER":
          case "YEAR":
          case "WEEK":
          case "WEEKOFYEAR":
          case "WEEKISO":
            isTime =
                fnOperation
                    .getOperands()
                    .get(0)
                    .getType()
                    .getSqlTypeName()
                    .toString()
                    .equals("TIME");
            isDate =
                fnOperation
                    .getOperands()
                    .get(0)
                    .getType()
                    .getSqlTypeName()
                    .toString()
                    .equals("DATE");
            return new Expr.Raw(
                generateExtractCode(fnName, operands.get(0).emit(), isTime, isDate));
          case "REGR_VALX":
          case "REGR_VALY":
            return getDoubleArgCondFnInfo(fnName, operands.get(0).emit(), operands.get(1).emit());
        }
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Function: " + fnOperation.getOperator().toString() + " not supported");
    }
  }

  @Override
  public Expr visitOver(RexOver over) {
    return visitOver(over, false);
  }

  protected Expr visitOver(RexOver over, boolean isSingleRow) {
    // Windowed aggregation is special, since it needs to add generated
    // code in order to define functions to be used with groupby apply.
    List<RexOver> tmp = Collections.singletonList(over);
    List<String> colNames = input.getRel().getRowType().getFieldNames();
    List<Expr> results = visitor.visitAggOverOp(tmp, colNames, nodeId, input, isSingleRow, ctx);
    return results.get(0);
  }

  @Override
  public Expr visitCorrelVariable(RexCorrelVariable correlVariable) {
    throw unsupportedNode();
  }

  @Override
  public Expr visitDynamicParam(RexDynamicParam dynamicParam) {
    throw unsupportedNode();
  }

  @Override
  public Expr visitNamedParam(RexNamedParam namedParam) {
    String paramName = namedParam.getName();
    // We just return the node name because that should match the input variable name
    ctx.getNamedParams().add(paramName);
    return new Variable(paramName);
  }

  @Override
  public Expr visitRangeRef(RexRangeRef rangeRef) {
    throw unsupportedNode();
  }

  @Override
  public Expr visitFieldAccess(RexFieldAccess fieldAccess) {
    throw unsupportedNode();
  }

  @Override
  public Expr visitSubQuery(RexSubQuery subQuery) {
    throw unsupportedNode();
  }

  @Override
  public Expr visitTableInputRef(RexTableInputRef fieldRef) {
    return visitInputRef(fieldRef);
  }

  @Override
  public Expr visitPatternFieldRef(RexPatternFieldRef fieldRef) {
    return visitInputRef(fieldRef);
  }

  /**
   * A version of the RexToPandasTranslator that is used when the expression is occurring in a
   * scalar context.
   */
  private static class ScalarContext extends RexToPandasTranslator {
    public ScalarContext(
        @NotNull PandasCodeGenVisitor visitor,
        @NotNull Module.Builder builder,
        @NotNull RelDataTypeSystem typeSystem,
        int nodeId,
        @NotNull Dataframe input) {
      super(visitor, builder, typeSystem, nodeId, input);
    }

    @Override
    public Expr visitInputRef(RexInputRef inputRef) {
      // Add a use for generating variables in case
      ctx.getUsedColumns().add(inputRef.getIndex());
      // NOTE: Codegen for bodosql_case_placeholder() expects column value accesses
      // (e.g. bodo.utils.indexing.scalar_optional_getitem(T1_1, i))
      Variable inputVar = new Variable(input.getVariable().getName() + "_" + inputRef.getIndex());
      Variable index = new Variable("i");
      return new Expr.Call("bodo.utils.indexing.scalar_optional_getitem", List.of(inputVar, index));
    }

    @Override
    public Expr visitLiteral(RexLiteral literal) {
      return LiteralCodeGen.generateLiteralCode(literal, true, visitor);
    }

    @Override
    protected Expr visitCastScan(RexCall operation) {
      RelDataType inputType = operation.operands.get(0).getType();
      RelDataType outputType = operation.getType();

      List<Expr> args = visitList(operation.operands);
      Expr child = args.get(0);
      String exprCode = generateCastCode(child.emit(), inputType, outputType, true);
      return new Expr.Raw(exprCode);
    }

    @Override
    protected Expr visitCaseOp(RexCall node) {
      return visitCaseOp(node, true);
    }

    @Override
    protected Expr visitInternalOp(RexCall node) {
      return visitInternalOp(node, true);
    }

    @Override
    protected Expr visitGenericFuncOp(RexCall fnOperation) {
      return visitGenericFuncOp(fnOperation, true);
    }

    @Override
    public Expr visitOver(RexOver over) {
      return visitOver(over, true);
    }
  }

  protected BodoSQLCodegenException unsupportedNode() {
    return new BodoSQLCodegenException(
        "Internal Error: Calcite Plan Produced an Unsupported RexNode");
  }
}
