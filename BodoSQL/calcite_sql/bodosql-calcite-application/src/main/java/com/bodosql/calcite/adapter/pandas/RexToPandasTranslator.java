package com.bodosql.calcite.adapter.pandas;

import static com.bodosql.calcite.application.BodoSQLCodeGen.BinOpCodeGen.generateBinOpCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CastCodeGen.generateCastCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CastCodeGen.generateTryCastCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CondOpCodeGen.getCondFuncCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CondOpCodeGen.getCondFuncCodeOptimized;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CondOpCodeGen.visitHash;
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
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.TIME_PART_UNITS;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateConvertTimezoneCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateCurrTimeCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateCurrTimestampCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateCurrentDateCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateDateFormatCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateDateTimeTypeFromPartsCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateDateTruncCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateLastDayCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateMakeDateInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateTimeSliceFnCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateToTimeCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateUTCDateCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateUTCTimestampCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.getDateTimeDataType;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.getDoubleArgDatetimeFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.getSingleArgDatetimeFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.getYearWeekFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.standardizeTimeUnit;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ExtractCodeGen.generateExtractCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.JsonCodeGen.getObjectConstructKeepNullCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.JsonCodeGen.visitJsonFunc;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NestedDataCodeGen.generateToArrayFnCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.genFloorCeilCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.generateLeastGreatestCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.generateLogFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.generateRandomFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.generateToNumberCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.generateTryToNumberCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.generateUniformFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.getNumericFnCode;
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
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateBase64DecodeFn;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateBase64Encode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateConcatCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateConcatWSCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateEditdistance;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateHexDecodeFn;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateHexEncode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateInitcapInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generatePadCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generatePosition;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateReplace;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateSHA2;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateStrtok;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateStrtokToArray;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateSubstringCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateTrimFnCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.getOptimizedStringFnCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.getStringFnCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.TrigCodeGen.getTrigFnCode;
import static com.bodosql.calcite.application.utils.BodoArrayHelpers.sqlTypeToBodoArrayType;
import static com.bodosql.calcite.application.utils.IsScalar.isScalar;
import static com.bodosql.calcite.application.utils.Utils.expectScalarArgument;

import com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.DateTimeType;
import com.bodosql.calcite.application.BodoSQLCodeGen.LiteralCodeGen;
import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.application.utils.BodoCtx;
import com.bodosql.calcite.application.utils.IsScalar;
import com.bodosql.calcite.ir.BodoEngineTable;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.FrameTripleQuotedString;
import com.bodosql.calcite.ir.Expr.None;
import com.bodosql.calcite.ir.ExprKt;
import com.bodosql.calcite.ir.Frame;
import com.bodosql.calcite.ir.Module;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.Op.Assign;
import com.bodosql.calcite.ir.Op.Continue;
import com.bodosql.calcite.ir.Op.Function;
import com.bodosql.calcite.ir.Op.If;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.rex.RexNamedParam;
import com.bodosql.calcite.sql.func.BodoSqlTryCastFunction;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import javax.annotation.Nullable;
import kotlin.Pair;
import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexCorrelVariable;
import org.apache.calcite.rex.RexDynamicParam;
import org.apache.calcite.rex.RexFieldAccess;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexLocalRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexOver;
import org.apache.calcite.rex.RexPatternFieldRef;
import org.apache.calcite.rex.RexRangeRef;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexSlot;
import org.apache.calcite.rex.RexSubQuery;
import org.apache.calcite.rex.RexTableInputRef;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.rex.RexVisitor;
import org.apache.calcite.sql.SqlBinaryOperator;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlInternalOperator;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNullTreatmentOperator;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlPostfixOperator;
import org.apache.calcite.sql.SqlPrefixOperator;
import org.apache.calcite.sql.SqlSpecialOperator;
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
  @NotNull protected final BodoEngineTable input;
  protected final List<? extends Expr> localRefs;

  @NotNull public final BodoCtx ctx;
  protected final @Nullable Integer weekStart;

  protected final @Nullable Integer weekOfYearPolicy;
  protected final @Nullable String currentDatabase;

  public RexToPandasTranslator(
      @NotNull PandasCodeGenVisitor visitor,
      @NotNull Module.Builder builder,
      @NotNull RelDataTypeSystem typeSystem,
      int nodeId,
      @NotNull BodoEngineTable input,
      @NotNull List<? extends Expr> localRefs) {
    this.visitor = visitor;
    this.builder = builder;
    this.typeSystem = typeSystem;
    this.nodeId = nodeId;
    this.input = input;
    this.localRefs = localRefs;
    this.ctx = new BodoCtx();
    if (this.typeSystem instanceof BodoSQLRelDataTypeSystem) {
      this.weekStart = ((BodoSQLRelDataTypeSystem) this.typeSystem).getWeekStart();
      this.weekOfYearPolicy = ((BodoSQLRelDataTypeSystem) this.typeSystem).getWeekOfYearPolicy();
      this.currentDatabase = ((BodoSQLRelDataTypeSystem) this.typeSystem).getCatalogName();
    } else {
      this.weekStart = 0;
      this.weekOfYearPolicy = 0;
      this.currentDatabase = null;
    }
  }

  public RexToPandasTranslator(
      @NotNull PandasCodeGenVisitor visitor,
      @NotNull Module.Builder builder,
      @NotNull RelDataTypeSystem typeSystem,
      int nodeId,
      @NotNull BodoEngineTable input) {
    this(visitor, builder, typeSystem, nodeId, input, List.of());
  }

  public @NotNull BodoEngineTable getInput() {
    return input;
  }

  @Override
  public Expr visitInputRef(RexInputRef inputRef) {
    return new Expr.Call(
        "bodo.hiframes.table.get_table_data",
        List.of(input, new Expr.IntegerLiteral(inputRef.getIndex())));
  }

  @Override
  public Expr visitLocalRef(RexLocalRef localRef) {
    return localRefs.get(localRef.getIndex());
  }

  @Override
  public Expr visitLiteral(RexLiteral literal) {
    return LiteralCodeGen.generateLiteralCode(literal, false, visitor);
  }

  @Override
  public Expr visitCall(RexCall call) {
    // TODO(jsternberg): Using instanceof here is problematic.
    // It would be better to use getKind(). Revisit this later.
    if (call instanceof RexNamedParam) {
      return visitNamedParam((RexNamedParam) call);
    } else if (call.getOperator() instanceof SqlNullTreatmentOperator) {
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
    } else if (call.getOperator() instanceof SqlSpecialOperator) {
      return visitSpecialOp(call);
    } else {
      throw new BodoSQLCodegenException(
          "Internal Error: Calcite Plan Produced an Unsupported RexCall:" + call.getOperator());
    }
  }

  private Expr visitSpecialOp(RexCall node) {
    List<Expr> operands = visitList(node.operands);
    Expr output = null;
    switch (node.getKind()) {
      case ITEM:
        assert operands.size() == 2;
        boolean inputScalar = isOperandScalar(node.getOperands().get(0));
        kotlin.Pair isScalarArg =
            new kotlin.Pair("is_scalar_arr", new Expr.BooleanLiteral(inputScalar));
        List<Pair<String, Expr>> namedArgs = List.of(isScalarArg);
        return new Expr.Call("bodo.libs.bodosql_array_kernels.arr_get", operands, namedArgs);
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Unsupported special operand call: "
                + node.getOperator());
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
            "Error during codegen: " + innerCallKind + " requires OVER clause.");
      default:
        throw new BodoSQLCodegenException(
            "Error during codegen: Unreachable code entered while evaluating the following rex"
                + " node in visitNullTreatmentOp: "
                + node);
    }
  }

  protected Expr visitBinOpScan(RexCall operation) {
    return this.visitBinOpScan(operation, List.of());
  }

  /**
   * @param operand
   * @return True if the operand is a scalar
   */
  protected Boolean isOperandScalar(RexNode operand) {
    return IsScalar.isScalar(operand);
  }

  /**
   * Generate the code for a Binary operation.
   *
   * @param operation The operation from which to generate the expression.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitBinOpScan(RexCall operation, List<Pair<String, Expr>> streamingNamedArgs) {
    List<Expr> args = new ArrayList<>();
    SqlOperator binOp = operation.getOperator();
    // Store the argument types for TZ-Aware data
    List<RelDataType> argDataTypes = new ArrayList<>();
    // Store whether the arguments were scalars vs columns
    List<Boolean> argScalars = new ArrayList<>();
    for (RexNode operand : operation.operands) {
      Expr exprCode = operand.accept(this);
      args.add(exprCode);
      argDataTypes.add(operand.getType());
      argScalars.add(isOperandScalar(operand));
    }
    if (binOp.getKind() == SqlKind.OTHER && binOp.getName().equals("||")) {
      // Support the concat operator by using the concat array kernel.
      return generateConcatCode(args, streamingNamedArgs, operation.getType());
    }
    return generateBinOpCode(
        args, binOp, argDataTypes, this.builder, streamingNamedArgs, argScalars);
  }

  private Expr visitPostfixOpScan(RexCall operation) {
    List<Expr> args = visitList(operation.operands);
    Expr seriesOp = args.get(0);
    return generatePostfixOpCode(seriesOp, operation.getOperator());
  }

  private Expr visitPrefixOpScan(RexCall operation) {
    List<Expr> args = visitList(operation.operands);
    Expr seriesOp = args.get(0);
    return generatePrefixOpCode(seriesOp, operation.getOperator());
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

          // TODO(jsternberg): This really should have been done before code generation
          // even started. This shouldn't be here so we're
          // hacking around the improper placement by recreating the type factory.
          RexBuilder rexBuilder = new RexBuilder(new JavaTypeFactoryImpl(typeSystem));
          RexNode searchNode = RexUtil.expandSearch(rexBuilder, null, node);
          return searchNode.accept(this);
        }
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Internal Operator");
    }
  }

  protected Expr visitLikeOp(RexCall node) {
    return visitLikeOp(node, List.of());
  }

  /**
   * Generate the code for a like operation.
   *
   * @param node The node from which to generate the expression.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitLikeOp(RexCall node, List<Pair<String, Expr>> streamingNamedArgs) {
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
    if (op.getKind() == SqlKind.RLIKE) {
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
          "bodo.libs.bodosql_array_kernels.regexp_like",
          List.of(arg, pattern, new Expr.StringLiteral("")),
          streamingNamedArgs);
    } else {
      return new Expr.Call(
          "bodo.libs.bodosql_array_kernels.like_kernel",
          List.of(
              arg,
              pattern,
              escape,
              // Use the opposite. The python call is for case insensitivity while
              // our boolean is for case sensitivity, so they are opposites.
              new Expr.BooleanLiteral(!op.isCaseSensitive())),
          streamingNamedArgs);
    }
  }

  /**
   * Overview of code generation for case:
   * https://bodo.atlassian.net/wiki/spaces/B/pages/1368752135/WIP+BodoSQL+Case+Implementation
   *
   * @param node The case node to visit
   * @return The resulting expression from generating case code.
   */
  protected Expr visitCaseOp(RexCall node) {
    // Extract the inputs as we need to know what columns to pass into
    // the case placeholder and we need to rewrite the refs to the ones
    // we will generate.
    CaseInputFinder inputFinder = new CaseInputFinder();
    ImmutableList.Builder<RexNode> operandsBuilder = ImmutableList.builder();
    for (RexNode operand : node.getOperands()) {
      RexNode newOperand = operand.accept(inputFinder);
      operandsBuilder.add(newOperand);
    }
    List<RexNode> operands = operandsBuilder.build();

    // Generate the initialization of the local variables and also
    // fill in the access names for those local variables.
    builder.startCodegenFrame();
    ImmutableList.Builder<Expr> localRefsBuilder = ImmutableList.builder();
    Variable arrs = new Variable("arrs");
    Variable indexingVar = builder.getSymbolTable().genGenericTempVar();
    List<Variable> closureVars = new ArrayList<>();
    for (int i = 0; i < inputFinder.size(); i++) {
      Variable localVar = new Variable(input.getName() + "_" + i);
      closureVars.add(localVar);
      localRefsBuilder.add(
          new Expr.Call("bodo.utils.indexing.scalar_optional_getitem", localVar, indexingVar));

      Op.Assign initLine = new Assign(localVar, new Expr.GetItem(arrs, new Expr.IntegerLiteral(i)));
      builder.add(initLine);
    }
    closureVars.add(indexingVar);
    List<Expr> localRefs = localRefsBuilder.build();

    // Create a local translator for the case operands and initialize it with the
    // local variables we initialized above.
    ScalarContext localTranslator =
        new RexToPandasTranslator.ScalarContext(
            visitor, builder, typeSystem, nodeId, input, localRefs, closureVars);

    // Start a new codegen frame as we will perform our processing there.
    Variable arrVar = builder.getSymbolTable().genArrayVar();

    Frame outputFrame =
        visitCaseOperands(localTranslator, operands, List.of(arrVar, indexingVar), false);
    Variable caseBodyGlobal = visitor.lowerAsMetaType(new FrameTripleQuotedString(outputFrame, 2));

    // Append all of the closures generated to the init frame
    List<Function> closures = localTranslator.getClosures();
    for (Function closure : closures) {
      builder.add(closure);
    }
    Variable caseBodyInit =
        visitor.lowerAsMetaType(new FrameTripleQuotedString(builder.endFrame(), 1));

    ImmutableList.Builder<Expr> caseArgsBuilder = ImmutableList.builder();
    for (RexNode ref : inputFinder.getRefs()) {
      caseArgsBuilder.add(ref.accept(this));
    }

    // Organize any named parameters if they exist.
    List<kotlin.Pair<String, Expr>> namedArgs = new ArrayList<>();
    for (String param : inputFinder.getNamedParams()) {
      namedArgs.add(new kotlin.Pair<>(param, new Expr.Raw(param)));
    }

    // Generate the call to bodosql_case_placeholder and assign the results
    // to a temporary value that we return as the output.
    Variable tempVar = builder.getSymbolTable().genGenericTempVar();
    Variable outputArrayTypeGlobal =
        visitor.lowerAsGlobal(sqlTypeToBodoArrayType(node.getType(), false));
    Expr casePlaceholder =
        new Expr.Call(
            "bodo.utils.typing.bodosql_case_placeholder",
            List.of(
                new Expr.Tuple(caseArgsBuilder.build()),
                new Expr.Call("len", input),
                caseBodyInit,
                caseBodyGlobal,
                // Note: The variable name must be a string literal here.
                new Expr.StringLiteral(arrVar.emit()),
                // Note: The variable name must be a string literal here.
                new Expr.StringLiteral(indexingVar.emit()),
                outputArrayTypeGlobal),
            namedArgs);
    builder.add(new Op.Assign(tempVar, casePlaceholder));
    return tempVar;
  }

  /** Utility to find variable uses inside a case statement. */
  private static class CaseInputFinder extends RexShuttle {
    private final ArrayList<RexSlot> refs;
    private final Set<String> namedParams;

    public CaseInputFinder() {
      refs = new ArrayList<>();
      namedParams = new HashSet<>();
    }

    public List<RexSlot> getRefs() {
      return ImmutableList.copyOf(refs);
    }

    public List<String> getNamedParams() {
      return ImmutableList.sortedCopyOf(namedParams);
    }

    public int size() {
      return refs.size();
    }

    @Override
    public RexNode visitInputRef(RexInputRef inputRef) {
      return visitGenericRef(inputRef);
    }

    @Override
    public RexNode visitLocalRef(RexLocalRef localRef) {
      return visitGenericRef(localRef);
    }

    @Override
    public RexNode visitCall(RexCall call) {
      if (call instanceof RexNamedParam) {
        namedParams.add(((RexNamedParam) call).getName());
        return call;
      }
      return super.visitCall(call);
    }

    private RexNode visitGenericRef(RexSlot ref) {
      // Identify if there's an identical RexSlot that was
      // already found.
      for (int i = 0; i < refs.size(); i++) {
        RexSlot it = refs.get(i);
        if (it.equals(ref)) {
          // Return a RexLocalRef that refers to this index position.
          return new RexLocalRef(i, ref.getType());
        }
      }

      // Record this as a new RexSlot and return a RexLocalRef
      // that refers to it.
      int next = refs.size();
      refs.add(ref);
      return new RexLocalRef(next, ref.getType());
    }
  }

  /**
   * Visit the operands to a call to case using the given translator. Each operand generates its
   * code in a unique frame to define a unique scope.
   *
   * <p>The operands to case are of the form: [cond1, truepath1, cond2, truepath2, ..., elsepath]
   * All of these components are present, so the operands is always of Length 2n + 1, where n is the
   * number of conditions and n > 0. The else is always present, even if not explicit in the
   * original SQL query.
   *
   * <p>For code generation purposes we cannot produce an arbitrary set of if/else statements
   * because this can potentially trigger a max indentation depth issue in Python. As a result we
   * generate each if statement as its own scope and either continue or return directly from that
   * block.
   *
   * <p>For example, if we have 5 inputs with no intermediate variables, the generated code might
   * look like this:
   *
   * <p><code>
   *     if bodo.libs.bodosql_array_kernels.is_true(cond1):
   *        out_arr[i] = truepath1
   *        continue
   *     if bodo.libs.bodosql_array_kernels.is_true(cond2):
   *        out_arr[i] = truepath2
   *        continue
   *     out_arr[i] = elsepath
   * </code> If instead this is called from a scalar context, then we will be generating a closure
   * so each out_arr[i] should be a return instead
   *
   * <p><code>
   *     if bodo.libs.bodosql_array_kernels.is_true(cond1):
   *        var = truepath1
   *        return var
   *     if bodo.libs.bodosql_array_kernels.is_true(cond2):
   *        var = truepath2
   *        return var
   *     var = elsepath
   *     return var
   * </code>
   *
   * @param translator The translator used to visit each operand.
   * @param operands The list of RexNodes to visit to capture the proper computation.
   * @param outputVars The variables used in generating the output.
   * @param isScalarContext Is this code generated in a scalar context?
   * @return A single frame containing all the generated code.
   */
  protected Frame visitCaseOperands(
      RexToPandasTranslator translator,
      List<RexNode> operands,
      List<Variable> outputVars,
      boolean isScalarContext) {
    // Generate the target Frame for the output
    builder.startCodegenFrame();
    for (int i = 0; i < operands.size() - 1; i += 2) {
      // Visit the cond code
      Expr cond = operands.get(i).accept(translator);
      // Visit the if code
      builder.startCodegenFrame();
      Expr ifPath = operands.get(i + 1).accept(translator);
      assignCasePathOutput(ifPath, outputVars, isScalarContext);
      // Pop the frame
      Frame ifFrame = builder.endFrame();
      // Generate the if statement
      Expr.Call condCall = new Expr.Call("bodo.libs.bodosql_array_kernels.is_true", List.of(cond));
      Op.If ifStatement = new If(condCall, ifFrame, null);
      builder.add(ifStatement);
    }
    // Process the else.
    Expr elsePath = operands.get(operands.size() - 1).accept(translator);
    assignCasePathOutput(elsePath, outputVars, isScalarContext);
    return builder.endFrame();
  }

  /**
   * Assign the output value from a singular case path.
   *
   * @param outputExpr The expression from one of the then/else paths that needs to be assigned to
   *     the final output.
   * @param outputVars The variables used in generating the output.
   * @param isScalarContext Is this code generated in a scalar context?
   */
  private void assignCasePathOutput(
      Expr outputExpr, List<Variable> outputVars, boolean isScalarContext) {
    if (isScalarContext) {
      // Scalar path. Assign and return the variable.
      Variable outputVar = outputVars.get(0);
      builder.add(new Op.Assign(outputVar, outputExpr));
      builder.add(new Op.ReturnStatement(outputVar));
    } else {
      // Unwrap the code
      Variable arrVar = outputVars.get(0);
      Variable indexVar = outputVars.get(1);
      Expr unwrappedExpr =
          new Expr.Call("bodo.utils.conversion.unbox_if_tz_naive_timestamp", List.of(outputExpr));
      builder.add(new Op.SetItem(arrVar, indexVar, unwrappedExpr));
      builder.add(Continue.INSTANCE);
    }
  }

  protected Expr visitCastScan(RexCall operation) {
    return visitCastScan(operation, isScalar(operation), List.of());
  }

  /**
   * Generate the code for a cast operation.
   *
   * @param operation The operation from which to generate the expression.
   * @param outputScalar Is the output a scalar or an array?
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitCastScan(
      RexCall operation, boolean outputScalar, List<Pair<String, Expr>> streamingNamedArgs) {
    RelDataType inputType = operation.operands.get(0).getType();
    RelDataType outputType = operation.getType();
    assert operation.operands.size() == 1;
    Expr child = operation.getOperands().get(0).accept(this);
    return generateCastCode(child, inputType, outputType, outputScalar, streamingNamedArgs);
  }

  protected Expr visitTryCastScan(RexCall operation) {
    return visitTryCastScan(operation, List.of());
  }

  /**
   * Generate the code for a tryCast operation.
   *
   * @param operation The operation from which to generate the expression.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitTryCastScan(RexCall operation, List<Pair<String, Expr>> streamingNamedArgs) {
    RelDataType inputType = operation.operands.get(0).getType();
    if (!SqlTypeName.CHAR_TYPES.contains(inputType.getSqlTypeName())) {
      throw new BodoSQLCodegenException("TRY_CAST only supports casting from strings.");
    }
    RelDataType outputType = operation.getType();
    assert operation.operands.size() == 1;
    Expr child = operation.operands.get(0).accept(this);
    return generateTryCastCode(child, outputType, streamingNamedArgs);
  }

  private Expr visitExtractScan(RexCall node) {
    List<Expr> args = visitList(node.operands);
    boolean isTime = node.operands.get(1).getType().getSqlTypeName().toString().equals("TIME");
    boolean isDate = node.operands.get(1).getType().getSqlTypeName().toString().equals("DATE");
    Expr dateVal = args.get(0);
    Expr column = args.get(1);
    return generateExtractCode(
        dateVal.emit(), column, isTime, isDate, this.weekStart, this.weekOfYearPolicy);
  }

  protected Expr visitSubstringScan(RexCall node) {
    return visitSubstringScan(node, List.of());
  }

  /**
   * Generate the code for a substring operation.
   *
   * @param node The operation from which to generate the expression.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitSubstringScan(RexCall node, List<Pair<String, Expr>> streamingNamedArgs) {
    // node.operands contains
    //  * String to perform the substring operation on
    //  * start index
    //  * substring length (optional)
    //  All of these values can be both scalars and columns
    // NOTE: check on number of arguments happen in generateSubstringCode
    List<Expr> operands = visitList(node.operands);
    return generateSubstringCode(operands, streamingNamedArgs);
  }

  protected Expr visitGenericFuncOp(RexCall fnOperation) {
    return visitGenericFuncOp(fnOperation, false);
  }

  protected Expr visitNullIgnoringGenericFunc(
      RexCall fnOperation, boolean isSingleRow, List<Boolean> argScalars) {
    return visitNullIgnoringGenericFunc(fnOperation, isSingleRow, List.of(), argScalars);
  }

  /**
   * Generate the code for generic functions that have special handling for null values.
   *
   * @param fnOperation The RexCall operation
   * @param isSingleRow Does the data operate on/output a single row?
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @param argScalars Whether each argument is a scalar or a column
   * @return The generated expression.
   */
  protected Expr visitNullIgnoringGenericFunc(
      RexCall fnOperation,
      boolean isSingleRow,
      List<Pair<String, Expr>> streamingNamedArgs,
      List<Boolean> argScalars) {
    String fnName = fnOperation.getOperator().getName();
    List<Expr> codeExprs = new ArrayList<>();
    if (fnName.equals("OBJECT_CONSTRUCT_KEEP_NULL")) {
      return getObjectConstructKeepNullCode(fnOperation, argScalars, this, visitor);
    }
    for (RexNode operand : fnOperation.operands) {
      Expr operandInfo = operand.accept(this);
      // Need to unbox scalar timestamp values.
      if (isSingleRow || isScalar(operand)) {
        operandInfo =
            new Expr.Call(
                "bodo.utils.conversion.unbox_if_tz_naive_timestamp", List.of(operandInfo));
      }
      codeExprs.add(operandInfo);
    }
    Expr result;
    switch (fnName) {
      case "IF":
      case "IFF":
      case "BOOLNOT":
      case "BOOLAND":
      case "BOOLOR":
      case "BOOLXOR":
      case "NVL2":
        result = getCondFuncCode(fnName, codeExprs);
        break;
      case "EQUAL_NULL":
        result = getCondFuncCodeOptimized(fnName, codeExprs, streamingNamedArgs, argScalars);
        break;
      case "COALESCE":
      case "ZEROIFNULL":
      case "IFNULL":
      case "NVL":
      case "DECODE":
        result = visitVariadic(fnName, codeExprs, streamingNamedArgs);
        break;
      case "ARRAY_CONSTRUCT":
        result = visitArrayConstruct(codeExprs, isSingleRow, argScalars);
        break;
      case "HASH":
        result = visitHash(fnName, codeExprs);
        break;
      default:
        throw new BodoSQLCodegenException("Internal Error: reached unreachable code");
    }
    return result;
  }

  /**
   * Represents a cast operation that isn't generated by the planner. TODO(njriasan): Remove and
   * update the planner to insert these casts.
   */
  protected Expr visitDynamicCast(
      Expr arg, RelDataType inputType, RelDataType outputType, boolean isScalar) {
    return visitDynamicCast(arg, inputType, outputType, isScalar, List.of());
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
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitDynamicCast(
      Expr arg,
      RelDataType inputType,
      RelDataType outputType,
      boolean isScalar,
      List<Pair<String, Expr>> streamingNamedArgs) {
    return generateCastCode(arg, inputType, outputType, isScalar, streamingNamedArgs);
  }

  protected Expr visitTrimFunc(String fnName, Expr stringToBeTrimmed, Expr charactersToBeTrimmed) {
    return visitTrimFunc(fnName, stringToBeTrimmed, charactersToBeTrimmed, List.of());
  }

  /**
   * Generate the code for the TRIM functions.
   *
   * @param fnName The name of the TRIM function.
   * @param stringToBeTrimmed Expr for the string to be trim.
   * @param charactersToBeTrimmed Expr for identifying the characters to trim.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitTrimFunc(
      String fnName,
      Expr stringToBeTrimmed,
      Expr charactersToBeTrimmed,
      List<Pair<String, Expr>> streamingNamedArgs) {
    return generateTrimFnCode(fnName, stringToBeTrimmed, charactersToBeTrimmed, streamingNamedArgs);
  }

  protected Expr visitNullIfFunc(List<Expr> operands) {
    return visitNullIfFunc(operands, List.of());
  }

  /**
   * Generate the code for a NULLIF function.
   *
   * @param operands The arguments to the function.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitNullIfFunc(List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    return new Expr.Call("bodo.libs.bodosql_array_kernels.nullif", operands, streamingNamedArgs);
  }

  protected Expr visitLeastGreatest(String fnName, List<Expr> operands) {
    return visitLeastGreatest(fnName, operands, List.of());
  }

  /**
   * Generate the code for the Least/Greatest.
   *
   * @param fnName The name of the function.
   * @param operands The arguments to the function.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitLeastGreatest(
      String fnName, List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    return generateLeastGreatestCode(fnName, operands, streamingNamedArgs);
  }

  protected Expr visitPosition(List<Expr> operands) {
    return visitPosition(operands, List.of());
  }

  /**
   * Generate the code for Position.
   *
   * @param operands The arguments to the function.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitPosition(List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    return generatePosition(operands, streamingNamedArgs);
  }

  protected Expr visitCastFunc(RexCall fnOperation, List<Expr> operands) {
    return visitCastFunc(fnOperation, operands, List.of());
  }

  /**
   * Generate the code for Cast function calls.
   *
   * @param fnOperation The RexNode for the function call.
   * @param operands The arguments to the function.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitCastFunc(
      RexCall fnOperation, List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    String fnName = fnOperation.getOperator().getName();
    switch (fnName) {
      case "TIMESTAMP":
        return generateTimestampFnCode(operands, streamingNamedArgs);
      case "DATE":
      case "TO_DATE":
      case "TRY_TO_DATE":
        return generateToDateFnCode(operands, fnName, streamingNamedArgs);
      case "TO_TIMESTAMP":
      case "TO_TIMESTAMP_NTZ":
      case "TO_TIMESTAMP_LTZ":
      case "TO_TIMESTAMP_TZ":
      case "TRY_TO_TIMESTAMP":
      case "TRY_TO_TIMESTAMP_NTZ":
      case "TRY_TO_TIMESTAMP_LTZ":
      case "TRY_TO_TIMESTAMP_TZ":
        Expr tzInfo;
        if (fnOperation.getType() instanceof TZAwareSqlType) {
          tzInfo = ((TZAwareSqlType) fnOperation.getType()).getTZInfo().getZoneExpr();
        } else {
          tzInfo = None.INSTANCE;
        }
        return generateToTimestampFnCode(operands, tzInfo, fnName, streamingNamedArgs);
      case "TRY_TO_BOOLEAN":
      case "TO_BOOLEAN":
        return generateToBooleanFnCode(operands, fnName, streamingNamedArgs);
      case "TRY_TO_BINARY":
      case "TO_BINARY":
        return generateToBinaryFnCode(operands, fnName, streamingNamedArgs);
      case "TO_CHAR":
      case "TO_VARCHAR":
        return generateToCharFnCode(operands);
      case "TO_NUMBER":
      case "TO_NUMERIC":
      case "TO_DECIMAL":
        return generateToNumberCode(operands, streamingNamedArgs);
      case "TRY_TO_NUMBER":
      case "TRY_TO_NUMERIC":
      case "TRY_TO_DECIMAL":
        return generateTryToNumberCode(operands, streamingNamedArgs);
      case "TO_DOUBLE":
      case "TRY_TO_DOUBLE":
        return generateToDoubleFnCode(operands, fnName, streamingNamedArgs);
      case "TIME":
      case "TO_TIME":
      case "TRY_TO_TIME":
        return generateToTimeCode(operands, fnName, streamingNamedArgs);
      case "TO_ARRAY":
        return generateToArrayFnCode(this.visitor, fnOperation, operands, streamingNamedArgs);
      case "ARRAY_TO_STRING":
        assert operands.size() == 2;
        return ExprKt.BodoSQLKernel("array_to_string", operands, List.of());
      default:
        throw new BodoSQLCodegenException(String.format("Unexpected Cast function: %s", fnName));
    }
  }

  /**
   * Generate the code for Regex function calls.
   *
   * @param fnOperation The RexNode for the function call.
   * @param operands The arguments to the function.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitRegexFunc(
      RexCall fnOperation, List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    String fnName = fnOperation.getOperator().getName();
    switch (fnName) {
      case "RLIKE":
      case "REGEXP_LIKE":
        if (!(2 <= operands.size() && operands.size() <= 3)) {
          throw new BodoSQLCodegenException(
              "Error, invalid number of arguments passed to REGEXP_LIKE");
        }
        if (!isScalar(fnOperation.operands.get(1))
            || (operands.size() == 3 && !isScalar(fnOperation.operands.get(2)))) {
          throw new BodoSQLCodegenException(
              "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
        }
        return generateRegexpLikeInfo(operands, streamingNamedArgs);
      case "REGEXP_COUNT":
        if (!(2 <= operands.size() && operands.size() <= 4)) {
          throw new BodoSQLCodegenException(
              "Error, invalid number of arguments passed to REGEXP_COUNT");
        }
        if (!isScalar(fnOperation.operands.get(1))
            || (operands.size() == 4 && !isScalar(fnOperation.operands.get(3)))) {
          throw new BodoSQLCodegenException(
              "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
        }
        return generateRegexpCountInfo(operands, streamingNamedArgs);
      case "REGEXP_REPLACE":
        if (!(2 <= operands.size() && operands.size() <= 6)) {
          throw new BodoSQLCodegenException(
              "Error, invalid number of arguments passed to REGEXP_REPLACE");
        }
        if (!isScalar(fnOperation.operands.get(1))
            || (operands.size() == 6 && !isScalar(fnOperation.operands.get(5)))) {
          throw new BodoSQLCodegenException(
              "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
        }
        return generateRegexpReplaceInfo(operands, streamingNamedArgs);
      case "REGEXP_SUBSTR":
        if (!(2 <= operands.size() && operands.size() <= 6)) {
          throw new BodoSQLCodegenException(
              "Error, invalid number of arguments passed to REGEXP_SUBSTR");
        }
        if (!isScalar(fnOperation.operands.get(1))
            || (operands.size() > 4 && !isScalar(fnOperation.operands.get(4)))) {
          throw new BodoSQLCodegenException(
              "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
        }
        return generateRegexpSubstrInfo(operands, streamingNamedArgs);
      case "REGEXP_INSTR":
        if (!(2 <= operands.size() && operands.size() <= 7)) {
          throw new BodoSQLCodegenException(
              "Error, invalid number of arguments passed to REGEXP_INSTR");
        }
        if (!isScalar(fnOperation.operands.get(1))
            || (operands.size() > 5 && !isScalar(fnOperation.operands.get(5)))) {
          throw new BodoSQLCodegenException(
              "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
        }
        return generateRegexpInstrInfo(operands, streamingNamedArgs);
      default:
        throw new BodoSQLCodegenException(String.format("Unexpected Regex function: %s", fnName));
    }
  }

  protected Expr visitStringFunc(RexCall fnOperation, List<Expr> operands) {
    return visitStringFunc(fnOperation, operands, List.of());
  }

  /**
   * Generate the code for String function calls.
   *
   * @param fnOperation The RexNode for the function call.
   * @param operands The arguments to the function.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitStringFunc(
      RexCall fnOperation, List<Expr> operands, List<Pair<String, Expr>> streamingNamedArgs) {
    String fnName = fnOperation.getOperator().getName();
    switch (fnName) {
      case "RLIKE":
      case "REGEXP_LIKE":
      case "REGEXP_COUNT":
      case "REGEXP_REPLACE":
      case "REGEXP_SUBSTR":
      case "REGEXP_INSTR":
        return visitRegexFunc(fnOperation, operands, streamingNamedArgs);
      case "CHR":
      case "CHAR":
      case "FORMAT":
        return getStringFnCode(fnName, operands);
      case "ORD":
      case "ASCII":
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
      case "JAROWINKLER_SIMILARITY":
      case "REPEAT":
      case "STRCMP":
      case "RIGHT":
      case "LEFT":
      case "CONTAINS":
      case "INSTR":
      case "INSERT":
      case "STARTSWITH":
      case "ENDSWITH":
      case "SPLIT_PART":
      case "MID":
      case "SUBSTRING_INDEX":
      case "TRANSLATE3":
      case "SPLIT":
        return getOptimizedStringFnCode(fnName, operands, streamingNamedArgs);
      case "RPAD":
      case "LPAD":
        return generatePadCode(fnOperation, operands, streamingNamedArgs);
      case "SUBSTR":
        return generateSubstringCode(operands, streamingNamedArgs);
      case "POSITION":
      case "CHARINDEX":
        return visitPosition(operands, streamingNamedArgs);
      case "STRTOK":
        return generateStrtok(operands, streamingNamedArgs);
      case "STRTOK_TO_ARRAY":
        return generateStrtokToArray(operands, streamingNamedArgs);
      case "EDITDISTANCE":
        return generateEditdistance(operands, streamingNamedArgs);
      case "INITCAP":
        return generateInitcapInfo(operands, streamingNamedArgs);
      case "REPLACE":
        return generateReplace(operands, streamingNamedArgs);
      case "SHA2":
      case "SHA2_HEX":
        return generateSHA2(operands, streamingNamedArgs);
      case "MD5":
      case "MD5_HEX":
        return ExprKt.BodoSQLKernel("md5", operands, streamingNamedArgs);
      case "HEX_ENCODE":
        return generateHexEncode(operands, streamingNamedArgs);
      case "HEX_DECODE_STRING":
      case "HEX_DECODE_BINARY":
      case "TRY_HEX_DECODE_STRING":
      case "TRY_HEX_DECODE_BINARY":
        return generateHexDecodeFn(fnName, operands, streamingNamedArgs);
      case "BASE64_ENCODE":
        return generateBase64Encode(operands, streamingNamedArgs);
      case "BASE64_DECODE_STRING":
      case "TRY_BASE64_DECODE_STRING":
      case "BASE64_DECODE_BINARY":
      case "TRY_BASE64_DECODE_BINARY":
        return generateBase64DecodeFn(fnName, operands, streamingNamedArgs);
      default:
        throw new BodoSQLCodegenException(String.format("Unexpected String function: %s", fnName));
    }
  }

  /**
   * Implementation for functions that match or resemble Snowflake General context functions.
   *
   * <p>https://docs.snowflake.com/en/sql-reference/functions-context
   *
   * <p>These function are typically non-deterministic, so they must be called outside any loop to
   * give consistent results and should be required to hold the same value on all ranks. If called
   * inside a Case statement then we won't make the results consistent.
   *
   * @param fnOperation The RexCall that is producing a system operation.
   * @param makeConsistent Should the function be made consistent. This influences the generated
   *     function call.
   * @return A variable holding the result. This function always writes its result to an
   *     intermediate variable because it needs to insert the code into the Builder without being
   *     caught in the body of a loop for streaming.
   */
  protected Variable visitGeneralContextFunction(RexCall fnOperation, boolean makeConsistent) {
    String fnName = fnOperation.getOperator().getName().toUpperCase(Locale.ROOT);
    Expr systemCall;
    switch (fnName) {
      case "GETDATE":
      case "CURRENT_TIMESTAMP":
      case "NOW":
      case "LOCALTIMESTAMP":
      case "SYSTIMESTAMP":
        assert fnOperation.getType() instanceof TZAwareSqlType;
        BodoTZInfo tzTimestampInfo = ((TZAwareSqlType) fnOperation.getType()).getTZInfo();
        systemCall = generateCurrTimestampCode(tzTimestampInfo, makeConsistent);
        break;
      case "CURRENT_TIME":
      case "LOCALTIME":
        BodoTZInfo tzTimeInfo = BodoTZInfo.getDefaultTZInfo(this.typeSystem);
        systemCall = generateCurrTimeCode(tzTimeInfo, makeConsistent);
        break;
      case "SYSDATE":
      case "UTC_TIMESTAMP":
        systemCall = generateUTCTimestampCode(makeConsistent);
        break;
      case "UTC_DATE":
        systemCall = generateUTCDateCode(makeConsistent);
        break;
      case "CURRENT_DATE":
      case "CURDATE":
        systemCall =
            generateCurrentDateCode(BodoTZInfo.getDefaultTZInfo(this.typeSystem), makeConsistent);
        break;
      case "CURRENT_DATABASE":
        if (this.currentDatabase != null) {
          systemCall = new Expr.StringLiteral(this.currentDatabase);
        } else {
          throw new BodoSQLCodegenException("No information about current database is found.");
        }
        break;
      default:
        throw new BodoSQLCodegenException(
            String.format(Locale.ROOT, "Unsupported System function: %s", fnName));
    }
    Variable var = builder.getSymbolTable().genGenericTempVar();
    Assign assign = new Assign(var, systemCall);
    builder.addPureScalarAssign(assign);
    return var;
  }

  protected Variable visitGeneralContextFunction(RexCall fnOperation) {
    return visitGeneralContextFunction(fnOperation, true);
  }

  /**
   * Implementation for functions that use nested arrays.
   *
   * @param fnName The name of the function.
   * @param operands The arguments to the function.
   * @param argScalars Indicates which arguments are scalars
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The generated expression.
   */
  protected Expr visitNestedArrayFunc(
      String fnName,
      List<Expr> operands,
      List<Boolean> argScalars,
      List<Pair<String, Expr>> streamingNamedArgs) {
    switch (fnName) {
      case "ARRAYS_OVERLAP":
      case "ARRAY_CONTAINS":
      case "ARRAY_POSITION":
      case "ARRAY_EXCEPT":
      case "ARRAY_INTERSECTION":
      case "ARRAY_CAT":
        Expr isScalar0 = new Expr.BooleanLiteral(argScalars.get(0));
        Expr isScalar1 = new Expr.BooleanLiteral(argScalars.get(0));
        ArrayList<Pair<String, Expr>> kwargs = new ArrayList();
        kwargs.add(new Pair<String, Expr>("is_scalar_0", isScalar0));
        kwargs.add(new Pair<String, Expr>("is_scalar_1", isScalar1));
        return ExprKt.BodoSQLKernel(fnName.toLowerCase(Locale.ROOT), operands, kwargs);
      case "ARRAY_SIZE":
        Expr isSingleRowLiteral = new Expr.BooleanLiteral(argScalars.get(0));
        List<Expr> all_operands = new ArrayList<>(operands);
        all_operands.add(isSingleRowLiteral);
        return ExprKt.BodoSQLKernel(fnName.toLowerCase(Locale.ROOT), all_operands, List.of());
      default:
        throw new BodoSQLCodegenException(
            String.format(Locale.ROOT, "Unsupported nested Array function: %s", fnName));
    }
  }

  /**
   * Constructs the Expression to make a call to the variadic function ARRAY_CONSTRUCT.
   *
   * @param codeExprs the Python expressions to calculate the arguments
   * @param isSingleRow Does the data operate on/output a single row?
   * @param argScalars Whether each argument is a scalar or a column
   * @return Expr containing the code generated for the relational expression.
   */
  public static Expr visitArrayConstruct(
      List<Expr> codeExprs, boolean isSingleRow, List<Boolean> argScalars) {
    ArrayList<Expr> scalarExprs = new ArrayList();
    for (Boolean isScalar : argScalars) {
      scalarExprs.add(new Expr.BooleanLiteral(isScalar));
    }
    return new Expr.Call(
        "bodo.libs.bodosql_array_kernels.array_construct",
        List.of(new Expr.Tuple(codeExprs), new Expr.Tuple(scalarExprs)));
  }

  /**
   * Constructs the Expression to make a call to the variadic function OBJECT_DELETE.
   *
   * @param codeExprs the Python expressions to calculate the arguments
   * @return Expr containing the code generated for the relational expression.
   */
  public static Expr visitObjectDelete(List<Expr> codeExprs) {
    return new Expr.Call(
        "bodo.libs.bodosql_array_kernels.object_delete", List.of(new Expr.Tuple(codeExprs)));
  }

  protected Expr visitNestedArrayFunc(
      String fnName, List<Expr> operands, List<Boolean> argScalars) {
    return visitNestedArrayFunc(fnName, operands, argScalars, List.of());
  }

  protected Expr visitGenericFuncOp(RexCall fnOperation, boolean isSingleRow) {
    String fnName = fnOperation.getOperator().toString();
    ArrayList<Boolean> argScalars = new ArrayList();
    for (RexNode operand : fnOperation.operands) {
      argScalars.add(isOperandScalar(operand));
    }
    // Handle functions that do not care about nulls separately
    if (fnName == "COALESCE"
        || fnName == "NVL"
        || fnName == "NVL2"
        || fnName == "ARRAY_CONSTRUCT"
        || fnName == "BOOLAND"
        || fnName == "BOOLOR"
        || fnName == "BOOLXOR"
        || fnName == "BOOLNOT"
        || fnName == "EQUAL_NULL"
        || fnName == "ZEROIFNULL"
        || fnName == "IFNULL"
        || fnName == "IF"
        || fnName == "IFF"
        || fnName == "DECODE"
        || fnName == "OBJECT_CONSTRUCT_KEEP_NULL"
        || fnName == "HASH") {
      return visitNullIgnoringGenericFunc(fnOperation, isSingleRow, argScalars);
    }

    // Extract all inputs to the current function.
    List<Expr> operands = visitList(fnOperation.operands);

    DateTimeType dateTimeExprType1;
    DateTimeType dateTimeExprType2;
    boolean isTime;
    boolean isDate;
    String unit;
    switch (fnOperation.getOperator().kind) {
      case MOD:
        return getNumericFnCode(fnName, operands);
      case GREATEST:
      case LEAST:
        return visitLeastGreatest(fnOperation.getOperator().toString(), operands);
      case TIMESTAMP_ADD:
        // Uses Calcite parser, accepts both quoted and unquoted time units
        dateTimeExprType1 = getDateTimeDataType(fnOperation.getOperands().get(2));
        unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType1);
        assert isScalar(fnOperation.operands.get(0));
        return generateSnowflakeDateAddCode(operands.subList(1, operands.size()), unit);
      case TIMESTAMP_DIFF:
        assert operands.size() == 3;
        dateTimeExprType1 = getDateTimeDataType(fnOperation.getOperands().get(1));
        dateTimeExprType2 = getDateTimeDataType(fnOperation.getOperands().get(2));
        if ((dateTimeExprType1 == DateTimeType.TIME) != (dateTimeExprType2 == DateTimeType.TIME)) {
          throw new BodoSQLCodegenException(
              "Invalid type of arguments to TIMESTAMPDIFF: cannot mix date/timestamp with time.");
        }
        unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType1);
        return generateDateDiffFnInfo(unit, operands.get(1), operands.get(2));
      case TRIM:
        assert operands.size() == 3;
        // NOTE: Operand 0 is one of BOTH/LEADING/TRAILING. We should make sure this is
        // remapped to the proper function name.
        assert fnOperation.operands.get(0) instanceof RexLiteral;
        RexLiteral literal = (RexLiteral) fnOperation.operands.get(0);
        String argValue = literal.getValue2().toString().toUpperCase(Locale.ROOT);
        if (argValue.equals("BOTH")) {
          fnName = "trim";
        } else if (argValue.equals("LEADING")) {
          fnName = "ltrim";
        } else {
          assert argValue.equals("TRAILING");
          fnName = "rtrim";
        }
        // Calcite expects: TRIM(<chars> FROM <expr>>) or TRIM(<chars>, <expr>)
        // However, Snowflake/BodoSQL expects: TRIM(<expr>, <chars>)
        // So we just need to swap the arguments here.
        return visitTrimFunc(fnName, operands.get(2), operands.get(1));
      case NULLIF:
        assert operands.size() == 2;
        return visitNullIfFunc(operands);
      case POSITION:
        return visitPosition(operands);
      case RANDOM:
        return generateRandomFnInfo(input, isSingleRow);
      case OTHER:
      case OTHER_FUNCTION:
        /* If sqlKind = other function, the only recourse is to match on the name of the function. */
        switch (fnName) {
          case "CEIL":
          case "FLOOR":
            return genFloorCeilCode(fnName, operands);
            // TODO (allai5): update this in a future PR for clean-up so it re-uses the
            // SQLLibraryOperator definition.
          case "LTRIM":
          case "RTRIM":
            if (operands.size() == 1) { // no optional characters to be trimmed
              return visitTrimFunc(
                  fnName, operands.get(0), new Expr.StringLiteral(" ")); // remove spaces by default
            } else if (operands.size() == 2) {
              return visitTrimFunc(fnName, operands.get(0), operands.get(1));
            } else {
              throw new BodoSQLCodegenException(
                  "Invalid number of arguments to TRIM: must be either 1 or 2.");
            }
          case "WIDTH_BUCKET":
            {
              int numOps = operands.size();
              assert numOps == 4 : "WIDTH_BUCKET takes 4 arguments, but found " + numOps;
              return new Expr.Call("bodo.libs.bodosql_array_kernels.width_bucket", operands);
            }
          case "HAVERSINE":
            {
              assert operands.size() == 4;
              return new Expr.Call("bodo.libs.bodosql_array_kernels.haversine", operands);
            }
          case "DIV0":
            {
              assert operands.size() == 2 && fnOperation.operands.size() == 2;
              return new Expr.Call("bodo.libs.bodosql_array_kernels.div0", operands);
            }
          case "NULLIFZERO":
            assert operands.size() == 1;
            return visitNullIfFunc(List.of(operands.get(0), new Expr.IntegerLiteral(0)));
          case "DATEADD":
          case "TIMEADD":
            // If DATEADD receives 3 arguments, use the Snowflake DATEADD.
            // Otherwise, fall back to the normal DATEADD. TIMEADD and TIMESTAMPADD are aliases.
            if (operands.size() == 3) {
              dateTimeExprType1 = getDateTimeDataType(fnOperation.getOperands().get(2));
              unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType1);
              assert isScalar(fnOperation.operands.get(0));
              return generateSnowflakeDateAddCode(operands.subList(1, operands.size()), unit);
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
                arg0 =
                    visitDynamicCast(
                        operands.get(0),
                        inputType,
                        outputType,
                        isSingleRow || isScalar(fnOperation.operands.get(0)));
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
            return generateStrToDateCode(
                operands.get(0), isScalar(fnOperation.operands.get(0)), operands.get(1).emit());
          case "TIME_SLICE":
            return generateTimeSliceFnCode(operands, 0);
          case "TIMESTAMP":
          case "DATE":
          case "TO_DATE":
          case "TRY_TO_DATE":
          case "TO_TIMESTAMP":
          case "TO_TIMESTAMP_NTZ":
          case "TO_TIMESTAMP_LTZ":
          case "TO_TIMESTAMP_TZ":
          case "TRY_TO_TIMESTAMP":
          case "TRY_TO_TIMESTAMP_NTZ":
          case "TRY_TO_TIMESTAMP_LTZ":
          case "TRY_TO_TIMESTAMP_TZ":
          case "TRY_TO_BOOLEAN":
          case "TO_BOOLEAN":
          case "TRY_TO_BINARY":
          case "TO_BINARY":
          case "TO_CHAR":
          case "TO_VARCHAR":
          case "TO_NUMBER":
          case "TO_NUMERIC":
          case "TO_DECIMAL":
          case "TRY_TO_NUMBER":
          case "TRY_TO_NUMERIC":
          case "TRY_TO_DECIMAL":
          case "TO_DOUBLE":
          case "TRY_TO_DOUBLE":
          case "TIME":
          case "TO_TIME":
          case "TRY_TO_TIME":
          case "TO_ARRAY":
          case "ARRAY_TO_STRING":
            return visitCastFunc(fnOperation, operands);
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
          case "ATAN2":
            return getTrigFnCode(fnName, operands);
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
          case "POWER":
          case "POW":
          case "BITAND":
          case "BITOR":
          case "BITXOR":
          case "BITSHIFTLEFT":
          case "BITSHIFTRIGHT":
          case "GETBIT":
            return getNumericFnCode(fnName, operands);
          case "TRUNC":
          case "TRUNCATE":
          case "ROUND":
            List<Expr> args = new ArrayList<>();
            args.addAll(operands);
            if (operands.size() == 1) {
              // If no value is specified by, default to 0
              args.add(new Expr.IntegerLiteral(0));
            }
            assert args.size() == 2;
            return getNumericFnCode(fnName, args);

          case "LOG":
            return generateLogFnInfo(operands);
          case "CONV":
            assert operands.size() == 3;
            return new Expr.Call("bodo.libs.bodosql_array_kernels.conv", operands);
          case "RAND":
            return new Expr.Call("np.random.rand");
          case "PI":
            return new Expr.Raw("np.pi");
          case "UNIFORM":
            assert operands.size() == 3;
            expectScalarArgument(fnOperation.operands.get(0), "UNIFORM", "lo");
            expectScalarArgument(fnOperation.operands.get(1), "UNIFORM", "hi");
            return generateUniformFnInfo(operands);
          case "CONCAT":
            return generateConcatCode(operands, List.of(), fnOperation.operands.get(0).getType());
          case "CONCAT_WS":
            assert operands.size() >= 2;
            return generateConcatWSCode(
                operands.get(0), operands.subList(1, operands.size()), List.of());
          case "GETDATE":
          case "CURRENT_TIMESTAMP":
          case "NOW":
          case "LOCALTIMESTAMP":
          case "SYSTIMESTAMP":
          case "CURRENT_TIME":
          case "LOCALTIME":
          case "SYSDATE":
          case "UTC_TIMESTAMP":
          case "UTC_DATE":
          case "CURRENT_DATE":
          case "CURRENT_DATABASE":
          case "CURDATE":
            assert operands.size() == 0;
            return visitGeneralContextFunction(fnOperation);
          case "MAKEDATE":
            assert operands.size() == 2;
            return generateMakeDateInfo(operands.get(0), operands.get(1));
          case "DATE_FORMAT":
            if (!(operands.size() == 2) && isScalar(fnOperation.operands.get(1))) {
              throw new BodoSQLCodegenException(
                  "Error, invalid argument types passed to DATE_FORMAT");
            }
            if (!(fnOperation.operands.get(1) instanceof RexLiteral)) {
              throw new BodoSQLCodegenException(
                  "Error DATE_FORMAT(): 'Format' must be a string literal");
            }
            return generateDateFormatCode(operands.get(0), operands.get(1));
          case "CONVERT_TIMEZONE":
            assert operands.size() == 2 || operands.size() == 3;
            return generateConvertTimezoneCode(
                operands, BodoTZInfo.getDefaultTZInfo(this.typeSystem));
          case "YEARWEEK":
            assert operands.size() == 1;
            return getYearWeekFnInfo(operands.get(0));
          case "MONTHS_BETWEEN":
            assert operands.size() == 2;
            return ExprKt.BodoSQLKernel("months_between", operands, List.of());
          case "ADD_MONTHS":
            assert operands.size() == 2;
            return ExprKt.BodoSQLKernel("add_months", operands, List.of());
          case "MONTHNAME":
          case "MONTH_NAME":
          case "DAYNAME":
          case "WEEKDAY":
          case "YEAROFWEEKISO":
            assert operands.size() == 1;
            if (getDateTimeDataType(fnOperation.getOperands().get(0)) == DateTimeType.TIME)
              throw new BodoSQLCodegenException("Time object is not supported by " + fnName);
            return getSingleArgDatetimeFnInfo(fnName, operands.get(0).emit());
          case "YEAROFWEEK":
            assert operands.size() == 1;
            args = new ArrayList<>(operands);
            args.add(new Expr.IntegerLiteral(this.weekStart));
            args.add(new Expr.IntegerLiteral(this.weekOfYearPolicy));
            return ExprKt.BodoSQLKernel("yearofweek", args, List.of());
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
          case "TO_DAYS":
            return generateToDaysCode(operands.get(0));
          case "TO_SECONDS":
            return generateToSecondsCode(operands.get(0));
          case "FROM_DAYS":
            return generateFromDaysCode(operands.get(0));
          case "DATE_FROM_PARTS":
          case "DATEFROMPARTS":
            assert operands.size() == 3;
            return generateDateTimeTypeFromPartsCode(fnName, operands, None.INSTANCE);
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
            Expr tzExpr = None.INSTANCE;
            if (fnOperation.getType() instanceof TZAwareSqlType) {
              tzExpr = ((TZAwareSqlType) fnOperation.getType()).getTZInfo().getZoneExpr();
            }
            return generateDateTimeTypeFromPartsCode(fnName, operands, tzExpr);
          case "UNIX_TIMESTAMP":
            return generateUnixTimestamp();
          case "FROM_UNIXTIME":
            return generateFromUnixTimeCode(operands.get(0));
          case "JSON_EXTRACT_PATH_TEXT":
          case "OBJECT_KEYS":
            return visitJsonFunc(fnName, operands);
          case "OBJECT_DELETE":
            return visitObjectDelete(operands);
          case "RLIKE":
          case "REGEXP_LIKE":
          case "REGEXP_COUNT":
          case "REGEXP_REPLACE":
          case "REGEXP_SUBSTR":
          case "REGEXP_INSTR":
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
          case "FORMAT":
          case "REPEAT":
          case "STRCMP":
          case "RIGHT":
          case "LEFT":
          case "CONTAINS":
          case "INSTR":
          case "STARTSWITH":
          case "ENDSWITH":
          case "RPAD":
          case "LPAD":
          case "SPLIT_PART":
          case "MID":
          case "SUBSTRING_INDEX":
          case "TRANSLATE3":
          case "REPLACE":
          case "SUBSTR":
          case "INSERT":
          case "POSITION":
          case "CHARINDEX":
          case "STRTOK":
          case "STRTOK_TO_ARRAY":
          case "SPLIT":
          case "EDITDISTANCE":
          case "JAROWINKLER_SIMILARITY":
          case "INITCAP":
          case "SHA2":
          case "SHA2_HEX":
          case "MD5":
          case "MD5_HEX":
          case "HEX_ENCODE":
          case "HEX_DECODE_STRING":
          case "HEX_DECODE_BINARY":
          case "TRY_HEX_DECODE_STRING":
          case "TRY_HEX_DECODE_BINARY":
          case "BASE64_ENCODE":
          case "BASE64_DECODE_STRING":
          case "TRY_BASE64_DECODE_STRING":
          case "BASE64_DECODE_BINARY":
          case "TRY_BASE64_DECODE_BINARY":
            return visitStringFunc(fnOperation, operands);
          case "DATE_TRUNC":
            dateTimeExprType1 = getDateTimeDataType(fnOperation.getOperands().get(1));
            unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType1);
            return generateDateTruncCode(unit, operands.get(1));
          case "MICROSECOND":
          case "NANOSECOND":
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
          case "EPOCH_SECOND":
          case "EPOCH_MILLISECOND":
          case "EPOCH_MICROSECOND":
          case "EPOCH_NANOSECOND":
          case "TIMEZONE_HOUR":
          case "TIMEZONE_MINUTE":
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
            return generateExtractCode(
                fnName, operands.get(0), isTime, isDate, this.weekStart, this.weekOfYearPolicy);
          case "REGR_VALX":
          case "REGR_VALY":
            return getCondFuncCode(fnName, operands);
          case "ARRAY_CONTAINS":
          case "ARRAY_POSITION":
          case "ARRAY_EXCEPT":
          case "ARRAY_INTERSECTION":
          case "ARRAY_CAT":
          case "ARRAYS_OVERLAP":
          case "ARRAY_SIZE":
            return visitNestedArrayFunc(fnName, operands, argScalars);
        }
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Function: " + fnOperation.getOperator().toString() + " not supported");
    }
  }

  @Override
  public Expr visitOver(RexOver over) {
    throw new BodoSQLCodegenException(
        "Internal Error: Calcite Plan Produced an Unsupported RexOver: " + over.getOperator());
  }

  @Override
  public Expr visitCorrelVariable(RexCorrelVariable correlVariable) {
    throw unsupportedNode();
  }

  @Override
  public Expr visitDynamicParam(RexDynamicParam dynamicParam) {
    throw unsupportedNode();
  }

  private Expr visitNamedParam(RexNamedParam namedParam) {
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

    /**
     * List of functions generated by this scalar context. This is needed for nested case statements
     * to maintain control flow.
     */
    private final @NotNull List<Function> closures = new ArrayList<>();

    /** Variable names used in generated closures. */
    private final @NotNull List<? extends Variable> closurevars;

    public ScalarContext(
        @NotNull PandasCodeGenVisitor visitor,
        @NotNull Module.Builder builder,
        @NotNull RelDataTypeSystem typeSystem,
        int nodeId,
        @NotNull BodoEngineTable input,
        @NotNull List<? extends Expr> localRefs,
        @NotNull List<? extends Variable> closurevars) {
      super(visitor, builder, typeSystem, nodeId, input, localRefs);
      this.closurevars = closurevars;
    }

    @Override
    public Expr visitLiteral(RexLiteral literal) {
      return LiteralCodeGen.generateLiteralCode(literal, true, visitor);
    }

    @Override
    protected Expr visitCastScan(RexCall operation) {
      return visitCastScan(operation, true, List.of());
    }

    protected Boolean isOperandScalar(RexNode operand) {
      return true;
    }

    /**
     * Generating code in a scalar context requires building a closure that will be "bubbled up" to
     * the original RexToPandasTranslator and then generating an expression that looks like var =
     * func(...).
     *
     * <p>This updates closures in and the builder to generate the function call.
     *
     * @return The final Variable for the output of the closure.
     */
    @Override
    protected Expr visitCaseOp(RexCall node) {
      // Generate the frame for the closure.
      Frame closureFrame =
          visitCaseOperands(
              this,
              node.getOperands(),
              List.of(builder.getSymbolTable().genGenericTempVar()),
              true);
      Variable funcVar = builder.getSymbolTable().genClosureVar();
      Function closure = new Function(funcVar.emit(), this.closurevars, closureFrame);
      closures.add(closure);
      return new Expr.Call(funcVar, this.closurevars);
    }

    @Override
    protected Expr visitInternalOp(RexCall node) {
      return visitInternalOp(node, true);
    }

    @Override
    protected Variable visitGeneralContextFunction(RexCall fnOperation) {
      // Case statements are not called consistently on all ranks, so we cannot
      // generate code that tries to make all ranks generate a consistent output.
      return visitGeneralContextFunction(fnOperation, false);
    }

    @Override
    protected Expr visitGenericFuncOp(RexCall fnOperation) {
      return visitGenericFuncOp(fnOperation, true);
    }

    @Override
    public Expr visitOver(RexOver over) {
      throw new BodoSQLCodegenException(
          "Internal Error: Calcite Plan Produced an Unsupported RexOver: " + over.getOperator());
    }

    /** @return The closures generated by this scalar context. */
    public List<Function> getClosures() {
      return closures;
    }
  }

  protected BodoSQLCodegenException unsupportedNode() {
    return new BodoSQLCodegenException(
        "Internal Error: Calcite Plan Produced an Unsupported RexNode");
  }
}
