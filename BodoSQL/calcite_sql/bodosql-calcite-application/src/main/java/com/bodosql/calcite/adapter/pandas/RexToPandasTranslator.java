package com.bodosql.calcite.adapter.pandas;

import static com.bodosql.calcite.application.BodoSQLCodeGen.BinOpCodeGen.generateBinOpCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CastCodeGen.generateCastCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CondOpCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ConversionCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DateAddCodeGen.generateMySQLDateAddCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DateAddCodeGen.generateSnowflakeDateAddCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DateDiffCodeGen.generateDateDiffFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ExtractCodeGen.generateDatePart;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ExtractCodeGen.generateExtractCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.JsonCodeGen.generateJsonTwoArgsInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.LikeCodeGen.generateLikeCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.generateTryToNumberCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.PostfixOpCodeGen.generatePostfixOpCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.PrefixOpCodeGen.generatePrefixOpCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.RegexpCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SinceEpochFnCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.generateInitcapInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.TrigCodeGen.getDoubleArgTrigFnInfo;
import static com.bodosql.calcite.application.BodoSQLCodeGen.TrigCodeGen.getSingleArgTrigFnInfo;
import static com.bodosql.calcite.application.Utils.Utils.*;
import static com.bodosql.calcite.application.Utils.Utils.renameExprsList;

import com.bodosql.calcite.application.*;
import com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen;
import com.bodosql.calcite.application.BodoSQLCodeGen.LiteralCodeGen;
import com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen;
import com.bodosql.calcite.application.Utils.BodoCtx;
import com.bodosql.calcite.ir.Dataframe;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Module;
import com.bodosql.calcite.ir.Variable;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.*;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.fun.*;
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
    String refValue =
        String.format(
            "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(%s, %d)",
            input.getVariable().getName(), inputRef.getIndex());
    return new Expr.Raw(refValue);
  }

  @Override
  public Expr visitLocalRef(RexLocalRef localRef) {
    throw unsupportedNode();
  }

  @Override
  public Expr visitLiteral(RexLiteral literal) {
    String code =
        LiteralCodeGen.generateLiteralCode(literal, false, visitor, builder.getUseDateRuntime());
    return new Expr.Raw(code);
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
    String opName = node.getOperator().getName();
    List<RexNode> operands = node.getOperands();
    RexNode colIndex = operands.get(0);
    Expr arg = colIndex.accept(this);
    String argCode = arg.emit();
    RexNode patternNode = operands.get(1);

    // The regular expression functions only support literal patterns
    if ((opName.equals("REGEXP") || opName.equals("RLIKE"))
        && !(patternNode instanceof RexLiteral)) {
      throw new BodoSQLCodegenException(
          String.format("%s Error: Pattern must be a string literal", opName));
    }
    Expr pattern = patternNode.accept(this);
    String sqlPattern = pattern.emit();
    // If escape is missing use the empty string.
    String sqlEscape = "''";
    if (operands.size() == 3) {
      RexNode escapeNode = operands.get(2);
      Expr escape = escapeNode.accept(this);
      sqlEscape = escape.emit();
    }
    /* Assumption: Typing in LIKE requires this to be a string type. */
    String likeCode = generateLikeCode(opName, argCode, sqlPattern, sqlEscape);
    return new Expr.Raw(likeCode);
  }

  protected Expr visitCaseOp(RexCall node) {
    return visitCaseOp(node, false);
  }

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
    // are inside another apply. We choose to generate apply code in this case because
    // we only compile single basic blocks.
    boolean generateApply = !isSingleRow;

    // The body of case may require wrapping everything inside a global variable.
    // In the future this all needs to be wrapped inside the original builder, but we
    // can't support that yet.
    Module.Builder oldBuilder = this.builder;
    Module.Builder innerBuilder = this.builder;

    if (generateApply) {
      // If we're generating an apply, the input set of columns to add should be empty,
      // since we only add columns to the colsToAddList when inside an apply.
      assert ctx.getColsToAddList().size() == 0;
      // If we generate an apply we need to write the generated if/else
      innerBuilder = new Module.Builder(this.builder);
    }

    RexToPandasTranslator localTranslator =
        new RexToPandasTranslator.ScalarContext(visitor, innerBuilder, typeSystem, nodeId, input);
    List<String> args = new ArrayList<>(operands.size());
    for (Expr expr : localTranslator.visitList(operands)) {
      args.add(expr.emit());
    }

    String inputVar = input.getVariable().getName();
    if (generateApply && localTranslator.ctx.getColsToAddList().size() > 0) {
      // If we do generate an apply, add the columns that we need to the dataframe
      // and change the variables to reference the new dataframe
      List<String> colNames = input.getRel().getRowType().getFieldNames();
      final String indent = getBodoIndent();
      String tmp_case_name = "tmp_case_" + builder.getSymbolTable().genDfVar().emit();
      this.builder.append(
          indent
              + tmp_case_name
              + " = "
              + generateCombinedDf(inputVar, colNames, localTranslator.ctx.getColsToAddList())
              + "\n");
      args = renameExprsList(args, inputVar, tmp_case_name);
      inputVar = tmp_case_name;
      // get column names including the added columns
      List<String> newColNames = new ArrayList<>();
      for (String col : colNames) newColNames.add(col);
      for (String col : localTranslator.ctx.getColsToAddList()) newColNames.add(col);
    } else if (!generateApply) {
      // If we're not the top level apply, we need to pass back the information so that it is
      // properly handled
      // by the actual top level apply
      // null columns are handled by the cond itself, so don't need to pass those back
      ctx.getNamedParams().addAll(localTranslator.ctx.getNamedParams());
      ctx.getColsToAddList().addAll(localTranslator.ctx.getColsToAddList());
      ctx.getUsedColumns().addAll(localTranslator.ctx.getUsedColumns());
    }

    String codeExpr =
        generateCaseCode(
            args,
            generateApply,
            localTranslator.ctx,
            inputVar,
            node.getType(),
            visitor,
            oldBuilder,
            innerBuilder,
            builder.getUseDateRuntime());

    return new Expr.Raw(codeExpr);
  }

  protected Expr visitCastScan(RexCall operation) {
    RelDataType inputType = operation.operands.get(0).getType();
    RelDataType outputType = operation.getType();

    boolean outputScalar =
        visitor.exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(operation, nodeId))
            == BodoSQLExprType.ExprType.SCALAR;
    List<Expr> args = visitList(operation.operands);
    Expr child = args.get(0);
    String exprCode =
        generateCastCode(
            child.emit(), inputType, outputType, outputScalar, builder.getUseDateRuntime());
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
    //  * substring length
    //  All of these values can be both scalars and columns
    assert node.operands.size() == 3;
    List<BodoSQLExprType.ExprType> exprTypes = new ArrayList<>(node.operands.size());
    for (RexNode operand_node : node.operands) {
      exprTypes.add(
          visitor.exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(operand_node, nodeId)));
    }
    List<Expr> operands = visitList(node.operands);
    String fnName = node.getOperator().getName();
    String code =
        getThreeArgStringFnInfo(
            fnName, operands.get(0).emit(), operands.get(1).emit(), operands.get(2).emit());
    return new Expr.Raw(code);
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
      ctx.getColsToAddList().addAll(localCtx.getColsToAddList());
      ctx.getNamedParams().addAll(localCtx.getNamedParams());
      ctx.getUsedColumns().addAll(localCtx.getUsedColumns());

      return result;
    }

    // Extract all inputs to the current function.
    List<Expr> operands = visitList(fnOperation.operands);

    String expr;
    String strExpr;
    DatetimeFnCodeGen.DateTimeType dateTimeExprType;
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
      case MOD:
        return new Expr.Raw(
            getDoubleArgNumericFnInfo(
                fnOperation.getOperator().toString(),
                operands.get(0).emit(),
                operands.get(1).emit()));
      case TIMESTAMP_ADD:
        // Uses Calcite parser, accepts both quoted and unquoted time units
        dateTimeExprType = getDateTimeExprType(fnOperation.getOperands().get(2));
        unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType);
        assert exprTypes.get(0) == BodoSQLExprType.ExprType.SCALAR;
        return new Expr.Raw(generateSnowflakeDateAddCode(operands, unit));
      case TIMESTAMP_DIFF:
        assert operands.size() == 3;
        dateTimeExprType = getDateTimeExprType(fnOperation.getOperands().get(1));
        if (dateTimeExprType != getDateTimeExprType(fnOperation.getOperands().get(2)))
          throw new BodoSQLCodegenException(
              "Invalid type of arguments to TIMESTAMPDIFF: arg1 and arg2 must be the same type.");
        unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType);
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
              dateTimeExprType = getDateTimeExprType(fnOperation.getOperands().get(2));
              unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType);
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
                  && getDateTimeExprType(fnOperation.getOperands().get(0))
                      .equals(DateTimeType.TIME)) {
                throw new BodoSQLCodegenException("Cannot add/subtract days from TIME");
              }
              Set<SqlTypeName> DATE_INTERVAL_TYPES =
                  Sets.immutableEnumSet(SqlTypeName.INTERVAL_YEAR_MONTH,
                      SqlTypeName.INTERVAL_YEAR,
                      SqlTypeName.INTERVAL_MONTH,
                      SqlTypeName.INTERVAL_WEEK,
                      SqlTypeName.INTERVAL_DAY);
              boolean is_date_interval = DATE_INTERVAL_TYPES.contains(
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
                        exprTypes.get(0) == BodoSQLExprType.ExprType.SCALAR || isSingleRow,
                        builder.getUseDateRuntime());
                arg0 = new Expr.Raw(casted_expr);
              }
              // add/minus a date interval to a date object should return a date object
              if (is_date_interval &&
                  getDateTimeExprType(fnOperation.getOperands().get(0)) == DateTimeType.DATE) {
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
              dateTimeExprType = getDateTimeExprType(fnOperation.getOperands().get(1));
              if (dateTimeExprType != getDateTimeExprType(fnOperation.getOperands().get(0)))
                throw new BodoSQLCodegenException(
                    "Invalid type of arguments to DATEDIFF: arg0 and arg1 must be the same type.");
            } else if (operands.size() == 3) { // this is the Snowflake option
              unit = operands.get(0).emit();
              arg1 = operands.get(1);
              arg2 = operands.get(2);
              dateTimeExprType = getDateTimeExprType(fnOperation.getOperands().get(1));
              if (dateTimeExprType != getDateTimeExprType(fnOperation.getOperands().get(2)))
                throw new BodoSQLCodegenException(
                    "Invalid type of arguments to DATEDIFF: arg1 and arg2 must be the same type.");
            } else {
              throw new BodoSQLCodegenException(
                  "Invalid number of arguments to DATEDIFF: must be 2 or 3.");
            }
            unit = standardizeTimeUnit(fnName, unit, dateTimeExprType);
            return generateDateDiffFnInfo(unit, arg1, arg2);
          case "TIMEDIFF":
            assert operands.size() == 3;
            dateTimeExprType = getDateTimeExprType(fnOperation.getOperands().get(1));
            if (dateTimeExprType != getDateTimeExprType(fnOperation.getOperands().get(2)))
              throw new BodoSQLCodegenException(
                  "Invalid type of arguments to TIMEDIFF: arg1 and arg2 must be the same type.");
            unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType);
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
            return generateToCharFnCode(
                operands,
                fnName,
                fnOperation.getOperands().get(0).getType().getSqlTypeName() == SqlTypeName.DATE);
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
            return generateLogFnInfo(operands, exprTypes, isSingleRow);
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
            if (getDateTimeExprType(fnOperation.getOperands().get(0)) == DateTimeType.TIME)
              throw new BodoSQLCodegenException("Time object is not supported by " + fnName);
            return getSingleArgDatetimeFnInfo(fnName, operands.get(0).emit());
          case "LAST_DAY":
            dateTimeExprType = getDateTimeExprType(fnOperation.getOperands().get(0));
            if (dateTimeExprType == DateTimeType.TIME)
              throw new BodoSQLCodegenException("Time object is not supported by " + fnName);
            if (operands.size() == 2) {
              unit = standardizeTimeUnit(fnName, operands.get(1).emit(), dateTimeExprType);
              if (unit.equals("day") || TIME_PART_UNITS.contains(unit))
                throw new BodoSQLCodegenException(operands.get(1).emit() + " is not a valid time unit for " + fnName);
              return generateLastDayCode(operands.get(0).emit(), unit);
            }
            assert  operands.size() == 1;
            // the default time unit is month
            return generateLastDayCode(operands.get(0).emit(), "month");
          case "NEXT_DAY":
          case "PREVIOUS_DAY":
            assert operands.size() == 2;
            if (getDateTimeExprType(fnOperation.getOperands().get(0)) == DateTimeType.TIME)
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
          case "SUBSTR":
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
            dateTimeExprType = getDateTimeExprType(fnOperation.getOperands().get(1));
            unit = standardizeTimeUnit(fnName, operands.get(0).emit(), dateTimeExprType);
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
      ctx.getUsedColumns().add(inputRef.getIndex());
      // NOTE: Codegen for bodosql_case_placeholder() expects column value accesses
      // (e.g. bodo.utils.indexing.scalar_optional_getitem(T1_1, i))
      String refValue =
          "bodo.utils.indexing.scalar_optional_getitem("
              + input.getVariable().getName()
              + "_"
              + inputRef.getIndex()
              + ", i)";
      return new Expr.Raw(refValue);
    }

    @Override
    public Expr visitLiteral(RexLiteral literal) {
      String code =
          LiteralCodeGen.generateLiteralCode(literal, true, visitor, builder.getUseDateRuntime());
      return new Expr.Raw(code);
    }

    @Override
    protected Expr visitCastScan(RexCall operation) {
      RelDataType inputType = operation.operands.get(0).getType();
      RelDataType outputType = operation.getType();

      List<Expr> args = visitList(operation.operands);
      Expr child = args.get(0);
      String exprCode =
          generateCastCode(child.emit(), inputType, outputType, true, builder.getUseDateRuntime());
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
