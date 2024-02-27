package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.DateAddCodeGen.generateMySQLDateAddCode;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import com.bodosql.calcite.ir.Module.Builder;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.Op.Assign;
import com.bodosql.calcite.ir.Variable;
import com.google.common.collect.Sets;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import kotlin.Pair;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.type.IntervalSqlType;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.TZAwareSqlType;

/**
 * Class that returns the generated code for a BinOp expression after all inputs have been visited.
 */
public class BinOpCodeGen {

  /**
   * Function that return the necessary generated code for a BinOp Call. This function may have more
   * than two arguments because repeated use of the same operator are grouped into the same Node.
   *
   * @param args The arguments to the binop.
   * @param binOp The Binary operator to apply to each pair of arguments.
   * @param argDataTypes List of SQL data types for the input to the binary operation.
   * @param builder The Module.Builder for appending generated code. This is used for generating
   *     intermediate variables.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @param argScalars Whether each argument is a scalar or a column
   * @return The code generated for the BinOp call.
   */
  public static Expr generateBinOpCode(
      List<Expr> args,
      SqlOperator binOp,
      List<RelDataType> argDataTypes,
      Builder builder,
      List<Pair<String, Expr>> streamingNamedArgs,
      List<Boolean> argScalars) {
    SqlKind binOpKind = binOp.getKind();
    // Handle the Datetime functions that are either tz-aware or tz-naive
    if (argDataTypes.size() == 2
        && (binOpKind.equals(SqlKind.PLUS)
            || binOpKind.equals(SqlKind.MINUS)
            || binOpKind.equals(SqlKind.TIMES))) {
      // If we are passing a pair of arguments to +/-, check if we are adding a tz-aware
      // value with an interval. If so we have to take a specialized code path.
      boolean isArg0TZAware = argDataTypes.get(0) instanceof TZAwareSqlType;
      boolean isArg1TZAware = argDataTypes.get(1) instanceof TZAwareSqlType;
      // Check if we have Datetime or Date data
      SqlTypeName arg0TypeName = argDataTypes.get(0).getSqlTypeName();
      SqlTypeName arg1TypeName = argDataTypes.get(1).getSqlTypeName();

      boolean isArg0Datetime =
          arg0TypeName.equals(SqlTypeName.TIMESTAMP)
              || arg0TypeName.equals(SqlTypeName.DATE)
              || arg0TypeName.equals(SqlTypeName.TIME);
      boolean isArg1Datetime =
          arg1TypeName.equals(SqlTypeName.TIMESTAMP)
              || arg1TypeName.equals(SqlTypeName.DATE)
              || arg1TypeName.equals(SqlTypeName.TIME);
      Set<SqlTypeName> DATE_INTERVAL_TYPES =
          Sets.immutableEnumSet(
              SqlTypeName.INTERVAL_YEAR_MONTH,
              SqlTypeName.INTERVAL_YEAR,
              SqlTypeName.INTERVAL_MONTH,
              SqlTypeName.INTERVAL_DAY);

      boolean isArg0Interval = argDataTypes.get(0) instanceof IntervalSqlType;
      boolean isArg1Interval = argDataTypes.get(1) instanceof IntervalSqlType;
      boolean isAddOrSub = binOpKind.equals(SqlKind.PLUS) || binOpKind.equals(SqlKind.MINUS);
      if (isArg0Interval && isArg1Interval) {
        assert isAddOrSub;
        return genIntervalAddCode(args, binOpKind);
      } else if ((isArg0TZAware && isArg1Interval) || (isArg1TZAware && isArg0Interval)) {
        assert isAddOrSub;
        return genTZAwareIntervalArithCode(args, binOpKind, isArg0TZAware);
      } else if (isArg0Datetime && isArg1Datetime) {
        assert binOpKind.equals(SqlKind.MINUS);
        return genDateSubCode(args);
      } else if (isArg0Datetime) { // timestamp/date +- interval
        assert isAddOrSub;
        if (arg0TypeName.equals(SqlTypeName.DATE) && DATE_INTERVAL_TYPES.contains(arg1TypeName)) {
          // add/minus a date interval to a date object should return a date object
          Expr arg1 = args.get(1);
          if (binOpKind.equals(SqlKind.MINUS))
            arg1 = ExprKt.bodoSQLKernel("negate", List.of(args.get(1)), List.of());
          return ExprKt.bodoSQLKernel(
              "add_date_interval_to_date", List.of(args.get(0), arg1), List.of());
        }
        return genDatetimeArithCode(args, binOpKind, isArg0Datetime, isArg1Interval);
      } else if (isArg1Datetime) { // inverval + timestamp/date
        assert binOpKind.equals(SqlKind.PLUS); // interval - timestamp/date is an invalid syntax
        if (arg1TypeName.equals(SqlTypeName.DATE) && DATE_INTERVAL_TYPES.contains(arg0TypeName)) {
          // add/minus a date interval to a date object should return a date object
          return ExprKt.bodoSQLKernel(
              "add_date_interval_to_date", List.of(args.get(1), args.get(0)), List.of());
        }
        return genDatetimeArithCode(args, binOpKind, isArg0Datetime, isArg0Interval);
      } else if ((isArg0Interval || isArg1Interval) && binOpKind.equals(SqlKind.TIMES)) {
        return genIntervalMultiplyCode(args, isArg0Interval);
      }
    }
    return generateBinOpCodeHelper(args, binOpKind, builder, streamingNamedArgs, argScalars);
  }
  /**
   * Helper function that returns the necessary generated code for a BinOp Call. This function may
   * have more than two arguments because repeated use of the same operator are grouped into the
   * same Node.
   *
   * @param args The arguments to the binop.
   * @param binOpKind The SqlKind of the binary operator
   * @param builder The build for intermediate variables.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @param argScalars Whether the arguments are scalars or columns.
   * @return The code generated for the BinOp call.
   */
  public static Variable generateBinOpCodeHelper(
      List<Expr> args,
      SqlKind binOpKind,
      Builder builder,
      List<Pair<String, Expr>> streamingNamedArgs,
      List<Boolean> argScalars) {
    final String fn;
    // Does the function support additional streaming arguments.
    boolean supportsStreamingArgs = false;
    boolean requiresScalarInfo = false;
    switch (binOpKind) {
      case EQUALS:
        fn = "equal";
        supportsStreamingArgs = true;
        break;
      case IS_NOT_DISTINCT_FROM:
      case NULL_EQUALS:
        fn = "equal_null";
        requiresScalarInfo = true;
        supportsStreamingArgs = true;
        break;
      case IS_DISTINCT_FROM:
        fn = "not_equal_null";
        requiresScalarInfo = true;
        supportsStreamingArgs = true;
        break;
      case NOT_EQUALS:
        fn = "not_equal";
        supportsStreamingArgs = true;
        break;
      case LESS_THAN:
        fn = "less_than";
        supportsStreamingArgs = true;
        break;
      case GREATER_THAN:
        fn = "greater_than";
        supportsStreamingArgs = true;
        break;
      case LESS_THAN_OR_EQUAL:
        fn = "less_than_or_equal";
        supportsStreamingArgs = true;
        break;
      case GREATER_THAN_OR_EQUAL:
        fn = "greater_than_or_equal";
        supportsStreamingArgs = true;
        break;
      case PLUS:
        fn = "add_numeric";
        break;
      case MINUS:
        fn = "subtract_numeric";
        break;
      case TIMES:
        fn = "multiply_numeric";
        break;
      case DIVIDE:
        fn = "divide_numeric";
        break;
      case AND:
        fn = "booland";
        break;
      case OR:
        fn = "boolor";
        break;
      default:
        throw new BodoSQLCodegenException(
            "Unsupported Operator, " + binOpKind + " specified in query.");
    }
    /** Create the function calls */
    Expr prevVar = args.get(0);
    Variable outputVar = null;

    for (int i = 1; i < args.size(); i++) {
      // Generate the function call. Pass in the additional streaming arguments if supported.
      ArrayList<Pair<String, Expr>> kwargs = new ArrayList();
      if (requiresScalarInfo) {
        kwargs.add(
            new Pair<String, Expr>("is_scalar_a", new Expr.BooleanLiteral(argScalars.get(i - 1))));
        kwargs.add(
            new Pair<String, Expr>("is_scalar_b", new Expr.BooleanLiteral(argScalars.get(i))));
      }
      if (supportsStreamingArgs) kwargs.addAll(streamingNamedArgs);
      Expr callExpr = ExprKt.bodoSQLKernel(fn, List.of(prevVar, args.get(i)), kwargs);
      // Generate a new variable
      outputVar = builder.getSymbolTable().genGenericTempVar();
      Op.Assign assign = new Assign(outputVar, callExpr);
      builder.add(assign);
      // Set the new prevVar
      prevVar = outputVar;
    }
    return outputVar;
  }

  /**
   * Generate code for Adding or Subtracting an two intervals.
   *
   * @param args List of length 2 with the generated code for the arguments.
   * @param binOp The SQLkind for the binop. Either SqlKind.PLUS or SqlKind.MINUS.
   * @return The generated code that creates the BodoSQL array kernel call.
   */
  public static Expr genIntervalAddCode(List<Expr> args, SqlKind binOp) {
    assert args.size() == 2;
    final Expr arg0 = args.get(0);
    Expr arg1 = args.get(1);
    if (binOp.equals(SqlKind.MINUS)) {
      // Negate the input for Minus
      arg1 = ExprKt.bodoSQLKernel("negate", List.of(arg1), List.of());
    } else {
      assert binOp.equals(SqlKind.PLUS);
    }
    return ExprKt.bodoSQLKernel("interval_add_interval", List.of(arg0, arg1), List.of());
  }

  /**
   * Generate code for Adding or Subtracting an interval from tz_aware data.
   *
   * @param args List of length 2 with the generated code for the arguments.
   * @param binOp The SQLkind for the binop. Either SqlKind.PLUS or SqlKind.MINUS.
   * @param isArg0TZAware Is arg0 the tz aware argument. This is used for generating standard code.
   * @return The generated code that creates the BodoSQL array kernel call.
   */
  public static Expr genTZAwareIntervalArithCode(
      List<Expr> args, SqlKind binOp, boolean isArg0TZAware) {
    assert args.size() == 2;
    final Expr arg0;
    Expr arg1;
    // Standardize the kernel to always put the tz aware argument
    // in the first slot to limit computation + simplify the kernels.
    // This is fine because + commutes.
    if (isArg0TZAware) {
      arg0 = args.get(0);
      arg1 = args.get(1);
    } else {
      arg0 = args.get(1);
      arg1 = args.get(0);
    }
    if (binOp.equals(SqlKind.MINUS)) {
      assert isArg0TZAware;
      // Negate the input for Minus
      arg1 = ExprKt.bodoSQLKernel("negate", List.of(arg1), List.of());
    } else {
      assert binOp.equals(SqlKind.PLUS);
    }
    return ExprKt.bodoSQLKernel("tz_aware_interval_add", List.of(arg0, arg1), List.of());
  }

  /**
   * Generate code for subtracting two dates.
   *
   * @param args List of length 2 with the generated code for the arguments.
   * @return The generated code that creates the BodoSQL array kernel call.
   */
  public static Expr genDateSubCode(List<Expr> args) {
    assert args.size() == 2;
    final Expr arg0 = args.get(0);
    final Expr arg1 = args.get(1);
    return ExprKt.bodoSQLKernel("diff_day", List.of(arg1, arg0), List.of());
  }

  /**
   * Generate code for Adding or Subtracting an interval from datetime data.
   *
   * @param args List of length 2 with the generated code for the arguments.
   * @param binOp The SQLkind for the binop. Either SqlKind.PLUS or SqlKind.MINUS.
   * @param isArg0Datetime Is arg0 the datetime argument. This is used for generating standard code.
   * @param isOtherArgInterval Is the argument an interval type or an integer that needs to convert
   *     to an interval.
   * @return The generated code that creates the BodoSQL array kernel call.
   */
  public static Expr genDatetimeArithCode(
      List<Expr> args, SqlKind binOp, boolean isArg0Datetime, boolean isOtherArgInterval) {
    assert args.size() == 2;
    final Expr arg0;
    final Expr arg1;
    // Standardize the kernel to always put the datetime argument
    // in the first slot to limit computation + simplify the kernels.
    // This is fine because + commutes.
    if (isArg0Datetime) {
      arg0 = args.get(0);
      arg1 = args.get(1);
    } else {
      arg0 = args.get(1);
      arg1 = args.get(0);
    }
    if (binOp.equals(SqlKind.MINUS)) {
      assert isArg0Datetime;
      return generateMySQLDateAddCode(arg0, arg1, isOtherArgInterval, "DATE_SUB");
    } else {
      assert binOp.equals(SqlKind.PLUS);
      return generateMySQLDateAddCode(arg0, arg1, isOtherArgInterval, "DATE_ADD");
    }
  }

  /**
   * Generate code for multiplying an interval by an integer.
   *
   * @param args List of length 2 with the generated code for the arguments.
   * @param isArg0Interval Is arg0 the interval argument. This is used for generating standard code.
   * @return The generated code that creates the BodoSQL array kernel call.
   */
  public static Expr genIntervalMultiplyCode(List<Expr> args, boolean isArg0Interval) {
    assert args.size() == 2;
    final Expr arg0;
    Expr arg1;
    // Standardize the kernel to always put the interval argument
    // in the first slot to limit computation + simplify the kernels.
    // This is fine because * commutes.
    if (isArg0Interval) {
      arg0 = args.get(0);
      arg1 = args.get(1);
    } else {
      arg0 = args.get(1);
      arg1 = args.get(0);
    }
    return ExprKt.bodoSQLKernel("interval_multiply", List.of(arg0, arg1), List.of());
  }
}
