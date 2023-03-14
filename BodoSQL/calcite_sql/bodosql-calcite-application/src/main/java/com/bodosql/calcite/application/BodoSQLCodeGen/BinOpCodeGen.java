package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.DateAddCodeGen.generateMySQLDateAddCode;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import java.util.*;
import org.apache.calcite.avatica.*;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.type.*;

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
   * @return The code generated for the BinOp call.
   */
  public static Expr generateBinOpCode(
      List<Expr> args, SqlOperator binOp, List<RelDataType> argDataTypes) {
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
          arg0TypeName.equals(SqlTypeName.TIMESTAMP) || arg0TypeName.equals(SqlTypeName.DATE);
      boolean isArg1Datetime =
          arg1TypeName.equals(SqlTypeName.TIMESTAMP) || arg1TypeName.equals(SqlTypeName.DATE);

      boolean isArg0Interval = argDataTypes.get(0) instanceof IntervalSqlType;
      boolean isArg1Interval = argDataTypes.get(1) instanceof IntervalSqlType;
      if (isArg0Interval && isArg1Interval) {
        assert binOpKind.equals(SqlKind.PLUS) || binOpKind.equals(SqlKind.MINUS);
        return genIntervalAddCode(args, binOpKind);
      } else if ((isArg0TZAware && isArg1Interval) || (isArg1TZAware && isArg0Interval)) {
        assert binOpKind.equals(SqlKind.PLUS) || binOpKind.equals(SqlKind.MINUS);
        return genTZAwareIntervalArithCode(args, binOpKind, isArg0TZAware);
      } else if (isArg0Datetime && isArg1Datetime) {
        assert binOpKind.equals(SqlKind.MINUS);
        return genDateSubCode(args);
      } else if (isArg0Datetime || isArg1Datetime) {
        assert binOpKind.equals(SqlKind.PLUS) || binOpKind.equals(SqlKind.MINUS);
        return genDatetimeArithCode(
            args, binOpKind, isArg0Datetime, isArg0Interval || isArg1Interval);
      } else if ((isArg0Interval || isArg1Interval) && binOpKind.equals(SqlKind.TIMES)) {
        return genIntervalMultiplyCode(args, isArg0Interval);
      }
    }
    return generateBinOpCodeHelper(args, binOpKind);
  }
  /**
   * Helper function that returns the necessary generated code for a BinOp Call. This function may
   * have more than two arguments because repeated use of the same operator are grouped into the
   * same Node.
   *
   * @param args The arguments to the binop.
   * @param binOpKind The SqlKind of the binary operator
   * @return The code generated for the BinOp call.
   */
  public static Expr generateBinOpCodeHelper(List<Expr> args, SqlKind binOpKind) {
    final String fn;
    switch (binOpKind) {
      case EQUALS:
        fn = "bodo.libs.bodosql_array_kernels.equal";
        break;
      case NULL_EQUALS:
        fn = "bodo.libs.bodosql_array_kernels.equal_null";
        break;
      case NOT_EQUALS:
        fn = "bodo.libs.bodosql_array_kernels.not_equal";
        break;
      case LESS_THAN:
        fn = "bodo.libs.bodosql_array_kernels.less_than";
        break;
      case GREATER_THAN:
        fn = "bodo.libs.bodosql_array_kernels.greater_than";
        break;
      case LESS_THAN_OR_EQUAL:
        fn = "bodo.libs.bodosql_array_kernels.less_than_or_equal";
        break;
      case GREATER_THAN_OR_EQUAL:
        fn = "bodo.libs.bodosql_array_kernels.greater_than_or_equal";
        break;
      case PLUS:
        fn = "bodo.libs.bodosql_array_kernels.add_numeric";
        break;
      case MINUS:
        fn = "bodo.libs.bodosql_array_kernels.subtract_numeric";
        break;
      case TIMES:
        fn = "bodo.libs.bodosql_array_kernels.multiply_numeric";
        break;
      case DIVIDE:
        fn = "bodo.libs.bodosql_array_kernels.divide_numeric";
        break;
      case AND:
        fn = "bodo.libs.bodosql_array_kernels.booland";
        break;
      case OR:
        fn = "bodo.libs.bodosql_array_kernels.boolor";
        break;
      default:
        throw new BodoSQLCodegenException(
            "Unsupported Operator, " + binOpKind + " specified in query.");
    }
    StringBuilder codeBuilder = new StringBuilder();
    /** Create the function calls */
    for (int i = 0; i < args.size() - 1; i++) {
      // Generate n - 1 function calls
      codeBuilder.append(fn).append("(");
    }
    for (int i = 0; i < args.size(); i++) {
      // Insert arguments and add , and closing )
      codeBuilder.append(args.get(i).emit());
      if (i != 0) {
        codeBuilder.append(")");
      }
      if (i != (args.size() - 1)) {
        codeBuilder.append(", ");
      }
    }
    return new Expr.Raw(codeBuilder.toString());
  }

  /**
   * Function that returns the generated name for a Binop Call. This function may have more than two
   * arguments because repeated use of the same operator are grouped into the same Node.
   *
   * @param names The names of the arguments to the binop
   * @param binOp The Binary operator to apply to each pair of arguments.
   * @return The name generated that matches the Binop expression.
   */
  public static String generateBinOpName(List<String> names, SqlOperator binOp) {
    StringBuilder nameBuilder = new StringBuilder();
    for (int i = 0; i < names.size(); i++) {
      nameBuilder.append(binOp.toString()).append("(");
    }
    nameBuilder.append(names.get(0));
    for (int i = 1; i < names.size(); i++) {
      nameBuilder.append(", ").append(names.get(i)).append(")");
    }
    return nameBuilder.toString();
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
      arg1 = new Expr.Call("bodo.libs.bodosql_array_kernels.negate", arg1);
    } else {
      assert binOp.equals(SqlKind.PLUS);
    }
    return new Expr.Call("bodo.libs.bodosql_array_kernels.interval_add_interval", arg0, arg1);
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
      arg1 = new Expr.Call("bodo.libs.bodosql_array_kernels.negate", arg1);
    } else {
      assert binOp.equals(SqlKind.PLUS);
    }
    return new Expr.Call("bodo.libs.bodosql_array_kernels.tz_aware_interval_add", arg0, arg1);
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
    return new Expr.Call(
        "bodo.libs.bodosql_array_kernels.date_sub_date_unit", new Expr.Raw("'DAY'"), arg1, arg0);
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
    return new Expr.Call("bodo.libs.bodosql_array_kernels.interval_multiply", arg0, arg1);
  }
}
