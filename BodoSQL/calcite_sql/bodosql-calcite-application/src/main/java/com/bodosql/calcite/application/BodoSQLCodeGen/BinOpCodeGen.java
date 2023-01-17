package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.DateAddCodeGen.generateMySQLDateAddCode;
import static com.bodosql.calcite.application.BodoSQLExprType.meet_elementwise_op;
import static com.bodosql.calcite.application.Utils.Utils.generateNullCheck;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLExprType;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
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
   * @param exprTypes The exprType of each argument.
   * @param binOp The Binary operator to apply to each pair of arguments.
   * @param isScalar Is this function used inside an apply and should always generate scalar code.
   * @param argDataTypes List of SQL data types for the input to the binary operation.
   * @return The code generated for the BinOp call.
   */
  public static String generateBinOpCode(
      List<String> args,
      List<BodoSQLExprType.ExprType> exprTypes,
      SqlOperator binOp,
      boolean isScalar,
      List<RelDataType> argDataTypes) {
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
    // add pd.Series() around column arguments since binary operators assume Series input
    // (arrays don't have full support)
    List<String> update_args = new ArrayList<>();
    for (int i = 0; i < args.size(); i++) {
      String new_arg = args.get(i);
      if (!isScalar && exprTypes.get(i) == BodoSQLExprType.ExprType.COLUMN) {
        new_arg = String.format("pd.Series(%s)", new_arg);
      }
      update_args.add(new_arg);
    }
    return generateBinOpCodeHelper(update_args, exprTypes, binOpKind, isScalar);
  }
  /**
   * Helper function that returns the necessary generated code for a BinOp Call. This function may
   * have more than two arguments because repeated use of the same operator are grouped into the
   * same Node.
   *
   * @param args The arguments to the binop.
   * @param exprTypes The exprType of each argument.
   * @param binOpKind The SqlKind of the binary operator
   * @param isScalar Is this function used inside an apply and should always generate scalar code.
   * @return The code generated for the BinOp call.
   */
  public static String generateBinOpCodeHelper(
      List<String> args,
      List<BodoSQLExprType.ExprType> exprTypes,
      SqlKind binOpKind,
      boolean isScalar) {
    String columnOperator;
    String scalarOperator;
    // Is the column operator generated as a function call
    boolean columnIsFunction = false;
    // Is the scalar operator generated as a function call
    boolean scalarIsFunction = false;
    switch (binOpKind) {
      case EQUALS:
        columnOperator = "==";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_equal";
        break;
      case NULL_EQUALS:
        columnIsFunction = true;
        columnOperator = "bodosql.libs.sql_operators.sql_null_equal_column";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.null_handling.sql_null_equal_scalar";
        break;
      case NOT_EQUALS:
        columnOperator = "!=";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_not_equal";
        break;
      case LESS_THAN:
        columnOperator = "<";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_less_than";
        break;
      case GREATER_THAN:
        columnOperator = ">";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_greater_than";
        break;
      case LESS_THAN_OR_EQUAL:
        columnOperator = "<=";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal";
        break;
      case GREATER_THAN_OR_EQUAL:
        columnOperator = ">=";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal";
        break;
      case PLUS:
        columnIsFunction = true;
        columnOperator = "bodo.libs.bodosql_array_kernels.add_numeric";
        scalarIsFunction = true;
        scalarOperator = "bodo.libs.bodosql_array_kernels.add_numeric";
        break;
      case MINUS:
        columnIsFunction = true;
        columnOperator = "bodo.libs.bodosql_array_kernels.subtract_numeric";
        scalarIsFunction = true;
        scalarOperator = "bodo.libs.bodosql_array_kernels.subtract_numeric";
        break;
      case TIMES:
        columnIsFunction = true;
        columnOperator = "bodo.libs.bodosql_array_kernels.multiply_numeric";
        scalarIsFunction = true;
        scalarOperator = "bodo.libs.bodosql_array_kernels.multiply_numeric";
        break;
      case DIVIDE:
        columnIsFunction = true;
        columnOperator = "bodo.libs.bodosql_array_kernels.divide_numeric";
        scalarIsFunction = true;
        scalarOperator = "bodo.libs.bodosql_array_kernels.divide_numeric";
        break;
      case AND:
        columnIsFunction = true;
        columnOperator = "bodo.libs.bodosql_array_kernels.booland";
        scalarIsFunction = true;
        scalarOperator = "bodo.libs.bodosql_array_kernels.booland";
        break;
      default:
        throw new BodoSQLCodegenException(
            "Unsupported Operator, " + binOpKind + " specified in query.");
    }
    StringBuilder newOp = new StringBuilder();
    StringBuilder functionStack = new StringBuilder();
    newOp.append("(");
    BodoSQLExprType.ExprType exprType = exprTypes.get(0);
    String operator;
    boolean wasFunction;
    boolean isFunction = false;
    /**
     * Iterate through the arguments to generate the necessary code. For operators that map to
     * operators in Pandas, generate ARG0 OP ARG1. For operators that map to functions generate
     * OP(ARG0, ARG1)
     *
     * <p>To keep inserting function calls to front, we use two StringBuilders so we can keep
     * finding the starting position.
     */
    for (int i = 0; i < args.size() - 1; i++) {
      exprType = meet_elementwise_op(exprType, exprTypes.get(i + 1));
      wasFunction = isFunction;
      if (isScalar || exprType == BodoSQLExprType.ExprType.SCALAR) {
        operator = scalarOperator;
        isFunction = scalarIsFunction;
      } else {
        operator = columnOperator;
        isFunction = columnIsFunction;
      }
      if (isFunction) {
        functionStack.insert(0, operator + "(");
        functionStack.append(args.get(i));
        if (wasFunction) {
          functionStack.append(")");
        }
        functionStack.append(", ");
      } else {
        if (wasFunction) {
          newOp.append(functionStack).append(args.get(i)).append(")");
          // Clear the function stack
          functionStack = new StringBuilder();
        } else {
          newOp.append(args.get(i));
        }
        newOp.append(" ").append(operator).append(" ");
      }
    }
    newOp.append(functionStack);
    newOp.append(args.get(args.size() - 1)).append(")");
    if (isFunction) {
      newOp.append(")");
    }
    // make sure the output is array type for column cases
    // (output is Series here since input is converted to Series in generateBinOpCode)
    // The exception is AND and numerics always output an array
    if (!isScalar
        && exprType == BodoSQLExprType.ExprType.COLUMN
        && (binOpKind != SqlKind.AND
            && binOpKind != SqlKind.PLUS
            && binOpKind != SqlKind.MINUS
            && binOpKind != SqlKind.TIMES
            && binOpKind != SqlKind.DIVIDE)) {
      newOp.insert(0, "bodo.hiframes.pd_series_ext.get_series_data(");
      newOp.append(")");
    }
    return newOp.toString();
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
   * Function that generates code for a single argument in OR.
   *
   * @param argCode The code generate for that single argument
   * @param appendOperator Should the operator be added to generated string
   * @param inputVar The name of the input table which columns reference
   * @param nullSet Set of columns that may need to be treated as null. If this argument is treated
   *     as column then the set will be empty.
   * @param isScalar Is this function used inside an apply and should always generate scalar code.
   * @param isSingleRow Boolean for if table references refer to a single row or the whole table.
   *     Operations that operate per row (i.e. Case switch this to True). This is used for
   *     determining if an expr returns a scalar or a column.
   * @return The code generated for that single argument.
   */
  public static String generateOrCode(
      String argCode,
      boolean appendOperator,
      String inputVar,
      List<String> colNames,
      HashSet<String> nullSet,
      boolean isScalar,
      boolean isSingleRow) {
    StringBuilder newArg = new StringBuilder();
    // This generates code as (False if (pd.isna(col0) or ... pd.isna(coln)) else argCode)
    // There may be ways to refactor this to generate more efficient code.
    newArg.append("(");
    newArg.append(generateNullCheck(inputVar, colNames, nullSet, "False", argCode, isSingleRow));
    newArg.append(")");
    if (appendOperator) {
      if (isScalar) {
        newArg.append(" or ");
      } else {
        newArg.append(" | ");
      }
    }
    return newArg.toString();
  }

  /**
   * Generate code for Adding or Subtracting an two intervals.
   *
   * @param args List of length 2 with the generated code for the arguments.
   * @param binOp The SQLkind for the binop. Either SqlKind.PLUS or SqlKind.MINUS.
   * @return The generated code that creates the BodoSQL array kernel call.
   */
  public static String genIntervalAddCode(List<String> args, SqlKind binOp) {
    assert args.size() == 2;
    final String arg0 = args.get(0);
    String arg1 = args.get(1);
    if (binOp.equals(SqlKind.MINUS)) {
      // Negate the input for Minus
      arg1 = String.format("bodo.libs.bodosql_array_kernels.negate(%s)", arg1);
    } else {
      assert binOp.equals(SqlKind.PLUS);
    }
    return String.format(
        "bodo.libs.bodosql_array_kernels.interval_add_interval(%s, %s)", arg0, arg1);
  }

  /**
   * Generate code for Adding or Subtracting an interval from tz_aware data.
   *
   * @param args List of length 2 with the generated code for the arguments.
   * @param binOp The SQLkind for the binop. Either SqlKind.PLUS or SqlKind.MINUS.
   * @param isArg0TZAware Is arg0 the tz aware argument. This is used for generating standard code.
   * @return The generated code that creates the BodoSQL array kernel call.
   */
  public static String genTZAwareIntervalArithCode(
      List<String> args, SqlKind binOp, boolean isArg0TZAware) {
    assert args.size() == 2;
    final String arg0;
    String arg1;
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
      arg1 = String.format("bodo.libs.bodosql_array_kernels.negate(%s)", arg1);
    } else {
      assert binOp.equals(SqlKind.PLUS);
    }
    return String.format(
        "bodo.libs.bodosql_array_kernels.tz_aware_interval_add(%s, %s)", arg0, arg1);
  }

  /**
   * Generate code for subtracting two dates.
   *
   * @param args List of length 2 with the generated code for the arguments.
   * @return The generated code that creates the BodoSQL array kernel call.
   */
  public static String genDateSubCode(List<String> args) {
    assert args.size() == 2;
    final String arg0 = args.get(0);
    final String arg1 = args.get(1);
    return String.format("bodo.libs.bodosql_array_kernels.date_sub_date(%s, %s)", arg0, arg1);
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
  public static String genDatetimeArithCode(
      List<String> args, SqlKind binOp, boolean isArg0Datetime, boolean isOtherArgInterval) {
    assert args.size() == 2;
    final String arg0;
    String arg1;
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
  public static String genIntervalMultiplyCode(List<String> args, boolean isArg0Interval) {
    assert args.size() == 2;
    final String arg0;
    String arg1;
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
    return String.format("bodo.libs.bodosql_array_kernels.interval_multiply(%s, %s)", arg0, arg1);
  }
}
