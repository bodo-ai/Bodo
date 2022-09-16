package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.fun.SqlBasicAggFunction;
import org.apache.calcite.sql.fun.SqlLibraryOperators;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlOperandTypeInference;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.apache.calcite.util.Optionality;

public final class NumericOperatorTable implements SqlOperatorTable {

  private static @Nullable NumericOperatorTable instance;

  /** Returns the Datetime operator table, creating it if necessary. */
  public static synchronized NumericOperatorTable instance() {
    NumericOperatorTable instance = NumericOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new NumericOperatorTable();
      NumericOperatorTable.instance = instance;
    }
    return instance;
  }

  // TODO: Extend the Library Operator and use the builtin Libraries

  public static final SqlFunction BITAND =
      new SqlFunction(
          "BITAND",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction BITOR =
      new SqlFunction(
          "BITOR",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction BITXOR =
      new SqlFunction(
          "BITXOR",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction BITNOT =
      new SqlFunction(
          "BITNOT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.INTEGER,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction BITSHIFTLEFT =
      new SqlFunction(
          "BITSHIFTLEFT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction BITSHIFTRIGHT =
      new SqlFunction(
          "BITSHIFTRIGHT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction GETBIT =
      new SqlFunction(
          "GETBIT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction CEILING =
      new SqlFunction(
          "CEILING",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts one numeric argument
          OperandTypes.NUMERIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction DIV0 =
      new SqlFunction(
          "DIV0",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts two numerics
          // arguments
          OperandTypes.NUMERIC_NUMERIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction HAVERSINE =
      new SqlFunction(
          "HAVERSINE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts four numeric arguments
          OperandTypes.family(
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction LOG =
      new SqlFunction(
          "LOG",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          // This function one or two numeric argument
          OperandTypes.or(OperandTypes.NUMERIC, OperandTypes.NUMERIC_NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction LOG2 =
      new SqlFunction(
          "LOG2",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts one numeric argument
          OperandTypes.NUMERIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction POW =
      new SqlFunction(
          "POW",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          (SqlOperandTypeInference) null,
          OperandTypes.NUMERIC_NUMERIC,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction TRUNC =
      new SqlFunction(
          "TRUNC",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          (SqlOperandTypeInference) null,
          OperandTypes.NUMERIC_INTEGER,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction CONV =
      new SqlFunction(
          "CONV",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts one string,
          // and two numeric arguments
          OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction WIDTH_BUCKET =
      new SqlFunction(
          "WIDTH_BUCKET",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts three numeric
          // arguments
          OperandTypes.family(
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction ACOSH =
      new SqlFunction(
          "ACOSH",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          null,
          OperandTypes.NUMERIC,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction ASINH =
      new SqlFunction(
          "ASINH",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          null,
          OperandTypes.NUMERIC,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction ATANH =
      new SqlFunction(
          "ATANH",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          null,
          OperandTypes.NUMERIC,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction FACTORIAL =
      new SqlFunction(
          "FACTORIAL",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BIGINT_NULLABLE,
          null,
          OperandTypes.INTEGER,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction SQUARE =
      new SqlFunction(
          "SQUARE",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          null,
          OperandTypes.NUMERIC,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction COSH = SqlLibraryOperators.COSH;
  public static final SqlFunction SINH = SqlLibraryOperators.SINH;
  public static final SqlFunction TANH = SqlLibraryOperators.TANH;

  public static final SqlFunction GREATEST = SqlLibraryOperators.GREATEST;
  public static final SqlFunction LEAST = SqlLibraryOperators.LEAST;

  public static final SqlBasicAggFunction VARIANCE_POP =
      SqlBasicAggFunction.create(
          "VARIANCE_POP", SqlKind.OTHER_FUNCTION, ReturnTypes.INTEGER, OperandTypes.NUMERIC);

  public static final SqlBasicAggFunction VARIANCE_SAMP =
      SqlBasicAggFunction.create(
          "VARIANCE_SAMP", SqlKind.OTHER_FUNCTION, ReturnTypes.INTEGER, OperandTypes.NUMERIC);

  public static final SqlAggFunction MEDIAN =
      SqlBasicAggFunction.create(
              "MEDIAN", SqlKind.MEDIAN, ReturnTypes.ARG0_NULLABLE_IF_EMPTY, OperandTypes.NUMERIC)
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  private List<SqlOperator> functionList =
      Arrays.asList(
          ACOSH,
          ASINH,
          ATANH,
          COSH,
          SINH,
          TANH,
          BITAND,
          BITOR,
          BITXOR,
          BITNOT,
          BITSHIFTLEFT,
          BITSHIFTRIGHT,
          GETBIT,
          CEILING,
          DIV0,
          HAVERSINE,
          LOG,
          LOG2,
          MEDIAN,
          POW,
          CONV,
          FACTORIAL,
          SQUARE,
          TRUNC,
          WIDTH_BUCKET,
          GREATEST,
          LEAST,
          VARIANCE_POP,
          VARIANCE_SAMP);

  @Override
  public void lookupOperatorOverloads(
      SqlIdentifier opName,
      @Nullable SqlFunctionCategory category,
      SqlSyntax syntax,
      List<SqlOperator> operatorList,
      SqlNameMatcher nameMatcher) {
    // Heavily copied from Calcite:
    // https://github.com/apache/calcite/blob/4bc916619fd286b2c0cc4d5c653c96a68801d74e/core/src/main/java/org/apache/calcite/sql/util/ListSqlOperatorTable.java#L57
    for (SqlOperator operator : functionList) {
      // All DateTime Operators are functions so far.
      SqlFunction func = (SqlFunction) operator;
      if (syntax != func.getSyntax()) {
        continue;
      }
      // Check that the name matches the desired names.
      if (!opName.isSimple() || !nameMatcher.matches(func.getName(), opName.getSimple())) {
        continue;
      }
      // TODO: Check the category. The Lexing currently thinks
      //  all of these functions are user defined functions.
      operatorList.add(func);
    }
  }

  @Override
  public List<SqlOperator> getOperatorList() {
    return functionList;
  }
}
