package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.fun.SqlLibraryOperators;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlOperandTypeInference;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.validate.SqlNameMatcher;

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

  public static final SqlFunction LOG10 =
      new SqlFunction(
          "LOG10",
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

  public static final SqlFunction GREATEST = SqlLibraryOperators.GREATEST;
  public static final SqlFunction LEAST = SqlLibraryOperators.LEAST;

  private List<SqlOperator> functionList =
      Arrays.asList(CEILING, DIV0, LOG, LOG2, POW, CONV, GREATEST, LEAST);

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
