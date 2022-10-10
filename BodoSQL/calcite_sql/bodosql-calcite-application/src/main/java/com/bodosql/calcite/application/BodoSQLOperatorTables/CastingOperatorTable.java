package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.validate.SqlNameMatcher;

/**
 * Operator table which contains function definitions for functions usable in BodoSQL. This operator
 * table contains definition for functions which handle converting between different SQL types.
 */
public class CastingOperatorTable implements SqlOperatorTable {

  private static @Nullable CastingOperatorTable instance;

  /** Returns the operator table, creating it if necessary. */
  public static synchronized CastingOperatorTable instance() {
    CastingOperatorTable instance = CastingOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new CastingOperatorTable();
      CastingOperatorTable.instance = instance;
    }
    return instance;
  }

  // TODO: make all of these actually coerce to cast, so that Calcite can properly typecheck it at
  // compile
  // time as of right now, we do this check in our code.
  public static final SqlFunction TO_BOOLEAN =
      new SqlFunction(
          "TO_BOOLEAN",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BOOLEAN_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // For conversion to boolean, snowflake allows a string or numeric expr.
          OperandTypes.or(OperandTypes.STRING, OperandTypes.NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction TRY_TO_BOOLEAN =
      new SqlFunction(
          "TRY_TO_BOOLEAN",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BOOLEAN_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // For conversion to boolean, snowflake allows a string or numeric expr.
          OperandTypes.or(OperandTypes.STRING, OperandTypes.NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction TO_DATE =
      new SqlFunction(
          "TO_DATE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DATE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // For conversion to date, snowflake allows a single string, datetime expr, or integer. If
          // the
          // first argument is string, an optional format string is allowed as a second argument.
          OperandTypes.or(
              OperandTypes.or(
                  OperandTypes.STRING,
                  OperandTypes.DATETIME,
                  OperandTypes.DATE,
                  OperandTypes.TIMESTAMP,
                  OperandTypes.INTEGER),
              OperandTypes.STRING_STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TRY_TO_DATE =
      new SqlFunction(
          "TRY_TO_DATE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DATE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // For conversion to date, snowflake allows a single string, datetime/timestamp expr, or
          // integer.
          // If the
          // first argument is string, an optional format string is allowed as a second argument.
          OperandTypes.or(
              OperandTypes.or(
                  OperandTypes.STRING,
                  OperandTypes.DATETIME,
                  OperandTypes.DATE,
                  OperandTypes.TIMESTAMP,
                  OperandTypes.INTEGER),
              OperandTypes.STRING_STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  private List<SqlOperator> functionList =
      Arrays.asList(TO_BOOLEAN, TRY_TO_BOOLEAN, TO_DATE, TRY_TO_DATE);

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
      // All String Operators added are functions so far.
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
