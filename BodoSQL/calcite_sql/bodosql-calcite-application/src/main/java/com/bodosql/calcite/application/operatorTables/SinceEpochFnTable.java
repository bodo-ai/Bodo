package com.bodosql.calcite.application.operatorTables;

import java.util.Arrays;
import java.util.List;
import org.apache.calcite.sql.SqlBasicFunction;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.checkerframework.checker.nullness.qual.Nullable;

public final class SinceEpochFnTable implements SqlOperatorTable {

  private static @Nullable SinceEpochFnTable instance;

  /** Returns the Datetime operator table, creating it if necessary. */
  public static synchronized SinceEpochFnTable instance() {
    SinceEpochFnTable instance = SinceEpochFnTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new SinceEpochFnTable();
      SinceEpochFnTable.instance = instance;
    }
    return instance;
  }

  public static final SqlFunction TO_DAYS =
      SqlBasicFunction.create(
          "TO_DAYS",
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What Input Types does the function accept?
          // Note: When have proper Date type support this should be restricted to just date.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TO_SECONDS =
      SqlBasicFunction.create(
          "TO_SECONDS",
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction FROM_DAYS =
      SqlBasicFunction.create(
          "FROM_DAYS",
          // What Value should the return type be
          ReturnTypes.DATE_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction UNIX_TIMESTAMP =
      SqlBasicFunction.create(
          "UNIX_TIMESTAMP",
          // What Value should the return type be
          ReturnTypes.BIGINT,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction FROM_UNIXTIME =
      SqlBasicFunction.create(
          "FROM_UNIXTIME",
          // What Value should the return type be
          ReturnTypes.TIMESTAMP_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  private List<SqlOperator> functionList =
      Arrays.asList(TO_DAYS, TO_SECONDS, FROM_DAYS, UNIX_TIMESTAMP, FROM_UNIXTIME);

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
