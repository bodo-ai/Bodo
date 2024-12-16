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
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.checkerframework.checker.nullness.qual.Nullable;

public final class ContextOperatorTable implements SqlOperatorTable {

  private static @Nullable ContextOperatorTable instance;

  /** Returns the Context operator table, creating it if necessary. */
  public static synchronized ContextOperatorTable instance() {
    ContextOperatorTable instance = ContextOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new ContextOperatorTable();
      ContextOperatorTable.instance = instance;
    }
    return instance;
  }

  public static final SqlFunction CURRENT_DATABASE =
      SqlBasicFunction.create(
          "CURRENT_DATABASE",
          // What Value should the return type be
          // Snowflake returns VARCHAR(16K)
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.SYSTEM);

  public static final SqlBasicFunction CURRENT_ACCOUNT =
      SqlBasicFunction.create(
          "CURRENT_ACCOUNT",
          // What Value should the return type be
          // Snowflake returns VARCHAR(16K)
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.SYSTEM);
  public static final SqlFunction CURRENT_ACCOUNT_NAME =
      CURRENT_ACCOUNT.withName("CURRENT_ACCOUNT_NAME");

  private List<SqlOperator> functionList =
      Arrays.asList(CURRENT_DATABASE, CURRENT_ACCOUNT, CURRENT_ACCOUNT_NAME);

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
      // All Context Operators added are functions so far.

      if (syntax != operator.getSyntax()) {
        continue;
      }
      // Check that the name matches the desired names.
      if (!opName.isSimple() || !nameMatcher.matches(operator.getName(), opName.getSimple())) {
        continue;
      }
      // TODO: Check the category. The Lexing currently thinks
      //  all of these functions are user defined functions.
      operatorList.add(operator);
    }
  }

  @Override
  public List<SqlOperator> getOperatorList() {
    return functionList;
  }
}
