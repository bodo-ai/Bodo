package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.validate.SqlNameMatcher;

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
      new SqlFunction(
          "TO_DAYS",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.TIMESTAMP,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TO_SECONDS =
      new SqlFunction(
          "TO_SECONDS",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.TIMESTAMP,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction FROM_DAYS =
      new SqlFunction(
          "FROM_DAYS",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction UNIX_TIMESTAMP =
      new SqlFunction(
          "UNIX_TIMESTAMP",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction FROM_UNIXTIME =
      new SqlFunction(
          "FROM_UNIXTIME",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
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
