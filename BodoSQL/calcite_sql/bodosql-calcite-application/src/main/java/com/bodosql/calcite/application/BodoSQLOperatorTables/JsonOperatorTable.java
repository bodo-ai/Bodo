package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlSingleOperandTypeChecker;
import org.apache.calcite.sql.validate.SqlNameMatcher;

public final class JsonOperatorTable implements SqlOperatorTable {

  private static @Nullable JsonOperatorTable instance;

  /** Returns the JSON operator table, creating it if necessary. */
  public static synchronized JsonOperatorTable instance() {
    JsonOperatorTable instance = JsonOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new JsonOperatorTable();
      JsonOperatorTable.instance = instance;
    }
    return instance;
  }

  public static final SqlSingleOperandTypeChecker SEMI_STRUCTURED =
      SemiStructuredOperandChecker.INSTANCE;

  public static final SqlFunction GET_PATH =
      new SqlFunction(
          "GET_PATH",
          SqlKind.OTHER_FUNCTION,
          BodoReturnTypes.VARIANT_NULLABLE,
          null,
          OperandTypes.sequence(
              "GET_PATH(SEMI_STRUCTURED, STRING_LITERAL)", SEMI_STRUCTURED, OperandTypes.CHARACTER),
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction JSON_EXTRACT_PATH_TEXT =
      new SqlFunction(
          "JSON_EXTRACT_PATH_TEXT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.VARCHAR_2000_NULLABLE,
          null,
          OperandTypes.STRING_STRING,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  private List<SqlOperator> functionList = Arrays.asList(GET_PATH, JSON_EXTRACT_PATH_TEXT);

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
      // All JSON Operators added are functions so far.

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
