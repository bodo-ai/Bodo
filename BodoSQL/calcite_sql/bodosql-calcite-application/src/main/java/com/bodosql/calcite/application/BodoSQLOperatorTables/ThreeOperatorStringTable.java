package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.validate.SqlNameMatcher;

// Operator table for three operator string functions
// This table will likely be merged into a single table, but is being kept separate for now
// to avoid git conflicts

public final class ThreeOperatorStringTable implements SqlOperatorTable {

  private static @Nullable ThreeOperatorStringTable instance;

  /** Returns the Datetime operator table, creating it if necessary. */
  public static synchronized ThreeOperatorStringTable instance() {
    ThreeOperatorStringTable instance = ThreeOperatorStringTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new ThreeOperatorStringTable();
      ThreeOperatorStringTable.instance = instance;
    }
    return instance;
  }

  // TODO: Extend the Library Operator and use the builtin Libraries
  public static final SqlFunction LPAD =
      new SqlFunction(
          "LPAD",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts only
          // (Datetime, Interval)
          OperandTypes.sequence(
              "LPAD(STRING, INTEGER, STRING)",
              OperandTypes.STRING,
              OperandTypes.INTEGER,
              OperandTypes.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction RPAD =
      new SqlFunction(
          "RPAD",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts only
          // (Datetime, Interval)
          OperandTypes.sequence(
              "RPAD(STRING, INTEGER, STRING)",
              OperandTypes.STRING,
              OperandTypes.INTEGER,
              OperandTypes.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction REPLACE =
      new SqlFunction(
          "REPLACE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts only
          // (Datetime, Interval)
          OperandTypes.sequence(
              "REPLACE(STRING, FIND_STRING, REPLACE_STRING)",
              OperandTypes.STRING,
              OperandTypes.STRING,
              OperandTypes.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  private List<SqlOperator> functionList = Arrays.asList(LPAD, RPAD, REPLACE);

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
