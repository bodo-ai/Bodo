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
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.checkerframework.checker.nullness.qual.Nullable;

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
      SqlBasicFunction.create(
          "LPAD",
          // Output precision cannot be statically determined.
          BodoReturnTypes.ARG0_NULLABLE_VARYING_UNDEFINED_PRECISION,
          // What Input Types does the function accept? This function accepts only
          // (string, binary)
          // Takes in 3 arguments.
          // The 1st is string/binary, the 2nd is an integer, and the 3rd is the same as the 1st.
          // The 3rd argument can be optionally omitted if the 1st argument is a string.
          OperandTypes.family(
                  SqlTypeFamily.CHARACTER, SqlTypeFamily.INTEGER, SqlTypeFamily.CHARACTER)
              .or(
                  OperandTypes.family(
                      SqlTypeFamily.BINARY, SqlTypeFamily.INTEGER, SqlTypeFamily.BINARY))
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.INTEGER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction RPAD =
      SqlBasicFunction.create(
          "RPAD",
          // Output precision cannot be statically determined.
          BodoReturnTypes.ARG0_NULLABLE_VARYING_UNDEFINED_PRECISION,
          // What Input Types does the function accept? This function accepts only
          // (string, binary)
          // Takes in 3 arguments.
          // The first is string/binary, the second is an integer, and the third is the same as the
          // first.
          // The 3rd argument can be optionally omitted if the first argument is a string.
          OperandTypes.family(
                  SqlTypeFamily.CHARACTER, SqlTypeFamily.INTEGER, SqlTypeFamily.CHARACTER)
              .or(
                  OperandTypes.family(
                      SqlTypeFamily.BINARY, SqlTypeFamily.INTEGER, SqlTypeFamily.BINARY))
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.INTEGER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction REPLACE =
      SqlBasicFunction.create(
          "REPLACE",
          // Output precision cannot be statically determined.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          // What Input Types does the function accept.
          // Note: Calcite already defines REPLACE with 3 arguments
          // in the SqlStdOperatorTable, so we just define when
          // the third argument is missing.
          OperandTypes.STRING_STRING,
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
