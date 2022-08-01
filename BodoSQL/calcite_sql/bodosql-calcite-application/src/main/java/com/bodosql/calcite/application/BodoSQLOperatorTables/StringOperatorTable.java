package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.fun.SqlLibraryOperators;
import org.apache.calcite.sql.type.*;
import org.apache.calcite.sql.validate.SqlNameMatcher;

public final class StringOperatorTable implements SqlOperatorTable {

  private static @Nullable StringOperatorTable instance;

  /** Returns the String operator table, creating it if necessary. */
  public static synchronized StringOperatorTable instance() {
    StringOperatorTable instance = StringOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new StringOperatorTable();
      StringOperatorTable.instance = instance;
    }
    return instance;
  }

  // TODO: Extend the Library Operator and use the builtin Libraries

  public static final SqlFunction CONCAT =
      new SqlFunction(
          "CONCAT",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DYADIC_STRING_SUM_PRECISION_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts a collection
          OperandTypes.repeat(SqlOperandCountRanges.from(2), OperandTypes.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction CONCAT_WS =
      new SqlFunction(
          "CONCAT_WS",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DYADIC_STRING_SUM_PRECISION_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts a collection
          OperandTypes.repeat(SqlOperandCountRanges.from(2), OperandTypes.STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction MID =
      new SqlFunction(
          "MID",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.ARG0_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. CHAR_INT_INT
          OperandTypes.family(
              SqlTypeFamily.CHARACTER, SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction SUBSTR =
      new SqlFunction(
          "SUBSTR",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.ARG0_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. CHAR_INT_INT
          OperandTypes.family(
              SqlTypeFamily.CHARACTER, SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction INSTR =
      new SqlFunction(
          "INSTR",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          // currently, MSQL returns 0 on failure, so I'm doing the same
          ReturnTypes.BIGINT,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.STRING_STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction LEFT =
      new SqlFunction(
          "LEFT",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.ARG0_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.STRING_INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction RIGHT =
      new SqlFunction(
          "RIGHT",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.ARG0_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.STRING_INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction REPEAT =
      new SqlFunction(
          "REPEAT",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.ARG0_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.STRING_INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction STRCMP =
      new SqlFunction(
          "STRCMP",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.STRING_STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction FORMAT =
      new SqlFunction(
          "FORMAT",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.sequence(
              "FORMAT(NUMERIC, INTEGER)", OperandTypes.NUMERIC, OperandTypes.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction UCASE =
      new SqlFunction(
          "UCASE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.ARG0,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction LCASE =
      new SqlFunction(
          "LCASE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.ARG0,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction REVERSE =
      new SqlFunction(
          "REVERSE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.ARG0,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction LENGTH =
      new SqlFunction(
          "LENGTH",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction ORD =
      new SqlFunction(
          "ORD",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction CHAR =
      new SqlFunction(
          "CHAR",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction CHR =
      new SqlFunction(
          "CHR",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction SPACE =
      new SqlFunction(
          "SPACE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction LTRIM = SqlLibraryOperators.LTRIM;
  public static final SqlFunction RTRIM = SqlLibraryOperators.RTRIM;

  public static final SqlFunction SUBSTRING_INDEX =
      new SqlFunction(
          "SUBSTRING_INDEX",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DYADIC_STRING_SUM_PRECISION_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts a collection
          OperandTypes.STRING_STRING_INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  private List<SqlOperator> functionList =
      Arrays.asList(
          CONCAT,
          CONCAT_WS,
          MID,
          SUBSTR,
          INSTR,
          LEFT,
          RIGHT,
          REPEAT,
          STRCMP,
          FORMAT,
          UCASE,
          LCASE,
          REVERSE,
          ORD,
          CHR,
          CHAR,
          SPACE,
          SUBSTRING_INDEX,
          LTRIM,
          RTRIM,
          LENGTH);

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
