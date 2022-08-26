package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.ArrayList;
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

  /**
   * Creates an OperandTypeChecker for a function that has several arguments with consistent types,
   * but some of them are optional. Calling the function with arguments (3, A, B, C, D, E) is the
   * same as creating an OR of the families (A, B, C), (A, B, C, D), (A, B, C, D, E)
   *
   * @param min the minimum number of arguments required for the function call
   * @param families the types for each argument when they are provided
   * @return an OperandTypeChecker with the specs mentioned above
   */
  public static SqlOperandTypeChecker argumentRange(int min, SqlTypeFamily... families) {
    assert min <= families.length;
    List<SqlTypeFamily> familyList = new ArrayList<SqlTypeFamily>();
    for (int i = 0; i < min; i++) {
      familyList.add(families[i]);
    }
    SqlOperandTypeChecker rule = OperandTypes.family(familyList);
    for (int i = min; i < families.length; i++) {
      familyList.add(families[i]);
      rule = OperandTypes.or(rule, OperandTypes.family(familyList));
    }
    return rule;
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
          OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
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
          OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
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

  public static final SqlFunction EDITDISTANCE =
      new SqlFunction(
          "EDITDISTANCE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.or(OperandTypes.STRING_STRING, OperandTypes.STRING_STRING_INTEGER),
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

  public static final SqlFunction LEN =
      new SqlFunction(
          "LEN",
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

  public static final SqlFunction SPLIT_PART =
      new SqlFunction(
          "SPLIT_PART",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.VARCHAR_2000_NULLABLE,
          null,
          OperandTypes.STRING_STRING_INTEGER,
          SqlFunctionCategory.STRING);

  public static final SqlFunction STRTOK =
      new SqlFunction(
          "STRTOK",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.VARCHAR_2000_NULLABLE,
          null,
          OperandTypes.or(
              OperandTypes.STRING, OperandTypes.STRING_STRING, OperandTypes.STRING_STRING_INTEGER),
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

  public static final SqlFunction REGEXP_LIKE =
      new SqlFunction(
          "REGEXP_LIKE",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BOOLEAN_NULLABLE,
          null,
          argumentRange(
              2, SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction REGEXP_COUNT =
      new SqlFunction(
          "REGEXP_COUNT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.INTEGER_NULLABLE,
          null,
          argumentRange(
              2,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.CHARACTER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction REGEXP_REPLACE =
      new SqlFunction(
          "REGEXP_REPLACE",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.VARCHAR_2000_NULLABLE,
          null,
          argumentRange(
              2,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.CHARACTER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction REGEXP_SUBSTR =
      new SqlFunction(
          "REGEXP_SUBSTR",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.VARCHAR_2000_NULLABLE,
          null,
          argumentRange(
              2,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.INTEGER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction REGEXP_INSTR =
      new SqlFunction(
          "REGEXP_INSTR",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.INTEGER_NULLABLE,
          null,
          argumentRange(
              2,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.INTEGER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction INITCAP =
      new SqlFunction(
          "INITCAP",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.VARCHAR_2000_NULLABLE,
          null,
          OperandTypes.or(OperandTypes.STRING, OperandTypes.STRING_STRING),
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
          REGEXP_LIKE,
          REGEXP_COUNT,
          REGEXP_REPLACE,
          REGEXP_SUBSTR,
          REGEXP_INSTR,
          STRCMP,
          EDITDISTANCE,
          FORMAT,
          UCASE,
          LCASE,
          REVERSE,
          ORD,
          CHR,
          CHAR,
          SPACE,
          SPLIT_PART,
          STRTOK,
          SUBSTRING_INDEX,
          SqlLibraryOperators.TRANSLATE3,
          INITCAP,
          LTRIM,
          RTRIM,
          LEN,
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
