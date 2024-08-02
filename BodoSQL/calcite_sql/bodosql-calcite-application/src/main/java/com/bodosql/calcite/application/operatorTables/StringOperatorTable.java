package com.bodosql.calcite.application.operatorTables;

import static com.bodosql.calcite.application.operatorTables.OperatorTableUtils.argumentRange;
import static org.apache.calcite.sql.type.BodoReturnTypes.SPLIT_RETURN_TYPE;

import java.util.Arrays;
import java.util.List;
import org.apache.calcite.sql.SqlBasicFunction;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.fun.SqlLibraryOperators;
import org.apache.calcite.sql.type.BodoOperandTypes;
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlOperandCountRanges;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeTransforms;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.checkerframework.checker.nullness.qual.Nullable;

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
      SqlNullPolicyFunction.createAnyPolicy(
          "CONCAT",
          // Concat sums together all input precision.
          // DYADIC_STRING_SUM always expects at least two arguments, which means that we need
          // special logic to handle the single argument version of CONCAT.
          BodoReturnTypes.CONCAT_RETURN_TYPE,
          OperandTypes.repeat(SqlOperandCountRanges.from(1), OperandTypes.BINARY)
              .or(OperandTypes.repeat(SqlOperandCountRanges.from(1), OperandTypes.CHARACTER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction CONCAT_WS =
      SqlNullPolicyFunction.createAnyPolicy(
          "CONCAT_WS",
          // Concat sums together all input precision and
          // includes the separator where appropriate.
          // Unlike CONCAT, CONCAT_WS always takes at least two arguments - the separator and the
          // variadic list of expressions. This means it's safe to always call DYADIC_STRING_SUM
          // which expects at least 2 args.
          BodoReturnTypes.CONCAT_WS_RETURN_TYPE,
          OperandTypes.repeat(SqlOperandCountRanges.from(2), OperandTypes.BINARY)
              .or(OperandTypes.repeat(SqlOperandCountRanges.from(2), OperandTypes.CHARACTER)),
          SqlFunctionCategory.STRING);

  public static final SqlNullPolicyFunction SUBSTR =
      SqlNullPolicyFunction.createAnyPolicy(
          "SUBSTR",
          ReturnTypes.ARG0_NULLABLE_VARYING,
          OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER)
              .or(OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.INTEGER)),
          SqlFunctionCategory.STRING);

  public static final SqlFunction MID = SUBSTR.withName("MID");

  public static final SqlNullPolicyFunction INSTR =
      SqlNullPolicyFunction.createAnyPolicy(
          "INSTR",
          // What Value should the return type be
          // currently, MSQL returns 0 on failure, so I'm doing the same
          ReturnTypes.BIGINT_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.STRING_STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlNullPolicyFunction LEFT =
      SqlNullPolicyFunction.createAnyPolicy(
          "LEFT",
          // Match input precision. The substring is at most the same string.
          ReturnTypes.ARG0_NULLABLE_VARYING,
          // What Input Types does the function accept.
          OperandTypes.STRING_INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction RIGHT = LEFT.withName("RIGHT");

  public static final SqlFunction REPEAT =
      SqlNullPolicyFunction.createAnyPolicy(
          "REPEAT",
          // Repeat has an unknown static output precision.
          BodoReturnTypes.ARG0_NULLABLE_VARYING_UNKNOWN_PRECISION,
          // What Input Types does the function accept.
          OperandTypes.STRING_INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlNullPolicyFunction STRCMP =
      SqlNullPolicyFunction.createAnyPolicy(
          "STRCMP",
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.STRING_STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction EDITDISTANCE =
      SqlNullPolicyFunction.createAnyPolicy(
          "EDITDISTANCE",
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.STRING_STRING.or(OperandTypes.STRING_STRING_INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction JAROWINKLER_SIMILARITY =
      SqlNullPolicyFunction.createAnyPolicy(
          "JAROWINKLER_SIMILARITY",
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.STRING_STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction FORMAT =
      SqlNullPolicyFunction.createAnyPolicy(
          "FORMAT",
          // Return type has an unknown precision, but it should only be
          // null if there is a null input.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.sequence(
              "FORMAT(NUMERIC, INTEGER)", OperandTypes.NUMERIC, OperandTypes.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlNullPolicyFunction UCASE =
      SqlNullPolicyFunction.createAnyPolicy(
          "UCASE",
          // What Value should the return type be
          ReturnTypes.ARG0,
          // What Input Types does the function accept.
          OperandTypes.STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction LCASE = UCASE.withName("LCASE");

  public static final SqlFunction REVERSE = UCASE.withName("REVERSE");

  public static final SqlNullPolicyFunction LENGTH =
      SqlNullPolicyFunction.createAnyPolicy(
          "LENGTH", ReturnTypes.BIGINT_NULLABLE, OperandTypes.STRING, SqlFunctionCategory.STRING);

  public static final SqlFunction LEN = LENGTH.withName("LEN");

  public static final SqlFunction ORD =
      SqlNullPolicyFunction.createAnyPolicy(
          "ORD",
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.CHARACTER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlNullPolicyFunction CHAR =
      SqlNullPolicyFunction.createAnyPolicy(
          "CHAR",
          // Char outputs a single character
          BodoReturnTypes.VARCHAR_1_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction CHR = CHAR.withName("CHR");

  public static final SqlFunction RTRIMMED_LENGTH =
      SqlNullPolicyFunction.createAnyPolicy(
          "RTRIMMED_LENGTH",
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.CHARACTER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction SPACE =
      SqlNullPolicyFunction.createAnyPolicy(
          "SPACE",
          // What Value should the return type be
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlNullPolicyFunction STARTSWITH =
      SqlNullPolicyFunction.createAnyPolicy(
          "STARTSWITH",
          ReturnTypes.BOOLEAN_NULLABLE,
          OperandTypes.STRING_STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction ENDSWITH = STARTSWITH.withName("ENDSWITH");

  public static final SqlNullPolicyFunction INSERT =
      SqlNullPolicyFunction.createAnyPolicy(
          "INSERT",
          // Snowflake calculates the resulting precision
          // as 2 * arg0 + arg3. This seems like an error,
          // as the max possible precision should be arg0 + arg3,
          // but we will match Snowflake.
          BodoReturnTypes.INSERT_RETURN_TYPE,
          OperandTypes.family(
              SqlTypeFamily.STRING,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.STRING),
          SqlFunctionCategory.STRING);

  public static final SqlNullPolicyFunction CHARINDEX =
      SqlNullPolicyFunction.createAnyPolicy(
          "CHARINDEX",
          ReturnTypes.INTEGER_NULLABLE,
          argumentRange(2, SqlTypeFamily.STRING, SqlTypeFamily.STRING, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction POSITION = CHARINDEX.withName("POSITION");

  public static final SqlFunction SPLIT_PART =
      SqlNullPolicyFunction.createAnyPolicy(
          "SPLIT_PART",
          // The precision is the same as in the worst
          // case we keep the same string.
          ReturnTypes.ARG0_NULLABLE_VARYING,
          OperandTypes.STRING_STRING_INTEGER,
          SqlFunctionCategory.STRING);

  public static final SqlFunction STRTOK =
      SqlBasicFunction.create(
          "STRTOK",
          // The precision is the same as in the worst
          // case we keep the same string.
          BodoReturnTypes.ARG0_FORCE_NULLABLE_VARYING,
          OperandTypes.STRING.or(OperandTypes.STRING_STRING).or(OperandTypes.STRING_STRING_INTEGER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction STRTOK_TO_ARRAY =
      SqlNullPolicyFunction.createAnyPolicy(
          "STRTOK_TO_ARRAY",
          BodoReturnTypes.TO_NULLABLE_VARYING_ARRAY,
          OperandTypes.CHARACTER.or(BodoOperandTypes.CHARACTER_CHARACTER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction LTRIM =
      SqlNullPolicyFunction.createAnyPolicy(
          "LTRIM",
          // Precision matches in the input.
          ReturnTypes.ARG0_NULLABLE_VARYING,
          argumentRange(1, SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction RTRIM =
      SqlNullPolicyFunction.createAnyPolicy(
          "RTRIM",
          // Precision matches in the input.
          ReturnTypes.ARG0_NULLABLE_VARYING,
          argumentRange(1, SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction SUBSTRING_INDEX =
      SqlNullPolicyFunction.createAnyPolicy(
          "SUBSTRING_INDEX",
          // In the worst case we return the whole string,
          // so maintain the precision.
          ReturnTypes.ARG0_NULLABLE_VARYING,
          // What Input Types does the function accept. This function accepts a collection
          OperandTypes.STRING_STRING_INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlNullPolicyFunction REGEXP_LIKE =
      SqlNullPolicyFunction.createAnyPolicy(
          "REGEXP_LIKE",
          ReturnTypes.BOOLEAN_NULLABLE,
          argumentRange(
              2, SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER),
          SqlFunctionCategory.STRING);

  /* This RLIKE SqlFunction is to support SQL queries of the form:
   * RLIKE(<subject>, <pattern>, [<parameters>]) as opposed to the RLIKE SqlLikeOperator
   * which supports SQL queries of the form: <subject> RLIKE <pattern>
   * */
  public static final SqlFunction RLIKE = REGEXP_LIKE.withName("RLIKE");

  public static final SqlFunction REGEXP_COUNT =
      SqlNullPolicyFunction.createAnyPolicy(
          "REGEXP_COUNT",
          ReturnTypes.INTEGER_NULLABLE,
          argumentRange(
              2,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.CHARACTER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction REGEXP_REPLACE =
      SqlNullPolicyFunction.createAnyPolicy(
          "REGEXP_REPLACE",
          // Return string has an unknown precision.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
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
      SqlBasicFunction.create(
          "REGEXP_SUBSTR",
          // Match precision because this is a substring
          // returns null if substring DNE
          BodoReturnTypes.ARG0_FORCE_NULLABLE_VARYING,
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
      SqlNullPolicyFunction.createAnyPolicy(
          "REGEXP_INSTR",
          ReturnTypes.INTEGER_NULLABLE,
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
      SqlNullPolicyFunction.createAnyPolicy(
          "INITCAP",
          // This is a mappable function so the precision is the same.
          ReturnTypes.ARG0_NULLABLE,
          OperandTypes.STRING.or(OperandTypes.STRING_STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction CONTAINS = STARTSWITH.withName("CONTAINS");

  public static final SqlFunction SPLIT =
      SqlNullPolicyFunction.createAnyPolicy(
          "SPLIT",
          // Return type for split is the exact same as TO_ARRAY
          SPLIT_RETURN_TYPE,
          OperandTypes.CHARACTER_CHARACTER,
          SqlFunctionCategory.STRING);

  public static final SqlNullPolicyFunction SHA2 =
      SqlNullPolicyFunction.createAnyPolicy(
          "SHA2",
          // SHA2 outputs at most 128 characters.
          BodoReturnTypes.VARCHAR_128_NULLABLE,
          OperandTypes.STRING.or(OperandTypes.STRING_INTEGER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction SHA2_HEX = SHA2.withName("SHA2_HEX");

  public static final SqlNullPolicyFunction MD5 =
      SqlNullPolicyFunction.createAnyPolicy(
          "MD5",
          // MD5 outputs at most 32 characters.
          BodoReturnTypes.VARCHAR_32_NULLABLE,
          OperandTypes.STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction MD5_HEX = MD5.withName("MD5_HEX");

  public static final SqlFunction HEX_ENCODE =
      SqlBasicFunction.create(
          "HEX_ENCODE",
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          argumentRange(1, SqlTypeFamily.STRING, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction HEX_DECODE_STRING =
      SqlBasicFunction.create(
          "HEX_DECODE_STRING",
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          OperandTypes.STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction HEX_DECODE_BINARY =
      SqlBasicFunction.create(
          "HEX_DECODE_BINARY",
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          ReturnTypes.explicit(SqlTypeName.VARBINARY).andThen(SqlTypeTransforms.TO_NULLABLE),
          OperandTypes.STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction TRY_HEX_DECODE_STRING =
      SqlBasicFunction.create(
          "TRY_HEX_DECODE_STRING",
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_FORCE_NULLABLE,
          OperandTypes.STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction TRY_HEX_DECODE_BINARY =
      SqlBasicFunction.create(
          "TRY_HEX_DECODE_BINARY",
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARBINARY_FORCE_NULLABLE,
          OperandTypes.STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction BASE64_ENCODE =
      SqlBasicFunction.create(
          "BASE64_ENCODE",
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          argumentRange(1, SqlTypeFamily.STRING, SqlTypeFamily.INTEGER, SqlTypeFamily.STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction BASE64_DECODE_STRING =
      SqlBasicFunction.create(
          "BASE64_DECODE_STRING",
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          argumentRange(1, SqlTypeFamily.STRING, SqlTypeFamily.STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction TRY_BASE64_DECODE_STRING =
      SqlBasicFunction.create(
          "TRY_BASE64_DECODE_STRING",
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_FORCE_NULLABLE,
          argumentRange(1, SqlTypeFamily.STRING, SqlTypeFamily.STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction BASE64_DECODE_BINARY =
      SqlBasicFunction.create(
          "BASE64_DECODE_BINARY",
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          ReturnTypes.explicit(SqlTypeName.VARBINARY).andThen(SqlTypeTransforms.TO_NULLABLE),
          argumentRange(1, SqlTypeFamily.STRING, SqlTypeFamily.STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction TRY_BASE64_DECODE_BINARY =
      SqlBasicFunction.create(
          "TRY_BASE64_DECODE_BINARY",
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.

          BodoReturnTypes.VARBINARY_FORCE_NULLABLE,
          argumentRange(1, SqlTypeFamily.STRING, SqlTypeFamily.STRING),
          SqlFunctionCategory.STRING);

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

  public static final SqlFunction UUID_STRING =
      SqlBasicFunction.create(
          "UUID_STRING",
          BodoReturnTypes.VARCHAR_36,
          // either UUID_STRING() for uuid4 or UUID_STRING(<STRING>, <STRING>) for uuid5.
          OperandTypes.STRING_STRING.or(OperandTypes.NILADIC),
          SqlFunctionCategory.STRING);

  private List<SqlOperator> stringOperatorList =
      Arrays.asList(
          CONCAT,
          CONCAT_WS,
          CONTAINS,
          MD5,
          MD5_HEX,
          HEX_ENCODE,
          HEX_DECODE_STRING,
          TRY_HEX_DECODE_STRING,
          HEX_DECODE_BINARY,
          TRY_HEX_DECODE_BINARY,
          BASE64_ENCODE,
          BASE64_DECODE_STRING,
          TRY_BASE64_DECODE_STRING,
          BASE64_DECODE_BINARY,
          TRY_BASE64_DECODE_BINARY,
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
          JAROWINKLER_SIMILARITY,
          FORMAT,
          UCASE,
          LCASE,
          REVERSE,
          ORD,
          CHR,
          CHAR,
          RTRIMMED_LENGTH,
          SPACE,
          STARTSWITH,
          ENDSWITH,
          INSERT,
          POSITION,
          CHARINDEX,
          SHA2,
          SHA2_HEX,
          SPLIT,
          SPLIT_PART,
          STRTOK,
          STRTOK_TO_ARRAY,
          SUBSTRING_INDEX,
          SqlLibraryOperators.TRANSLATE3,
          INITCAP,
          LTRIM,
          RTRIM,
          LENGTH,
          LEN,
          SqlLibraryOperators.RLIKE,
          SqlLibraryOperators.NOT_RLIKE,
          SqlLibraryOperators.ILIKE,
          SqlLibraryOperators.NOT_ILIKE,
          RLIKE,
          LPAD,
          RPAD,
          REPLACE,
          UUID_STRING);

  @Override
  public void lookupOperatorOverloads(
      SqlIdentifier opName,
      @Nullable SqlFunctionCategory category,
      SqlSyntax syntax,
      List<SqlOperator> operatorList,
      SqlNameMatcher nameMatcher) {
    // Heavily copied from Calcite:
    // https://github.com/apache/calcite/blob/4bc916619fd286b2c0cc4d5c653c96a68801d74e/core/src/main/java/org/apache/calcite/sql/util/ListSqlOperatorTable.java#L57
    for (SqlOperator operator : stringOperatorList) {
      // All String Operators added are functions so far.

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
    return stringOperatorList;
  }
}
