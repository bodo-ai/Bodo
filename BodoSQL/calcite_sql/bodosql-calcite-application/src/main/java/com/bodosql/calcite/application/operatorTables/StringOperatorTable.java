package com.bodosql.calcite.application.operatorTables;

import static com.bodosql.calcite.application.operatorTables.OperatorTableUtils.argumentRange;
import static org.apache.calcite.sql.type.BodoReturnTypes.SPLIT_RETURN_TYPE;

import java.util.Arrays;
import java.util.List;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
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
      new SqlFunction(
          "CONCAT",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // Concat sums together all input precision.
          // DYADIC_STRING_SUM always expects at least two arguments, which means that we need
          // special logic to handle the single argument version of CONCAT.
          BodoReturnTypes.CONCAT_RETURN_TYPE,
          // What should be used to infer operand types. We don't use
          // this, so we set it to None.
          null,
          OperandTypes.repeat(SqlOperandCountRanges.from(1), OperandTypes.BINARY)
              .or(OperandTypes.repeat(SqlOperandCountRanges.from(1), OperandTypes.CHARACTER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction CONCAT_WS =
      new SqlFunction(
          "CONCAT_WS",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // Concat sums together all input precision and
          // includes the separator where appropriate.
          // Unlike CONCAT, CONCAT_WS always takes at least two arguments - the separator and the
          // variadic list of expressions. This means it's safe to always call DYADIC_STRING_SUM
          // which expects at least 2 args.
          BodoReturnTypes.CONCAT_WS_RETURN_TYPE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.repeat(SqlOperandCountRanges.from(2), OperandTypes.BINARY)
              .or(OperandTypes.repeat(SqlOperandCountRanges.from(2), OperandTypes.CHARACTER)),
          SqlFunctionCategory.STRING);

  public static final SqlFunction MID =
      new SqlFunction(
          "MID",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // Match input precision. The substring is at most the same string.
          ReturnTypes.ARG0_NULLABLE_VARYING,
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
          // Match input precision. The substring is at most the same string.
          ReturnTypes.ARG0_NULLABLE_VARYING,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. CHAR_INT_INT or CHAR_INT (length is
          // optional)
          OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER)
              .or(OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.INTEGER)),
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
          ReturnTypes.BIGINT_NULLABLE,
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
          // Match input precision. The substring is at most the same string.
          ReturnTypes.ARG0_NULLABLE_VARYING,
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
          // Match input precision. The substring is at most the same string.
          ReturnTypes.ARG0_NULLABLE_VARYING,
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
          // Repeat has an unknown static output precision.
          BodoReturnTypes.ARG0_NULLABLE_VARYING_UNKNOWN_PRECISION,
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
          OperandTypes.STRING_STRING.or(OperandTypes.STRING_STRING_INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction JAROWINKLER_SIMILARITY =
      new SqlFunction(
          "JAROWINKLER_SIMILARITY",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
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
          // Return type has an unknown precision, but it should only be
          // null if there is a null input.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
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
          // Char outputs a single character
          BodoReturnTypes.VARCHAR_1_NULLABLE,
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
          // Char outputs a single character
          BodoReturnTypes.VARCHAR_1_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction RTRIMMED_LENGTH =
      new SqlFunction(
          "RTRIMMED_LENGTH",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.CHARACTER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction SPACE =
      new SqlFunction(
          "SPACE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.INTEGER,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction STARTSWITH =
      new SqlFunction(
          "STARTSWITH",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BOOLEAN_NULLABLE,
          null,
          OperandTypes.STRING_STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction ENDSWITH =
      new SqlFunction(
          "ENDSWITH",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BOOLEAN_NULLABLE,
          null,
          OperandTypes.STRING_STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction INSERT =
      new SqlFunction(
          "INSERT",
          SqlKind.OTHER_FUNCTION,
          // Snowflake calculates the resulting precision
          // as 2 * arg0 + arg3. This seems like an error,
          // as the max possible precision should be arg0 + arg3,
          // but we will match Snowflake.
          BodoReturnTypes.INSERT_RETURN_TYPE,
          null,
          OperandTypes.family(
              SqlTypeFamily.STRING,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction CHARINDEX =
      new SqlFunction(
          "CHARINDEX",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.INTEGER_NULLABLE,
          null,
          argumentRange(2, SqlTypeFamily.STRING, SqlTypeFamily.STRING, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction POSITION =
      new SqlFunction(
          "POSITION",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.INTEGER_NULLABLE,
          null,
          argumentRange(2, SqlTypeFamily.STRING, SqlTypeFamily.STRING, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction SPLIT_PART =
      new SqlFunction(
          "SPLIT_PART",
          SqlKind.OTHER_FUNCTION,
          // The precision is the same as in the worst
          // case we keep the same string.
          ReturnTypes.ARG0_NULLABLE_VARYING,
          null,
          OperandTypes.STRING_STRING_INTEGER,
          SqlFunctionCategory.STRING);

  public static final SqlFunction STRTOK =
      new SqlFunction(
          "STRTOK",
          SqlKind.OTHER_FUNCTION,
          // The precision is the same as in the worst
          // case we keep the same string.
          ReturnTypes.ARG0_NULLABLE_VARYING,
          null,
          OperandTypes.STRING.or(OperandTypes.STRING_STRING).or(OperandTypes.STRING_STRING_INTEGER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction STRTOK_TO_ARRAY =
      new SqlFunction(
          "STRTOK_TO_ARRAY",
          SqlKind.OTHER_FUNCTION,
          BodoReturnTypes.TO_NULLABLE_VARYING_ARRAY,
          null,
          OperandTypes.CHARACTER.or(BodoOperandTypes.CHARACTER_CHARACTER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction LTRIM =
      new SqlFunction(
          "LTRIM",
          SqlKind.OTHER,
          // Precision matches in the input.
          ReturnTypes.ARG0_NULLABLE_VARYING,
          null,
          argumentRange(1, SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction RTRIM =
      new SqlFunction(
          "RTRIM",
          SqlKind.OTHER,
          // Precision matches in the input.
          ReturnTypes.ARG0_NULLABLE_VARYING,
          null,
          argumentRange(1, SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction SUBSTRING_INDEX =
      new SqlFunction(
          "SUBSTRING_INDEX",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // In the worst case we return the whole string,
          // so maintain the precision.
          ReturnTypes.ARG0_NULLABLE_VARYING,
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

  /* This RLIKE SqlFunction is to support SQL queries of the form:
   * RLIKE(<subject>, <pattern>, [<parameters>]) as opposed to the RLIKE SqlLikeOperator
   * which supports SQL queries of the form: <subject> RLIKE <pattern>
   * */
  public static final SqlFunction RLIKE =
      new SqlFunction(
          "RLIKE",
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
          // Return string has an unknown precision.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
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
          // Match precision because this is a substring
          // returns null if substring DNE
          BodoReturnTypes.ARG0_FORCE_NULLABLE_VARYING,
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
          // This is a mappable function so the precision is the same.
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.STRING.or(OperandTypes.STRING_STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction CONTAINS =
      new SqlFunction(
          "CONTAINS",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BOOLEAN,
          null,
          OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction SPLIT =
      new SqlFunction(
          "SPLIT",
          SqlKind.OTHER_FUNCTION,
          // Return type for split is the exact same as TO_ARRAY
          SPLIT_RETURN_TYPE,
          null,
          OperandTypes.CHARACTER_CHARACTER,
          SqlFunctionCategory.STRING);

  public static final SqlFunction SHA2 =
      new SqlFunction(
          "SHA2",
          SqlKind.OTHER_FUNCTION,
          // SHA2 outputs at most 128 characters.
          BodoReturnTypes.VARCHAR_128_NULLABLE,
          null,
          OperandTypes.STRING.or(OperandTypes.STRING_INTEGER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction SHA2_HEX =
      new SqlFunction(
          "SHA2_HEX",
          SqlKind.OTHER_FUNCTION,
          // SHA2 outputs at most 128 characters.
          BodoReturnTypes.VARCHAR_128_NULLABLE,
          null,
          OperandTypes.STRING.or(OperandTypes.STRING_INTEGER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction MD5 =
      new SqlFunction(
          "MD5",
          SqlKind.OTHER_FUNCTION,
          // MD5 outputs at most 32 characters.
          BodoReturnTypes.VARCHAR_32_NULLABLE,
          null,
          OperandTypes.STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction MD5_HEX =
      new SqlFunction(
          "MD5_HEX",
          SqlKind.OTHER_FUNCTION,
          // MD5 outputs at most 32 characters.
          BodoReturnTypes.VARCHAR_32_NULLABLE,
          null,
          OperandTypes.STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction HEX_ENCODE =
      new SqlFunction(
          "HEX_ENCODE",
          SqlKind.OTHER_FUNCTION,
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          null,
          argumentRange(1, SqlTypeFamily.STRING, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.STRING);

  public static final SqlFunction HEX_DECODE_STRING =
      new SqlFunction(
          "HEX_DECODE_STRING",
          SqlKind.OTHER_FUNCTION,
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          null,
          OperandTypes.STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction HEX_DECODE_BINARY =
      new SqlFunction(
          "HEX_DECODE_BINARY",
          SqlKind.OTHER_FUNCTION,
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          ReturnTypes.explicit(SqlTypeName.VARBINARY).andThen(SqlTypeTransforms.TO_NULLABLE),
          null,
          OperandTypes.STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction TRY_HEX_DECODE_STRING =
      new SqlFunction(
          "TRY_HEX_DECODE_STRING",
          SqlKind.OTHER_FUNCTION,
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_FORCE_NULLABLE,
          null,
          OperandTypes.STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction TRY_HEX_DECODE_BINARY =
      new SqlFunction(
          "TRY_HEX_DECODE_BINARY",
          SqlKind.OTHER_FUNCTION,
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARBINARY_FORCE_NULLABLE,
          null,
          OperandTypes.STRING,
          SqlFunctionCategory.STRING);

  public static final SqlFunction BASE64_ENCODE =
      new SqlFunction(
          "BASE64_ENCODE",
          SqlKind.OTHER_FUNCTION,
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          null,
          argumentRange(1, SqlTypeFamily.STRING, SqlTypeFamily.INTEGER, SqlTypeFamily.STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction BASE64_DECODE_STRING =
      new SqlFunction(
          "BASE64_DECODE_STRING",
          SqlKind.OTHER_FUNCTION,
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          null,
          argumentRange(1, SqlTypeFamily.STRING, SqlTypeFamily.STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction TRY_BASE64_DECODE_STRING =
      new SqlFunction(
          "TRY_BASE64_DECODE_STRING",
          SqlKind.OTHER_FUNCTION,
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_FORCE_NULLABLE,
          null,
          argumentRange(1, SqlTypeFamily.STRING, SqlTypeFamily.STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction BASE64_DECODE_BINARY =
      new SqlFunction(
          "BASE64_DECODE_BINARY",
          SqlKind.OTHER_FUNCTION,
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.
          ReturnTypes.explicit(SqlTypeName.VARBINARY).andThen(SqlTypeTransforms.TO_NULLABLE),
          null,
          argumentRange(1, SqlTypeFamily.STRING, SqlTypeFamily.STRING),
          SqlFunctionCategory.STRING);

  public static final SqlFunction TRY_BASE64_DECODE_BINARY =
      new SqlFunction(
          "TRY_BASE64_DECODE_BINARY",
          SqlKind.OTHER_FUNCTION,
          // [BSE-1714] TODO: calculate the output precision in terms of the input precision
          // and the second argument.

          BodoReturnTypes.VARBINARY_FORCE_NULLABLE,
          null,
          argumentRange(1, SqlTypeFamily.STRING, SqlTypeFamily.STRING),
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
          LEN,
          LENGTH,
          SqlLibraryOperators.RLIKE,
          SqlLibraryOperators.NOT_RLIKE,
          SqlLibraryOperators.ILIKE,
          SqlLibraryOperators.NOT_ILIKE,
          RLIKE);

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
