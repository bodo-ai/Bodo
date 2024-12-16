package com.bodosql.calcite.application.operatorTables;

import static org.apache.calcite.sql.type.BodoReturnTypes.toArrayReturnType;

import java.util.Arrays;
import java.util.List;
import org.apache.calcite.sql.SqlBasicFunction;
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
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeTransforms;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * Operator table which contains function definitions for functions usable in BodoSQL. This operator
 * table contains definition for functions which handle converting between different SQL types.
 */
public class CastingOperatorTable implements SqlOperatorTable {

  private static @Nullable CastingOperatorTable instance;

  // For conversion to timestamp, snowflake allows the input to be a string,
  // datetime/timestamp expr, or integer/float. If the argument is string, an
  // optional format string is allowed as a second argument. If the argument
  // is a number, an optional scale integer is allowed as a second argument.
  static final SqlOperandTypeChecker toTimestampAcceptedArguments =
      // Note: The order of operands matter for implicit casting
      OperandTypes.NUMERIC
          .or(OperandTypes.DATETIME)
          .or(OperandTypes.CHARACTER)
          .or(OperandTypes.NUMERIC_INTEGER)
          .or(BodoOperandTypes.VARIANT)
          .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER));

  /** Returns the operator table, creating it if necessary. */
  public static synchronized CastingOperatorTable instance() {
    CastingOperatorTable instance = CastingOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new CastingOperatorTable();
      CastingOperatorTable.instance = instance;
    }
    return instance;
  }

  public static final SqlFunction TO_BINARY =
      SqlBasicFunction.create(
          "TO_BINARY",
          // What Value should the return type be
          BodoReturnTypes.VARBINARY_NULLABLE_UNKNOWN_PRECISION,
          // The input can be a string or binary
          OperandTypes.STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction TRY_TO_BINARY =
      SqlBasicFunction.create(
          "TRY_TO_BINARY",
          // What Value should the return type be
          BodoReturnTypes.VARBINARY_FORCE_NULLABLE_UNKNOWN_PRECISION,
          // The input can be a string or binary
          OperandTypes.STRING,
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  // TODO: make all of these actually coerce to cast, so that Calcite can properly typecheck it at
  // compile
  // time as of right now, we do this check in our code.
  public static final SqlFunction TO_BOOLEAN =
      SqlBasicFunction.create(
          "TO_BOOLEAN",
          // What Value should the return type be
          ReturnTypes.BOOLEAN_NULLABLE,
          // For conversion to boolean, snowflake allows a string, numeric expr, or variant
          // (with runtime error being thrown if the underlying value is not string)
          OperandTypes.NUMERIC.or(OperandTypes.STRING).or(BodoOperandTypes.VARIANT),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction TRY_TO_BOOLEAN =
      SqlBasicFunction.create(
          "TRY_TO_BOOLEAN",
          // What Value should the return type be
          // TRY_TO_BOOLEAN will return null for invalid conversions, so we have to force the output
          // to be nullable
          ReturnTypes.BOOLEAN_FORCE_NULLABLE,
          // For conversion to boolean, snowflake allows a string or numeric expr.
          OperandTypes.NUMERIC.or(OperandTypes.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlBasicFunction TO_VARCHAR =
      SqlBasicFunction.create(
          "TO_VARCHAR",
          // TO_VARCHAR only converts nullable types to NULL
          // and has unknown precision
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          // For conversion to string, snowflake any expression type. If the first argument is
          // numeric,
          // datetime/timestamp, or binary, then an optional format string is allowed as a second
          // argument.
          OperandTypes.ANY
              .or(OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.CHARACTER))
              .or(OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.CHARACTER))
              .or(OperandTypes.family(SqlTypeFamily.BINARY, SqlTypeFamily.STRING)),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction TO_CHAR = TO_VARCHAR.withName("TO_CHAR");

  public static final SqlBasicFunction TO_DATE =
      SqlBasicFunction.create(
          "TO_DATE",
          ReturnTypes.DATE_NULLABLE,
          // For conversion to date, snowflake allows a string, datetime, or integer.
          // If the first argument is string, an optional format string is allowed
          // as a second argument.
          // TODO: Remove TIME and Integer
          OperandTypes.DATETIME
              .or(OperandTypes.INTEGER)
              .or(OperandTypes.STRING)
              .or(BodoOperandTypes.VARIANT)
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER)),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATE = TO_DATE.withName("DATE");

  public static final SqlFunction TRY_TO_DATE =
      SqlBasicFunction.create(
          "TRY_TO_DATE",
          // TRY_TO_X functions can create null outputs for non-null inputs if the conversion is
          // invalid
          BodoReturnTypes.DATE_FORCE_NULLABLE,
          // For conversion to date, snowflake allows a string, datetime, or integer.
          // If the first argument is string, an optional format string is allowed
          // as a second argument.
          OperandTypes.DATETIME
              .or(OperandTypes.INTEGER)
              .or(OperandTypes.STRING)
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER)),
          SqlFunctionCategory.TIMEDATE);

  // TO_TIMESTAMP defaults to _NTZ unless the session parameter (which isn't
  // supported yet) says otherwise.
  public static final SqlFunction TO_TIMESTAMP =
      SqlBasicFunction.create(
          "TO_TIMESTAMP",
          BodoReturnTypes.toTimestampReturnType("TO_TIMESTAMP", false, false),
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TRY_TO_TIMESTAMP =
      SqlBasicFunction.create(
          "TRY_TO_TIMESTAMP",
          BodoReturnTypes.toTimestampReturnType("TRY_TO_TIMESTAMP", false, true),
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TO_TIMESTAMP_NTZ =
      SqlBasicFunction.create(
          "TO_TIMESTAMP_NTZ",
          BodoReturnTypes.toTimestampReturnType("TO_TIMESTAMP_NTZ", false, false),
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TRY_TO_TIMESTAMP_NTZ =
      SqlBasicFunction.create(
          "TRY_TO_TIMESTAMP_NTZ",
          BodoReturnTypes.toTimestampReturnType("TRY_TO_TIMESTAMP_NTZ", false, true),
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TO_TIMESTAMP_LTZ =
      SqlBasicFunction.create(
          "TO_TIMESTAMP_LTZ",
          BodoReturnTypes.toTimestampReturnType("TO_TIMESTAMP_LTZ", true, false),
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TRY_TO_TIMESTAMP_LTZ =
      SqlBasicFunction.create(
          "TRY_TO_TIMESTAMP_LTZ",
          BodoReturnTypes.toTimestampReturnType("TRY_TO_TIMESTAMP_LTZ", true, true),
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TO_TIMESTAMP_TZ =
      SqlBasicFunction.create(
          "TO_TIMESTAMP_TZ",
          ReturnTypes.TIMESTAMP_TZ.andThen(SqlTypeTransforms.ARG0_NULLABLE),
          // Use the accepted operand types for all TO_TIMESTAMP functions variant
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TRY_TO_TIMESTAMP_TZ =
      SqlBasicFunction.create(
          "TRY_TO_TIMESTAMP_TZ",
          ReturnTypes.TIMESTAMP_TZ.andThen(SqlTypeTransforms.FORCE_NULLABLE),
          // Use the accepted operand types for all TO_TIMESTAMP functions variant
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TO_DOUBLE =
      SqlBasicFunction.create(
          "TO_DOUBLE",
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // For conversion to double, snowflake allows a string, numeric, or variant expr.
          // If the first argument is string, an optional format string is allowed as a second
          // argument.
          OperandTypes.NUMERIC
              .or(OperandTypes.STRING)
              .or(BodoOperandTypes.VARIANT)
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction TO_VARIANT =
      SqlBasicFunction.create(
              SqlKind.OTHER_FUNCTION, BodoReturnTypes.VARIANT_NULLABLE, OperandTypes.ANY)
          .withName("TO_VARIANT");

  public static final SqlFunction TO_OBJECT =
      SqlBasicFunction.create(
              SqlKind.OTHER_FUNCTION,
              BodoReturnTypes.TO_OBJECT_RETURN_TYPE_INFERENCE,
              OperandTypes.COLLECTION_OR_MAP
                  .or(BodoOperandTypes.VARIANT)
                  .or(OperandTypes.NULLABLE_LITERAL))
          .withName("TO_OBJECT");

  public static final SqlFunction TRY_TO_DOUBLE =
      SqlBasicFunction.create(
          "TRY_TO_DOUBLE",
          // What Value should the return type be
          BodoReturnTypes.DOUBLE_FORCE_NULLABLE,
          // For conversion to double, snowflake allows a string, numeric, or variant expr.
          // If the first argument is string, an optional format string is allowed as a second
          // argument.
          OperandTypes.NUMERIC
              .or(OperandTypes.STRING)
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlBasicFunction TO_TIME =
      SqlBasicFunction.create(
          "TO_TIME",
          BodoReturnTypes.TIME_DEFAULT_PRECISION_NULLABLE,
          // TODO: Remove date as an option.
          OperandTypes.DATETIME
              .or(OperandTypes.CHARACTER)
              .or(BodoOperandTypes.VARIANT)
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER)),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIME = TO_TIME.withName("TIME");

  public static final SqlFunction TRY_TO_TIME =
      SqlBasicFunction.create(
          "TRY_TO_TIME",
          // What Value should the return type be
          BodoReturnTypes.TIME_DEFAULT_PRECISION_FORCE_NULLABLE,
          // TODO: Remove date as an option.
          OperandTypes.DATETIME
              .or(OperandTypes.CHARACTER)
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlBasicFunction TO_NUMBER =
      SqlBasicFunction.create(
          "TO_NUMBER",
          BodoReturnTypes.TO_NUMBER_RET_TYPE,
          BodoOperandTypes.TO_NUMBER_OPERAND_TYPE_CHECKER,
          SqlFunctionCategory.NUMERIC);
  public static final SqlFunction TO_NUMERIC = TO_NUMBER.withName("TO_NUMERIC");
  public static final SqlFunction TO_DECIMAL = TO_NUMBER.withName("TO_DECIMAL");

  public static final SqlBasicFunction TRY_TO_NUMBER =
      SqlBasicFunction.create(
          "TRY_TO_NUMBER",
          BodoReturnTypes.TRY_TO_NUMBER_RET_TYPE,
          BodoOperandTypes.TRY_TO_NUMBER_OPERAND_TYPE_CHECKER,
          SqlFunctionCategory.NUMERIC);
  public static final SqlFunction TRY_TO_NUMERIC = TRY_TO_NUMBER.withName("TRY_TO_NUMERIC");

  public static final SqlFunction TRY_TO_DECIMAL = TRY_TO_NUMBER.withName("TRY_TO_DECIMAL");

  public static final SqlFunction TO_ARRAY =
      SqlBasicFunction.create(
          "TO_ARRAY",
          // What Value should the return type be
          opBinding -> toArrayReturnType(opBinding),
          // The input can be any data type.
          OperandTypes.ANY,
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlOperator INFIX_CAST = SqlLibraryOperators.INFIX_CAST;

  private List<SqlOperator> functionList =
      Arrays.asList(
          INFIX_CAST,
          SqlLibraryOperators.TRY_CAST,
          TO_BINARY,
          TO_BOOLEAN,
          TO_CHAR,
          TO_DATE,
          DATE,
          TO_DOUBLE,
          TO_TIMESTAMP,
          TO_TIMESTAMP_LTZ,
          TO_TIMESTAMP_NTZ,
          TO_TIMESTAMP_TZ,
          TO_VARCHAR,
          TO_TIME,
          TIME,
          TO_NUMBER,
          TO_DECIMAL,
          TO_NUMERIC,
          TRY_TO_BINARY,
          TRY_TO_BOOLEAN,
          TRY_TO_DATE,
          TRY_TO_DOUBLE,
          TRY_TO_TIMESTAMP,
          TRY_TO_TIMESTAMP_LTZ,
          TRY_TO_TIMESTAMP_NTZ,
          TRY_TO_TIMESTAMP_TZ,
          TRY_TO_TIME,
          TRY_TO_NUMBER,
          TRY_TO_DECIMAL,
          TRY_TO_NUMERIC,
          TO_VARIANT,
          TO_OBJECT,
          TO_ARRAY);

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
      if (operator instanceof SqlFunction) {
        // All String Operators added are functions so far.
        SqlFunction func = (SqlFunction) operator;
        if (syntax != func.getSyntax()) {
          continue;
        }
        // Check that the name matches the desired names.
        if (!opName.isSimple() || !nameMatcher.matches(func.getName(), opName.getSimple())) {
          continue;
        }
      }
      if (operator.getSyntax().family != syntax) {
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
