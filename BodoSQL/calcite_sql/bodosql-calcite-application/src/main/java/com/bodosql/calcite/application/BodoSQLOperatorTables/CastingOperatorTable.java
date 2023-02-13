package com.bodosql.calcite.application.BodoSQLOperatorTables;

import static com.bodosql.calcite.application.BodoSQLOperatorTables.OperatorTableUtils.isOutputNullableCompile;

import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.fun.SqlLibraryOperators;
import org.apache.calcite.sql.type.*;
import org.apache.calcite.sql.validate.SqlNameMatcher;

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
      OperandTypes.or(
          OperandTypes.CHARACTER,
          OperandTypes.DATETIME,
          OperandTypes.DATE,
          OperandTypes.TIMESTAMP,
          OperandTypes.NUMERIC,
          OperandTypes.sequence(
              "TO_TIMESTAMP(CHARACTER, CHARACTER)", OperandTypes.CHARACTER, OperandTypes.CHARACTER),
          OperandTypes.NUMERIC_INTEGER);

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

  // TODO: make all of these actually coerce to cast, so that Calcite can properly typecheck it at
  // compile
  // time as of right now, we do this check in our code.
  public static final SqlFunction TO_BOOLEAN =
      new SqlFunction(
          "TO_BOOLEAN",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BOOLEAN_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // For conversion to boolean, snowflake allows a string or numeric expr.
          OperandTypes.or(OperandTypes.STRING, OperandTypes.NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction TRY_TO_BOOLEAN =
      new SqlFunction(
          "TRY_TO_BOOLEAN",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BOOLEAN_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // For conversion to boolean, snowflake allows a string or numeric expr.
          OperandTypes.or(OperandTypes.STRING, OperandTypes.NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction TO_CHAR =
      new SqlFunction(
          "TO_CHAR",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // For conversion to string, snowflake any expresison type. If the first argument is
          // numeric,
          // datetime/timestamp, or binary, then an optional format string is allowed as a second
          // argument.
          OperandTypes.or(
              OperandTypes.ANY,
              OperandTypes.or(
                  OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.STRING),
                  OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.STRING),
                  OperandTypes.family(SqlTypeFamily.TIMESTAMP, SqlTypeFamily.STRING),
                  OperandTypes.family(SqlTypeFamily.BINARY, SqlTypeFamily.STRING))),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction TO_VARCHAR =
      new SqlFunction(
          "TO_VARCHAR",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // For conversion to string, snowflake any expression type. If the first argument is
          // numeric,
          // datetime/timestamp, or binary, then an optional format string is allowed as a second
          // argument.
          OperandTypes.or(
              OperandTypes.ANY,
              OperandTypes.or(
                  OperandTypes.family(SqlTypeFamily.NUMERIC, SqlTypeFamily.STRING),
                  OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.STRING),
                  OperandTypes.family(SqlTypeFamily.TIMESTAMP, SqlTypeFamily.STRING),
                  OperandTypes.family(SqlTypeFamily.BINARY, SqlTypeFamily.STRING))),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction TO_DATE =
      new SqlFunction(
          "TO_DATE",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DATE,
          null,
          // For conversion to date, snowflake allows a string, datetime, or integer.
          // If the first argument is string, an optional format string is allowed
          // as a second argument.
          OperandTypes.or(
              OperandTypes.or(
                  OperandTypes.STRING,
                  OperandTypes.DATETIME,
                  OperandTypes.DATE,
                  OperandTypes.TIMESTAMP,
                  OperandTypes.INTEGER),
              OperandTypes.STRING_STRING),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TRY_TO_DATE =
      new SqlFunction(
          "TRY_TO_DATE",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DATE,
          null,
          // For conversion to date, snowflake allows a string, datetime, or integer.
          // If the first argument is string, an optional format string is allowed
          // as a second argument.
          OperandTypes.or(
              OperandTypes.or(
                  OperandTypes.STRING,
                  OperandTypes.DATETIME,
                  OperandTypes.DATE,
                  OperandTypes.TIMESTAMP,
                  OperandTypes.INTEGER),
              OperandTypes.STRING_STRING),
          SqlFunctionCategory.TIMEDATE);

  /**
   * Generate the return type for TO_TIMESTAMP_TZ, TRY_TO_TIMESTAMP_TZ, TO_TIMESTAMP_LTZ and
   * TRY_TO_TIMESTAMP_LTZ
   *
   * @param binding The Operand inputs
   * @param runtimeFailureIsNull Can a runtime failure cause a null output? True for
   *     TRY_TO_TIMESTAMP and false for TO_TIMESTAMP (because it raises an exception instead).
   * @param keepTimezone If True input has a timezone, keep it in the output. True for _TZ and false
   *     for _LTZ, since _LTZ always overrides with the local time zone
   * @return The function's return type.
   */
  public static RelDataType toTimestampTZReturnType(
      SqlOperatorBinding binding, boolean runtimeFailureIsNull, boolean keepTimezone) {
    List<RelDataType> operandTypes = binding.collectOperandTypes();
    // Determine if the output is nullable.
    boolean nullable = isOutputNullableCompile(operandTypes);
    RelDataTypeFactory typeFactory = binding.getTypeFactory();

    // Determine output type based on arg0
    RelDataType arg0 = operandTypes.get(0);
    RelDataType returnType;
    boolean isTzAware = (arg0 instanceof TZAwareSqlType);
    if (keepTimezone && isTzAware) {
      // If the input is tzAware the output is as well.
      returnType = arg0;
    } else {
      // Otherwise we output a timezone-aware Timestamp with the local timestamp.
      returnType =
          typeFactory.createTZAwareSqlType(
              binding.getTypeFactory().getTypeSystem().getDefaultTZInfo());
      if (runtimeFailureIsNull) {
        // Note this path includes arguments for 0 that can fail at runtime.
        // If this runtimeFailureIsNull is set then failed conversions make
        // the output NULL and therefore the type nullable.
        // NOTE: Timestamp/Date will never fail to convert so they won't
        // make the output nullable.
        SqlTypeName typeName = arg0.getSqlTypeName();
        boolean conversionCantFail =
            typeName.equals(SqlTypeName.TIMESTAMP)
                || typeName.equals(SqlTypeName.DATE)
                || isTzAware;
        nullable = nullable || (!conversionCantFail);
      }
    }
    return typeFactory.createTypeWithNullability(returnType, nullable);
  }

  // TO_TIMESTAMP defaults to _NTZ unless the session parameter (which isn't
  // supported yet) says otherwise.
  public static final SqlFunction TO_TIMESTAMP =
      new SqlFunction(
          "TO_TIMESTAMP",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.TIMESTAMP,
          null,
          // Use the accepted operand types for all TO_TIMESTAMP functions variant
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TRY_TO_TIMESTAMP =
      new SqlFunction(
          "TRY_TO_TIMESTAMP",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.TIMESTAMP,
          null,
          // Use the accepted operand types for all TO_TIMESTAMP functions variant
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TO_TIMESTAMP_NTZ =
      new SqlFunction(
          "TO_TIMESTAMP_NTZ",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.TIMESTAMP,
          null,
          // Use the accepted operand types for all TO_TIMESTAMP functions variant
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TRY_TO_TIMESTAMP_NTZ =
      new SqlFunction(
          "TRY_TO_TIMESTAMP_NTZ",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.TIMESTAMP,
          null,
          // Use the accepted operand types for all TO_TIMESTAMP functions variant
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TO_TIMESTAMP_LTZ =
      new SqlFunction(
          "TO_TIMESTAMP_LTZ",
          SqlKind.OTHER_FUNCTION,
          opBinding -> toTimestampTZReturnType(opBinding, false, false),
          null,
          // Use the accepted operand types for all TO_TIMESTAMP functions variant
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TRY_TO_TIMESTAMP_LTZ =
      new SqlFunction(
          "TRY_TO_TIMESTAMP_LTZ",
          SqlKind.OTHER_FUNCTION,
          opBinding -> toTimestampTZReturnType(opBinding, true, false),
          null,
          // Use the accepted operand types for all TO_TIMESTAMP functions variant
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TO_TIMESTAMP_TZ =
      new SqlFunction(
          "TO_TIMESTAMP_TZ",
          SqlKind.OTHER_FUNCTION,
          opBinding -> toTimestampTZReturnType(opBinding, false, true),
          null,
          // Use the accepted operand types for all TO_TIMESTAMP functions variant
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TRY_TO_TIMESTAMP_TZ =
      new SqlFunction(
          "TRY_TO_TIMESTAMP_TZ",
          SqlKind.OTHER_FUNCTION,
          opBinding -> toTimestampTZReturnType(opBinding, true, true),
          null,
          // Use the accepted operand types for all TO_TIMESTAMP functions variant
          toTimestampAcceptedArguments,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TO_DOUBLE =
      new SqlFunction(
          "TO_DOUBLE",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // For conversion to double, snowflake allows a string, numeric, or variant expr.
          // If the first argument is string, an optional format string is allowed as a second
          // argument.
          OperandTypes.or(OperandTypes.STRING, OperandTypes.NUMERIC, OperandTypes.STRING_STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction TRY_TO_DOUBLE =
      new SqlFunction(
          "TRY_TO_DOUBLE",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // For conversion to double, snowflake allows a string, numeric, or variant expr.
          // If the first argument is string, an optional format string is allowed as a second
          // argument.
          OperandTypes.or(OperandTypes.STRING, OperandTypes.NUMERIC, OperandTypes.STRING_STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlOperator INFIX_CAST = SqlLibraryOperators.INFIX_CAST;

  private List<SqlOperator> functionList =
      Arrays.asList(
          INFIX_CAST,
          TO_BOOLEAN,
          TO_CHAR,
          TO_DATE,
          TO_DOUBLE,
          TO_TIMESTAMP,
          TO_TIMESTAMP_LTZ,
          TO_TIMESTAMP_NTZ,
          TO_TIMESTAMP_TZ,
          TO_VARCHAR,
          TRY_TO_BOOLEAN,
          TRY_TO_DATE,
          TRY_TO_DOUBLE,
          TRY_TO_TIMESTAMP,
          TRY_TO_TIMESTAMP_LTZ,
          TRY_TO_TIMESTAMP_NTZ,
          TRY_TO_TIMESTAMP_TZ);

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
