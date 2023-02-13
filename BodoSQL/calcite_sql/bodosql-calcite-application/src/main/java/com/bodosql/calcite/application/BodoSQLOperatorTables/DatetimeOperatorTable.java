package com.bodosql.calcite.application.BodoSQLOperatorTables;

import static com.bodosql.calcite.application.BodoSQLOperatorTables.OperatorTableUtils.argumentRange;
import static com.bodosql.calcite.application.BodoSQLOperatorTables.OperatorTableUtils.isOutputNullableCompile;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import java.util.*;
import javax.annotation.Nullable;
import org.apache.calcite.avatica.util.TimeUnit;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.fun.*;
import org.apache.calcite.sql.type.*;
import org.apache.calcite.sql.validate.SqlNameMatcher;

public final class DatetimeOperatorTable implements SqlOperatorTable {
  /**
   * Determine the return type for a function that outputs a timestamp (possibly tz-aware) based on
   * the first or last argument, such as DATEADD (Snowflake verison), DATEADD (MySQL verison)
   * DATE_ADD, and ADDATE
   *
   * @param binding The operand bindings for the function signature.
   * @return The return type of the first/last argument if either is a timezone-aware type,
   *     otherwise TIMESTAMP
   */
  public static RelDataType timezoneFirstOrLastArgumentReturnType(SqlOperatorBinding binding) {
    List<RelDataType> operandTypes = binding.collectOperandTypes();
    // Determine if the output is nullable.
    boolean nullable = isOutputNullableCompile(operandTypes);
    RelDataTypeFactory typeFactory = binding.getTypeFactory();

    // Determine output type based on the first/last argument
    RelDataType firstArg = operandTypes.get(0);
    RelDataType lastArg = operandTypes.get(operandTypes.size() - 1);
    RelDataType returnType;
    if (firstArg instanceof TZAwareSqlType) {
      // If the input is tzAware the output is as well.
      returnType = firstArg;
    } else if (lastArg instanceof TZAwareSqlType) {
      // If the input is tzAware the output is as well.
      returnType = lastArg;
    } else {
      // Otherwise we output a tzNaive Timestamp.
      // TODO: FIXME once we have proper date support.
      // The output should actually always be a date type for some fns.
      // https://docs.snowflake.com/en/sql-reference/functions/dateadd.html
      returnType = binding.getTypeFactory().createSqlType(SqlTypeName.TIMESTAMP);
    }
    return typeFactory.createTypeWithNullability(returnType, nullable);
  }

  private static @Nullable DatetimeOperatorTable instance;

  /** Returns the Datetime operator table, creating it if necessary. */
  public static synchronized DatetimeOperatorTable instance() {
    DatetimeOperatorTable instance = DatetimeOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new DatetimeOperatorTable();
      DatetimeOperatorTable.instance = instance;
    }
    return instance;
  }

  public static final SqlFunction DATEADD =
      new SqlFunction(
          "DATEADD",
          SqlKind.OTHER_FUNCTION,
          opBinding -> timezoneFirstOrLastArgumentReturnType(opBinding),
          null,
          OperandTypes.or(
              OperandTypes.sequence(
                  "DATEADD(UNIT, VALUE, DATETIME)",
                  OperandTypes.STRING,
                  OperandTypes.INTEGER,
                  OperandTypes.DATETIME),
              OperandTypes.sequence(
                  "DATEADD(DATETIME_OR_DATETIME_STRING, INTERVAL_OR_INTEGER)",
                  OperandTypes.or(OperandTypes.DATETIME, OperandTypes.STRING),
                  OperandTypes.or(OperandTypes.INTERVAL, OperandTypes.INTEGER))),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMEADD =
      new SqlFunction(
          "TIMEADD",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.TIME_NULLABLE,
          null,
          OperandTypes.sequence(
              "TIMEADD(UNIT, VALUE, TIME)",
              OperandTypes.STRING,
              OperandTypes.INTEGER,
              OperandTypes.DATETIME),
          SqlFunctionCategory.TIMEDATE);

  // TODO: Extend the Library Operator and use the builtin Libraries
  public static final SqlFunction DATE_ADD =
      new SqlFunction(
          "DATE_ADD",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding -> timezoneFirstOrLastArgumentReturnType(opBinding),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          /// What Input Types does the function accept. This function accepts the following
          // arguments (Datetime, Interval), (String, Interval)
          OperandTypes.sequence(
              "DATE_ADD(DATETIME_OR_DATETIME_STRING, INTERVAL_OR_INTEGER)",
              OperandTypes.or(OperandTypes.DATETIME, OperandTypes.STRING),
              OperandTypes.or(OperandTypes.INTERVAL, OperandTypes.INTEGER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TO_TIME =
      new SqlFunction(
          "TO_TIME",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIME_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.or(OperandTypes.INTEGER, OperandTypes.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIME =
      new SqlFunction(
          "TIME",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIME_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.or(OperandTypes.INTEGER, OperandTypes.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMEFROMPARTS =
      new SqlFunction(
          "TIMEFROMPARTS",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIME_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.or(
              OperandTypes.sequence(
                  "TIMEFROMPARTS(HOUR, MINUTE, SECOND)",
                  OperandTypes.INTEGER,
                  OperandTypes.INTEGER,
                  OperandTypes.INTEGER),
              OperandTypes.sequence(
                  "TIMEFROMPARTS(HOUR, MINUTE, SECOND, NANOSECOND)",
                  OperandTypes.INTEGER,
                  OperandTypes.INTEGER,
                  OperandTypes.INTEGER,
                  OperandTypes.INTEGER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIME_FROM_PARTS =
      new SqlFunction(
          "TIME_FROM_PARTS",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIME_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.or(
              OperandTypes.sequence(
                  "TIME_FROM_PARTS(HOUR, MINUTE, SECOND)",
                  OperandTypes.INTEGER,
                  OperandTypes.INTEGER,
                  OperandTypes.INTEGER),
              OperandTypes.sequence(
                  "TIME_FROM_PARTS(HOUR, MINUTE, SECOND, NANOSECOND)",
                  OperandTypes.INTEGER,
                  OperandTypes.INTEGER,
                  OperandTypes.INTEGER,
                  OperandTypes.INTEGER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  /**
   * Generate the return type for TO_TIMESTAMP_TZ, TRY_TO_TIMESTAMP_TZ, TO_TIMESTAMP_LTZ and
   * TRY_TO_TIMESTAMP_LTZ
   *
   * @param binding The Operand inputs
   * @param defaultToNaive If there is no timezone provided, default to no timezone. If false,
   *     default to the local timezone
   * @return The function's return type.
   */
  public static RelDataType timestampConstructionOutputType(
      SqlOperatorBinding binding, boolean defaultToNaive) {
    List<RelDataType> operandTypes = binding.collectOperandTypes();
    // Determine if the output is nullable.
    boolean nullable = isOutputNullableCompile(operandTypes);
    RelDataTypeFactory typeFactory = binding.getTypeFactory();

    RelDataType returnType;
    if (operandTypes.size() < 8) {
      if (defaultToNaive) {
        returnType = typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
      } else {
        returnType =
            typeFactory.createTZAwareSqlType(
                binding.getTypeFactory().getTypeSystem().getDefaultTZInfo());
      }
    } else {
      throw new BodoSQLCodegenException(
          "TIMESTAMP_FROM_PARTS_* with timezone argument not supported yet");
    }

    return typeFactory.createTypeWithNullability(returnType, nullable);
  }

  public static final SqlFunction DATE_FROM_PARTS =
      new SqlFunction(
          "DATE_FROM_PARTS",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DATE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.sequence(
              "DATE_FROM_PARTS(YEAR, MONTH, DAY)",
              OperandTypes.INTEGER,
              OperandTypes.INTEGER,
              OperandTypes.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATEFROMPARTS =
      new SqlFunction(
          "DATEFROMPARTS",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DATE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.sequence(
              "DATEFROMPARTS(YEAR, MONTH, DAY)",
              OperandTypes.INTEGER,
              OperandTypes.INTEGER,
              OperandTypes.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMP_FROM_PARTS =
      new SqlFunction(
          "TIMESTAMP_FROM_PARTS",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding -> timestampConstructionOutputType(opBinding, true),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          argumentRange(
              6,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMPFROMPARTS =
      new SqlFunction(
          "TIMESTAMPFROMPARTS",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding -> timestampConstructionOutputType(opBinding, true),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          argumentRange(
              6,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMP_NTZ_FROM_PARTS =
      new SqlFunction(
          "TIMESTAMP_NTZ_FROM_PARTS",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          argumentRange(
              6,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMPNTZFROMPARTS =
      new SqlFunction(
          "TIMESTAMPNTZFROMPARTS",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          argumentRange(
              6,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMP_LTZ_FROM_PARTS =
      new SqlFunction(
          "TIMESTAMP_LTZ_FROM_PARTS",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding -> timestampConstructionOutputType(opBinding, false),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          argumentRange(
              6,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMPLTZFROMPARTS =
      new SqlFunction(
          "TIMESTAMPLTZFROMPARTS",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding -> timestampConstructionOutputType(opBinding, false),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          argumentRange(
              6,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMP_TZ_FROM_PARTS =
      new SqlFunction(
          "TIMESTAMP_TZ_FROM_PARTS",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding -> timestampConstructionOutputType(opBinding, false),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          argumentRange(
              6,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMPTZFROMPARTS =
      new SqlFunction(
          "TIMESTAMPTZFROMPARTS",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding -> timestampConstructionOutputType(opBinding, false),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          argumentRange(
              6,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATE_SUB =
      new SqlFunction(
          "DATE_SUB",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding -> timezoneFirstOrLastArgumentReturnType(opBinding),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts only
          // (Datetime/String, Interval/Integer)
          OperandTypes.sequence(
              "DATE_SUB(DATETIME_OR_DATETIME_STRING, INTERVAL_OR_INTEGER)",
              OperandTypes.or(OperandTypes.DATETIME, OperandTypes.STRING),
              OperandTypes.or(OperandTypes.INTERVAL, OperandTypes.INTEGER)),

          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction SUBDATE =
      new SqlFunction(
          "SUBDATE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding -> timezoneFirstOrLastArgumentReturnType(opBinding),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts only
          // (Datetime/String, Interval/Integer)
          OperandTypes.sequence(
              "SUBDATE(DATETIME_OR_DATETIME_STRING, INTERVAL_OR_INTEGER)",
              OperandTypes.or(OperandTypes.DATETIME, OperandTypes.STRING),
              OperandTypes.or(OperandTypes.INTERVAL, OperandTypes.INTEGER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction ADDDATE =
      new SqlFunction(
          "ADDDATE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding -> timezoneFirstOrLastArgumentReturnType(opBinding),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts either
          // (Datetime, Interval) or (Datetime, Integer)
          OperandTypes.sequence(
              "ADDDATE(DATETIME_OR_DATETIME_STRING, INTERVAL_OR_INTEGER)",
              OperandTypes.or(OperandTypes.DATETIME, OperandTypes.STRING),
              OperandTypes.or(OperandTypes.INTERVAL, OperandTypes.INTEGER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATEDIFF =
      new SqlFunction(
          "DATEDIFF",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts only
          // (Datetime, Datetime)

          OperandTypes.or(
              OperandTypes.sequence(
                  "DATEDIFF(CHARACTER, TIMESTAMP, TIMESTAMP)",
                  OperandTypes.CHARACTER,
                  OperandTypes.TIMESTAMP,
                  OperandTypes.TIMESTAMP),
              OperandTypes.sequence(
                  "DATEDIFF(TIMESTAMP, TIMESTAMP)",
                  OperandTypes.TIMESTAMP,
                  OperandTypes.TIMESTAMP)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction STR_TO_DATE =
      new SqlFunction(
          "STR_TO_DATE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts only
          // (String, Literal String)
          OperandTypes.sequence(
              "STR_TO_DATE(STRING, STRING_LITERAL)",
              OperandTypes.STRING,
              OperandTypes.and(OperandTypes.STRING, OperandTypes.LITERAL)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction GETDATE =
      new SqlFunction(
          "GETDATE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction NOW =
      new SqlFunction(
          "NOW",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding ->
              opBinding
                  .getTypeFactory()
                  .createTZAwareSqlType(
                      opBinding.getTypeFactory().getTypeSystem().getDefaultTZInfo()),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction LOCALTIMESTAMP =
      new SqlFunction(
          "LOCALTIMESTAMP",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction UTC_TIMESTAMP =
      new SqlFunction(
          "UTC_TIMESTAMP",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction UTC_DATE =
      new SqlFunction(
          "UTC_DATE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction MICROSECOND =
      new SqlDatePartFunction("MICROSECOND", TimeUnit.MICROSECOND);

  public static final SqlFunction WEEKOFYEAR = new SqlDatePartFunction("WEEKOFYEAR", TimeUnit.WEEK);

  public static final SqlFunction WEEKISO = new SqlDatePartFunction("WEEKISO", TimeUnit.WEEK);

  public static final SqlFunction DAYNAME =
      new SqlFunction(
          "DAYNAME",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.TIMESTAMP,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DAYOFWEEKISO =
      new SqlDatePartFunction("DAYOFWEEKISO", TimeUnit.ISODOW);

  public static final SqlFunction MONTHNAME =
      new SqlFunction(
          "MONTHNAME",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.TIMESTAMP,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction MONTH_NAME =
      new SqlFunction(
          "MONTH_NAME",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.TIMESTAMP,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction CURDATE =
      new SqlFunction(
          "CURDATE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DATE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATE_FORMAT =
      new SqlFunction(
          "DATE_FORMAT",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts only
          // (String, Literal String)
          OperandTypes.sequence(
              "DATE_FORMAT(TIMESTAMP, STRING_LITERAL)",
              OperandTypes.TIMESTAMP,
              OperandTypes.and(OperandTypes.STRING, OperandTypes.LITERAL)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction MAKEDATE =
      new SqlFunction(
          "MAKEDATE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.sequence(
              "MAKEDATE(INTEGER, INTEGER)", OperandTypes.INTEGER, OperandTypes.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction WEEKDAY =
      new SqlFunction(
          "WEEKDAY",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.TIMESTAMP,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction YEARWEEK =
      new SqlFunction(
          "YEARWEEK",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.TIMESTAMP,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATE_TRUNC =
      new SqlFunction(
          "DATE_TRUNC",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding -> timezoneFirstOrLastArgumentReturnType(opBinding),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.sequence(
              "DATE_TRUNC(STRING_LITERAL, DATETIME)",
              OperandTypes.CHARACTER,
              OperandTypes.DATETIME),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction YEAROFWEEK =
      new SqlFunction(
          "YEAROFWEEK",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.TIMESTAMP,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);
  public static final SqlFunction YEAROFWEEKISO =
      new SqlFunction(
          "YEAROFWEEKISO",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.TIMESTAMP,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATE_PART =
      new SqlFunction(
          "DATE_PART",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.INTEGER_NULLABLE,
          null,
          OperandTypes.sequence(
              "DATE_PART(STRING, TIMESTAMP)", OperandTypes.STRING, OperandTypes.TIMESTAMP),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction NEXT_DAY =
      new SqlFunction(
          "NEXT_DAY",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DATE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this, so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.sequence(
              "PREVIOUS_DAY(DATETIME_OR_DATETIME_STRING, STRING_LITERAL)",
              OperandTypes.or(OperandTypes.DATETIME, OperandTypes.STRING),
              OperandTypes.or(OperandTypes.STRING, OperandTypes.LITERAL)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction PREVIOUS_DAY =
      new SqlFunction(
          "PREVIOUS_DAY",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DATE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this, so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.sequence(
              "PREVIOUS_DAY(DATETIME_OR_DATETIME_STRING, STRING_LITERAL)",
              OperandTypes.or(OperandTypes.DATETIME, OperandTypes.STRING),
              OperandTypes.or(OperandTypes.STRING, OperandTypes.LITERAL)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DAY = new SqlDatePartFunction("DAY", TimeUnit.DAY);

  private List<SqlOperator> functionList =
      Arrays.asList(
          DATE_PART,
          DATEADD,
          DATE_ADD,
          DATE_SUB,
          DATEDIFF,
          DATE_FROM_PARTS,
          DATEFROMPARTS,
          TIMESTAMP_FROM_PARTS,
          TIMESTAMPFROMPARTS,
          TIMESTAMP_NTZ_FROM_PARTS,
          TIMESTAMPNTZFROMPARTS,
          TIMESTAMP_LTZ_FROM_PARTS,
          TIMESTAMPLTZFROMPARTS,
          TIMESTAMP_TZ_FROM_PARTS,
          TIMESTAMPTZFROMPARTS,
          STR_TO_DATE,
          LOCALTIMESTAMP,
          GETDATE,
          NOW,
          UTC_TIMESTAMP,
          UTC_DATE,
          DAYNAME,
          DAYOFWEEKISO,
          MONTHNAME,
          MONTH_NAME,
          MICROSECOND,
          WEEKOFYEAR,
          WEEKISO,
          CURDATE,
          DATE_FORMAT,
          MAKEDATE,
          ADDDATE,
          SUBDATE,
          YEARWEEK,
          WEEKDAY,
          TO_TIME,
          TIMEFROMPARTS,
          TIME_FROM_PARTS,
          TIME,
          TIMEADD,
          DATE_TRUNC,
          YEAROFWEEK,
          YEAROFWEEKISO,
          NEXT_DAY,
          PREVIOUS_DAY,
          DAY);

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
