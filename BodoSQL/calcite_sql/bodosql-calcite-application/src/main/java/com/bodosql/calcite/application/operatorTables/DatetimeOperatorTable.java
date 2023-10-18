package com.bodosql.calcite.application.operatorTables;

import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.standardizeTimeUnit;
import static com.bodosql.calcite.application.operatorTables.OperatorTableUtils.argumentRange;
import static com.bodosql.calcite.application.operatorTables.OperatorTableUtils.isOutputNullableCompile;

import com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.DateTimeType;
import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.rel.type.BodoRelDataTypeFactory;
import com.google.common.collect.Sets;
import java.util.*;
import javax.annotation.Nullable;
import org.apache.calcite.avatica.util.TimeUnit;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.fun.SqlDatePartFunction;
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.type.TZAwareSqlType;
import org.apache.calcite.sql.validate.SqlNameMatcher;

public final class DatetimeOperatorTable implements SqlOperatorTable {
  /**
   * Determine the return type for the DATE_TRUNC function
   *
   * @param binding The operand bindings for the function signature.
   * @return The return type of the function
   */
  public static RelDataType datetruncReturnType(SqlOperatorBinding binding) {
    List<RelDataType> operandTypes = binding.collectOperandTypes();
    // Determine if the output is nullable.
    boolean nullable = isOutputNullableCompile(operandTypes);
    RelDataTypeFactory typeFactory = binding.getTypeFactory();
    return typeFactory.createTypeWithNullability(operandTypes.get(1), nullable);
  }

  /**
   * Call the corresponding return type function for the DATEADD function
   *
   * @param binding The operand bindings for the function signature.
   * @return The return type of the function
   */
  public static RelDataType dateaddReturnType(SqlOperatorBinding binding) {
    List<RelDataType> operandTypes = binding.collectOperandTypes();
    if (operandTypes.size() == 2) return mySqlDateaddReturnType(binding);
    return snowflakeDateaddReturnType(binding, "DATEADD");
  }

  /**
   * Determine the return type of the Snowflake DATEADD function
   *
   * @param binding The operand bindings for the function signature.
   * @return The return type of the function
   */
  public static RelDataType snowflakeDateaddReturnType(SqlOperatorBinding binding, String fnName) {
    List<RelDataType> operandTypes = binding.collectOperandTypes();
    // Determine if the output is nullable.
    boolean nullable = isOutputNullableCompile(operandTypes);
    RelDataTypeFactory typeFactory = binding.getTypeFactory();
    RelDataType datetimeType = operandTypes.get(2);
    String unit;
    if (operandTypes.get(0).getSqlTypeName().equals(SqlTypeName.SYMBOL))
      unit = ((SqlCallBinding) binding).operand(0).toString();
    else unit = binding.getOperandLiteralValue(0, String.class);
    unit = standardizeTimeUnit(fnName, unit, DateTimeType.TIMESTAMP);
    // TODO: refactor standardizeTimeUnit function to change the third argument to SqlTypeName
    RelDataType returnType;
    if (datetimeType.getSqlTypeName().equals(SqlTypeName.DATE)) {
      Set<String> DATE_UNITS =
          new HashSet<>(Arrays.asList("year", "quarter", "month", "week", "day"));
      if (DATE_UNITS.contains(unit))
        returnType = binding.getTypeFactory().createSqlType(SqlTypeName.DATE);
      else returnType = binding.getTypeFactory().createSqlType(SqlTypeName.TIMESTAMP);
    } else {
      // If the input is tzAware/tzNaive/time the output is as well.
      returnType = datetimeType;
    }
    return typeFactory.createTypeWithNullability(returnType, nullable);
  }

  /**
   * Determine the return type of the MySQL DATEADD, DATE_ADD, ADDDATE, DATE_SUB, SUBDATE function
   *
   * @param binding The operand bindings for the function signature.
   * @return The return type of the function
   */
  public static RelDataType mySqlDateaddReturnType(SqlOperatorBinding binding) {
    List<RelDataType> operandTypes = binding.collectOperandTypes();
    // Determine if the output is nullable.
    boolean nullable = isOutputNullableCompile(operandTypes);
    RelDataTypeFactory typeFactory = binding.getTypeFactory();
    RelDataType datetimeType = operandTypes.get(0);
    RelDataType returnType;

    if (operandTypes.get(1).getSqlTypeName().equals(SqlTypeName.INTEGER)) {
      // when the second argument is integer, it is equivalent to adding day interval
      if (datetimeType.getSqlTypeName().equals(SqlTypeName.DATE))
        returnType = binding.getTypeFactory().createSqlType(SqlTypeName.DATE);
      else if (datetimeType instanceof TZAwareSqlType) returnType = datetimeType;
      else returnType = binding.getTypeFactory().createSqlType(SqlTypeName.TIMESTAMP);
    } else {
      // if the first argument is date, the return type depends on the interval type
      if (datetimeType.getSqlTypeName().equals(SqlTypeName.DATE)) {
        Set<SqlTypeName> DATE_INTERVAL_TYPES =
            Sets.immutableEnumSet(
                SqlTypeName.INTERVAL_YEAR_MONTH,
                SqlTypeName.INTERVAL_YEAR,
                SqlTypeName.INTERVAL_MONTH,
                SqlTypeName.INTERVAL_DAY);
        if (DATE_INTERVAL_TYPES.contains(operandTypes.get(1).getSqlTypeName()))
          returnType = binding.getTypeFactory().createSqlType(SqlTypeName.DATE);
        else returnType = binding.getTypeFactory().createSqlType(SqlTypeName.TIMESTAMP);
      } else if (datetimeType instanceof TZAwareSqlType) {
        returnType = datetimeType;
      } else {
        returnType = binding.getTypeFactory().createSqlType(SqlTypeName.TIMESTAMP);
      }
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
          opBinding -> dateaddReturnType(opBinding),
          null,
          OperandTypes.or(
              OperandTypes.sequence(
                  "DATEADD(UNIT, VALUE, DATETIME)",
                  OperandTypes.ANY,
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
          opBinding -> snowflakeDateaddReturnType(opBinding, "TIMEADD"),
          null,
          OperandTypes.sequence(
              "TIMEADD(UNIT, VALUE, TIME)",
              OperandTypes.ANY,
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
          opBinding -> mySqlDateaddReturnType(opBinding),
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
          OperandTypes.or(
              OperandTypes.or(OperandTypes.DATETIME, OperandTypes.STRING),
              OperandTypes.sequence(
                  "TO_TIME(STRING, STRING)", OperandTypes.STRING, OperandTypes.STRING)),
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
          OperandTypes.or(
              OperandTypes.or(OperandTypes.DATETIME, OperandTypes.STRING),
              OperandTypes.sequence(
                  "TIME(STRING, STRING)", OperandTypes.STRING, OperandTypes.STRING)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TRY_TO_TIME =
      new SqlFunction(
          "TRY_TO_TIME",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIME_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.or(
              OperandTypes.or(OperandTypes.DATETIME, OperandTypes.STRING),
              OperandTypes.sequence(
                  "TRY_TO_TIME(STRING, STRING)", OperandTypes.STRING, OperandTypes.STRING)),
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
        returnType = BodoRelDataTypeFactory.createTZAwareSqlType(typeFactory, null);
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
          opBinding -> mySqlDateaddReturnType(opBinding),
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
          opBinding -> mySqlDateaddReturnType(opBinding),
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
          opBinding -> mySqlDateaddReturnType(opBinding),
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
                  "DATEDIFF(TIMESTAMP/DATE, TIMESTAMP/DATE)",
                  OperandTypes.DATETIME,
                  OperandTypes.DATETIME),
              OperandTypes.sequence(
                  "DATEDIFF(UNIT, TIMESTAMP/DATE/TIME, TIMESTAMP/DATE/TIME)",
                  OperandTypes.ANY,
                  OperandTypes.DATETIME,
                  OperandTypes.DATETIME)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMEDIFF =
      new SqlFunction(
          "TIMEDIFF",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.sequence(
              "DATEDIFF(UNIT, TIMESTAMP/DATE/TIME, TIMESTAMP/DATE/TIME)",
              OperandTypes.ANY,
              OperandTypes.DATETIME,
              OperandTypes.DATETIME),
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
          BodoReturnTypes.TZAWARE_TIMESTAMP,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  // CURRENT_TIMESTAMP is already supported inside Calcite and will be picked
  // up automatically. No need to implement it again here

  // LOCALTIMESTAMP is already supported inside Calcite and will be picked
  // up automatically. No need to implement it again here

  public static final SqlFunction SYSTIMESTAMP =
      new SqlFunction(
          "SYSTIMESTAMP",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          BodoReturnTypes.TZAWARE_TIMESTAMP,
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
          BodoReturnTypes.TZAWARE_TIMESTAMP,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  // CURRENT_TIME is already supported inside Calcite and will be picked
  // up automatically. No need to implement it again here

  // LOCALTIME is already supported inside Calcite and will be picked
  // up automatically. No need to implement it again here

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

  public static final SqlFunction SYSDATE =
      new SqlFunction(
          "SYSDATE",
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
          // This always returns a 3 letter value.
          BodoReturnTypes.VARCHAR_3_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
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
          // MONTHNAME always return a 3 character month abbreviation
          BodoReturnTypes.VARCHAR_3_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction MONTH_NAME =
      new SqlFunction(
          "MONTH_NAME",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // MONTH_NAME always return a 3 character month abbreviation
          BodoReturnTypes.VARCHAR_3_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction MONTHS_BETWEEN =
      new SqlFunction(
          "MONTHS_BETWEEN",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          /// What Input Types does the function accept.
          OperandTypes.sequence(
              "MONTHS_BETWEEN(DATETIME, DATETIME)",
              OperandTypes.or(OperandTypes.DATE, OperandTypes.DATETIME, OperandTypes.TIMESTAMP),
              OperandTypes.or(OperandTypes.DATE, OperandTypes.DATETIME, OperandTypes.TIMESTAMP)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction ADD_MONTHS =
      new SqlFunction(
          "ADD_MONTHS",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.ARG0,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          /// What Input Types does the function accept. This function accepts the following
          // arguments (Datetime, Interval), (String, Interval)
          OperandTypes.sequence(
              "ADD_MONTHS(DATETIME, NUMERIC)",
              OperandTypes.or(OperandTypes.DATE, OperandTypes.DATETIME, OperandTypes.TIMESTAMP),
              OperandTypes.NUMERIC),
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
          // Precision cannot be statically determined.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts only
          // (String, Literal String)
          OperandTypes.sequence(
              "DATE_FORMAT(DATE/TIMESTAMP, STRING_LITERAL)",
              OperandTypes.or(OperandTypes.DATE, OperandTypes.TIMESTAMP),
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
          ReturnTypes.DATE,
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
          OperandTypes.DATETIME,
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
          opBinding -> datetruncReturnType(opBinding),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.sequence(
              "DATE_TRUNC(UNIT, DATETIME)", OperandTypes.ANY, OperandTypes.DATETIME),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIME_SLICE =
      new SqlFunction(
          "TIME_SLICE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.ARG0,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.or(
              OperandTypes.sequence(
                  "TIME_SLICE(DATETIME, INT, UNIT)",
                  OperandTypes.or(OperandTypes.DATETIME, OperandTypes.TIMESTAMP),
                  OperandTypes.INTEGER,
                  OperandTypes.ANY),
              OperandTypes.sequence(
                  "TIME_SLICE(DATETIME, INT, UNIT, STRING)",
                  OperandTypes.or(OperandTypes.DATETIME, OperandTypes.TIMESTAMP),
                  OperandTypes.INTEGER,
                  OperandTypes.ANY,
                  OperandTypes.STRING)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  private static RelDataType truncReturnType(SqlOperatorBinding binding) {
    RelDataTypeFactory typeFactory = binding.getTypeFactory();
    RelDataType inputType = binding.getOperandType(0);

    if (SqlTypeUtil.isNumeric(inputType)) {
      return typeFactory.createTypeWithNullability(inputType, inputType.isNullable());
    } else {
      return datetruncReturnType(binding);
    }
  }

  public static final SqlFunction TRUNC =
      new SqlFunction(
          "TRUNC",
          SqlKind.OTHER_FUNCTION,
          opBinding -> truncReturnType(opBinding),
          null,
          // What Input Types does the function accept.
          OperandTypes.or(
              OperandTypes.sequence(
                  "TRUNC(UNIT, DATETIME)", OperandTypes.ANY, OperandTypes.DATETIME),
              argumentRange(1, SqlTypeFamily.NUMERIC, SqlTypeFamily.INTEGER)),
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

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
          OperandTypes.DATETIME,
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
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATE_PART =
      new SqlFunction(
          "DATE_PART",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.INTEGER_NULLABLE,
          null,
          OperandTypes.sequence(
              "DATE_PART(UNIT, DATETIME)", OperandTypes.ANY, OperandTypes.DATETIME),
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

  public static final SqlFunction CONVERT_TIMEZONE =
      new SqlFunction(
          "CONVERT_TIMEZONE",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.TIMESTAMP,
          null,
          OperandTypes.or(
              OperandTypes.CHARACTER_CHARACTER_DATETIME,
              OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.DATETIME)),
          SqlFunctionCategory.TIMEDATE);

  private List<SqlOperator> functionList =
      Arrays.asList(
          CONVERT_TIMEZONE,
          DATE_PART,
          DATEADD,
          DATE_ADD,
          DATE_SUB,
          DATEDIFF,
          TIMEDIFF,
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
          GETDATE,
          SYSTIMESTAMP,
          NOW,
          UTC_TIMESTAMP,
          UTC_DATE,
          SYSDATE,
          DAYNAME,
          DAYOFWEEKISO,
          MONTHNAME,
          MONTH_NAME,
          MONTHS_BETWEEN,
          ADD_MONTHS,
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
          TRY_TO_TIME,
          TIMEFROMPARTS,
          TIME_FROM_PARTS,
          TIME_SLICE,
          TIME,
          TIMEADD,
          TRUNC,
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
