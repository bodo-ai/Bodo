package com.bodosql.calcite.application.operatorTables;

import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.standardizeTimeUnit;
import static com.bodosql.calcite.application.operatorTables.OperatorTableUtils.argumentRange;
import static com.bodosql.calcite.application.operatorTables.OperatorTableUtils.isOutputNullableCompile;

import com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.DateTimeType;
import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem;
import com.bodosql.calcite.rel.type.BodoRelDataTypeFactory;
import com.google.common.collect.Sets;
import java.util.*;
import javax.annotation.Nullable;
import org.apache.calcite.avatica.util.TimeUnit;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlBasicFunction;
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
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
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
    RelDataType typeArg0 = operandTypes.get(0);
    if (typeArg0.getSqlTypeName().equals(SqlTypeName.SYMBOL)
        || SqlTypeFamily.INTERVAL_YEAR_MONTH.contains(typeArg0)
        || SqlTypeFamily.INTERVAL_DAY_TIME.contains(typeArg0)) {
      unit = ((SqlCallBinding) binding).operand(0).toString();
    } else {
      unit = binding.getOperandLiteralValue(0, String.class);
    }
    unit = standardizeTimeUnit(fnName, unit, DateTimeType.TIMESTAMP);
    // TODO: refactor standardizeTimeUnit function to change the third argument to
    // SqlTypeName
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
          OperandTypes.sequence(
                  "DATEADD(UNIT, VALUE, DATETIME)",
                  OperandTypes.ANY,
                  OperandTypes.NUMERIC,
                  OperandTypes.DATETIME)
              .or(OperandTypes.DATETIME_INTERVAL)
              .or(OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.INTEGER))
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.DATETIME_INTERVAL))
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.INTEGER)),
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
              OperandTypes.NUMERIC,
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
          OperandTypes.DATETIME_INTERVAL
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.DATETIME_INTERVAL))
              .or(OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.INTEGER))
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.INTEGER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlBasicFunction TO_TIME =
      SqlBasicFunction.create(
          "TO_TIME",
          ReturnTypes.TIME_NULLABLE,
          OperandTypes.DATETIME
              .or(OperandTypes.CHARACTER)
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER)),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIME = TO_TIME.withName("TIME");

  public static final SqlFunction TRY_TO_TIME =
      new SqlFunction(
          "TRY_TO_TIME",
          // What SqlKind should match?
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          BodoReturnTypes.TIME_FORCE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.DATETIME
              .or(OperandTypes.CHARACTER)
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlBasicFunction TIMEFROMPARTS =
      SqlBasicFunction.create(
          "TIMEFROMPARTS",
          ReturnTypes.TIME_NULLABLE,
          OperandTypes.sequence(
                  "TIMEFROMPARTS(HOUR, MINUTE, SECOND)",
                  OperandTypes.INTEGER,
                  OperandTypes.INTEGER,
                  OperandTypes.INTEGER)
              .or(
                  OperandTypes.sequence(
                      "TIMEFROMPARTS(HOUR, MINUTE, SECOND, NANOSECOND)",
                      OperandTypes.INTEGER,
                      OperandTypes.INTEGER,
                      OperandTypes.INTEGER,
                      OperandTypes.INTEGER)),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIME_FROM_PARTS = TIMEFROMPARTS.withName("TIME_FROM_PARTS");

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
            BodoRelDataTypeFactory.createTZAwareSqlType(
                typeFactory, null, BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION);
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

  // Operand type checker for the overloade timestamp_from_parts functions. The
  // first overload has the signature:
  // timestamp_(ntz_)from_parts(year, month, day, hour, minute, second[,
  // nanosecond]),
  // while the second has the signature
  // timestamp_(ntz_)from_parts(date_expr, time_expr)
  public static final SqlOperandTypeChecker OVERLOADED_TIMESTAMP_FROM_PARTS_OPERAND_TYPE_CHECKER =
      argumentRange(
              6,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.STRING)
          .or(OperandTypes.family(SqlTypeFamily.DATE, SqlTypeFamily.TIME))
          .or(OperandTypes.family(SqlTypeFamily.DATE, SqlTypeFamily.TIMESTAMP))
          .or(OperandTypes.family(SqlTypeFamily.TIMESTAMP, SqlTypeFamily.TIMESTAMP))
          .or(OperandTypes.family(SqlTypeFamily.TIMESTAMP, SqlTypeFamily.TIME));

  public static final SqlBasicFunction TIMESTAMP_FROM_PARTS =
      SqlBasicFunction.create(
          "TIMESTAMP_FROM_PARTS",
          opBinding -> timestampConstructionOutputType(opBinding, true),
          OVERLOADED_TIMESTAMP_FROM_PARTS_OPERAND_TYPE_CHECKER,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMPFROMPARTS =
      TIMESTAMP_FROM_PARTS.withName("TIMESTAMPFROMPARTS");

  public static final SqlBasicFunction TIMESTAMP_NTZ_FROM_PARTS =
      SqlBasicFunction.create(
          "TIMESTAMP_NTZ_FROM_PARTS",
          opBinding -> timestampConstructionOutputType(opBinding, true),
          OVERLOADED_TIMESTAMP_FROM_PARTS_OPERAND_TYPE_CHECKER,
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMPNTZFROMPARTS =
      TIMESTAMP_NTZ_FROM_PARTS.withName("TIMESTAMPNTZFROMPARTS");

  public static final SqlBasicFunction TIMESTAMP_LTZ_FROM_PARTS =
      SqlBasicFunction.create(
          "TIMESTAMP_LTZ_FROM_PARTS",
          opBinding -> timestampConstructionOutputType(opBinding, false),
          argumentRange(
              6,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER,
              SqlTypeFamily.INTEGER),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMPLTZFROMPARTS =
      TIMESTAMP_LTZ_FROM_PARTS.withName("TIMESTAMPLTZFROMPARTS");

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

  public static final SqlBasicFunction DATE_SUB =
      SqlBasicFunction.create(
          "DATE_SUB",
          opBinding -> mySqlDateaddReturnType(opBinding),
          OperandTypes.DATETIME_INTERVAL
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.DATETIME_INTERVAL))
              .or(OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.INTEGER))
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.INTEGER)),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction SUBDATE = DATE_SUB.withName("SUBDATE");
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
          OperandTypes.DATETIME_INTERVAL
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.DATETIME_INTERVAL))
              .or(OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.INTEGER))
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.INTEGER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATEDIFF =
      new SqlFunction(
          "DATEDIFF",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts only
          // (Datetime, Datetime)

          OperandTypes.sequence(
                  "DATEDIFF(TIMESTAMP/DATE, TIMESTAMP/DATE)",
                  OperandTypes.DATETIME,
                  OperandTypes.DATETIME)
              .or(
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
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.sequence(
              "TIMEDIFF(UNIT, TIMESTAMP/DATE/TIME, TIMESTAMP/DATE/TIME)",
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
          // returns null if the string doesn't match the provided format
          BodoReturnTypes.TIMESTAMP_FORCE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts only
          // (String, Literal String)
          OperandTypes.sequence(
              "STR_TO_DATE(STRING, LITERAL)", OperandTypes.STRING, OperandTypes.LITERAL),
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

  // NOTE: This is unique to Bodo
  public static final SqlFunction NANOSECOND =
      new SqlDatePartFunction("NANOSECOND", TimeUnit.NANOSECOND);

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

  public static final SqlBasicFunction MONTHNAME =
      SqlBasicFunction.create(
          "MONTHNAME",
          // MONTHNAME always return a 3 character month abbreviation
          BodoReturnTypes.VARCHAR_3_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction MONTH_NAME = MONTHNAME.withName("MONTH_NAME");
  public static final SqlFunction MONTHS_BETWEEN =
      new SqlFunction(
          "MONTHS_BETWEEN",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          /// What Input Types does the function accept.
          OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.DATETIME),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction ADD_MONTHS =
      new SqlFunction(
          "ADD_MONTHS",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.ARG0_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.NUMERIC),
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
                  "DATE_FORMAT(DATE, STRING_LITERAL)", OperandTypes.DATE, OperandTypes.LITERAL)
              .or(
                  OperandTypes.sequence(
                      "DATE_FORMAT(TIMESTAMP, STRING_LITERAL)",
                      OperandTypes.TIMESTAMP,
                      OperandTypes.LITERAL)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction MAKEDATE =
      new SqlFunction(
          "MAKEDATE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DATE_NULLABLE,
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
          ReturnTypes.ARG0_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.sequence(
                  "TIME_SLICE(DATETIME, INT, UNIT)",
                  OperandTypes.DATETIME,
                  OperandTypes.INTEGER,
                  OperandTypes.ANY)
              .or(
                  OperandTypes.sequence(
                      "TIME_SLICE(DATETIME, INT, UNIT, STRING)",
                      OperandTypes.DATETIME,
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
          OperandTypes.sequence("TRUNC(UNIT, DATETIME)", OperandTypes.ANY, OperandTypes.DATETIME)
              .or(argumentRange(1, SqlTypeFamily.NUMERIC, SqlTypeFamily.INTEGER)),
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
          OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.CHARACTER)
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER)),
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
          OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.CHARACTER)
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DAY = new SqlDatePartFunction("DAY", TimeUnit.DAY);

  public static final SqlFunction CONVERT_TIMEZONE =
      new SqlFunction(
          "CONVERT_TIMEZONE",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.TIMESTAMP_NULLABLE,
          null,
          OperandTypes.CHARACTER_CHARACTER_DATETIME.or(
              OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.DATETIME)),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction EPOCH_SECOND =
      new SqlFunction(
          "EPOCH_SECOND",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction EPOCH_MILLISECOND =
      new SqlFunction(
          "EPOCH_MILLISECOND",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction EPOCH_MICROSECOND =
      new SqlFunction(
          "EPOCH_MICROSECOND",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction EPOCH_NANOSECOND =
      new SqlFunction(
          "EPOCH_NANOSECOND",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMEZONE_HOUR =
      new SqlFunction(
          "TIMEZONE_HOUR",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // Note: This is the max type the SF return precision.
          // It seems like a tinyint should be possible.
          BodoReturnTypes.SMALLINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMEZONE_MINUTE =
      new SqlFunction(
          "TIMEZONE_MINUTE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // Note: This is the max type the SF return precision.
          // It seems like a tinyint should be possible.
          BodoReturnTypes.SMALLINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  private List<SqlOperator> functionList =
      Arrays.asList(
          CONVERT_TIMEZONE,
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
          NANOSECOND,
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
          DAY,
          EPOCH_SECOND,
          EPOCH_MILLISECOND,
          EPOCH_MICROSECOND,
          EPOCH_NANOSECOND,
          TIMEZONE_HOUR,
          TIMEZONE_MINUTE);

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
      // all of these functions are user defined functions.
      operatorList.add(func);
    }
  }

  @Override
  public List<SqlOperator> getOperatorList() {
    return functionList;
  }
}
