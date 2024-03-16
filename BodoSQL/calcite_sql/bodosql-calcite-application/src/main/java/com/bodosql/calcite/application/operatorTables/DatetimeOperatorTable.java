package com.bodosql.calcite.application.operatorTables;

import static com.bodosql.calcite.application.operatorTables.OperatorTableUtils.argumentRange;
import static com.bodosql.calcite.application.operatorTables.OperatorTableUtils.isOutputNullableCompile;
import static org.apache.calcite.sql.type.BodoReturnTypes.CONVERT_TIMEZONE_RETURN_TYPE;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import java.util.*;
import org.apache.calcite.avatica.util.TimeUnit;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlBasicFunction;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.fun.SqlDatePartFunction;
import org.apache.calcite.sql.type.BodoOperandTypes;
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeTransforms;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.checkerframework.checker.nullness.qual.Nullable;

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
      SqlBasicFunction.create(
          "DATEADD",
          BodoReturnTypes.dateAddReturnType("DATEADD"),
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
      SqlBasicFunction.create(
          "TIMEADD",
          BodoReturnTypes.dateAddReturnType("TIMEADD"),
          OperandTypes.sequence(
              "TIMEADD(UNIT, VALUE, TIME)",
              OperandTypes.ANY,
              OperandTypes.NUMERIC,
              OperandTypes.DATETIME),
          SqlFunctionCategory.TIMEDATE);

  // TODO: Extend the Library Operator and use the builtin Libraries
  public static final SqlFunction DATE_ADD =
      SqlBasicFunction.create(
          "DATE_ADD",
          BodoReturnTypes.dateAddReturnType("DATE_ADD"),
          OperandTypes.DATETIME_INTERVAL
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.DATETIME_INTERVAL))
              .or(OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.INTEGER))
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.INTEGER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlBasicFunction TIMEFROMPARTS =
      SqlBasicFunction.create(
          "TIMEFROMPARTS",
          BodoReturnTypes.TIME_DEFAULT_PRECISION_NULLABLE,
          argumentRange(
              3,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIME_FROM_PARTS = TIMEFROMPARTS.withName("TIME_FROM_PARTS");

  /**
   * Generate the return type for TIMESTAMP_FROM_PARTS, TIMESTAMP_NTZ_FROM_PARTS,
   * TIMESTAMP_LTZ_FROM_PARTS and TIMESTAMP_TZ_FROM_PARTS.
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
        returnType = typeFactory.createSqlType(SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE);
      }
    } else {
      throw new BodoSQLCodegenException(
          "TIMESTAMP_FROM_PARTS_* with timezone argument not supported yet");
    }

    return typeFactory.createTypeWithNullability(returnType, nullable);
  }

  public static final SqlBasicFunction DATE_FROM_PARTS =
      SqlBasicFunction.create(
          "DATE_FROM_PARTS",
          ReturnTypes.DATE_NULLABLE,
          OperandTypes.sequence(
              "DATE_FROM_PARTS(HOUR, MINUTE, SECOND)",
              OperandTypes.NUMERIC,
              OperandTypes.NUMERIC,
              OperandTypes.NUMERIC),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATEFROMPARTS = DATE_FROM_PARTS.withName("DATEFROMPARTS");

  // Operand type checker for the overloade timestamp_from_parts functions. The
  // first overload has the signature:
  // timestamp_(ntz_)from_parts(year, month, day, hour, minute, second[,
  // nanosecond]),
  // while the second has the signature
  // timestamp_(ntz_)from_parts(date_expr, time_expr)
  public static final SqlOperandTypeChecker OVERLOADED_TIMESTAMP_FROM_PARTS_OPERAND_TYPE_CHECKER =
      argumentRange(
              6,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
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
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMPLTZFROMPARTS =
      TIMESTAMP_LTZ_FROM_PARTS.withName("TIMESTAMPLTZFROMPARTS");

  public static final SqlBasicFunction TIMESTAMP_TZ_FROM_PARTS =
      SqlBasicFunction.create(
          "TIMESTAMP_TZ_FROM_PARTS",
          ReturnTypes.TIMESTAMP_TZ.andThen(SqlTypeTransforms.TO_NULLABLE),
          BodoOperandTypes.TIMESTAMP_FROM_PARTS_BASE_CHECKER.or(
              BodoOperandTypes.TIMESTAMP_FROM_PARTS_TZ_CHECKER),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMESTAMPTZFROMPARTS =
      TIMESTAMP_TZ_FROM_PARTS.withName("TIMESTAMPTZFROMPARTS");

  public static final SqlBasicFunction DATE_SUB =
      SqlBasicFunction.create(
          "DATE_SUB",
          BodoReturnTypes.dateAddReturnType("DATE_SUB"),
          OperandTypes.DATETIME_INTERVAL
              .or(OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.DATETIME_INTERVAL))
              .or(OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.INTEGER)),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction SUBDATE = DATE_SUB.withName("SUBDATE");
  public static final SqlFunction ADDDATE =
      SqlBasicFunction.create(
          "ADDDATE",
          BodoReturnTypes.dateAddReturnType("ADDDATE"),
          // What Input Types does the function accept. This function accepts either
          // (Datetime, Interval) or (Datetime, Integer)
          OperandTypes.DATETIME_INTERVAL
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.DATETIME_INTERVAL))
              .or(OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.INTEGER))
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.INTEGER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATEDIFF =
      SqlBasicFunction.create(
          "DATEDIFF",
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
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
      SqlBasicFunction.create(
          "TIMEDIFF",
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          OperandTypes.sequence(
              "TIMEDIFF(UNIT, TIMESTAMP/DATE/TIME, TIMESTAMP/DATE/TIME)",
              OperandTypes.ANY,
              OperandTypes.DATETIME,
              OperandTypes.DATETIME),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction STR_TO_DATE =
      SqlBasicFunction.create(
          "STR_TO_DATE",
          // What Value should the return type be
          // returns null if the string doesn't match the provided format
          BodoReturnTypes.TIMESTAMP_FORCE_NULLABLE,
          // What Input Types does the function accept. This function accepts only
          // (String, Literal String)
          OperandTypes.sequence(
              "STR_TO_DATE(STRING, LITERAL)", OperandTypes.STRING, OperandTypes.LITERAL),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlBasicFunction GETDATE =
      SqlBasicFunction.create(
          "GETDATE",
          // What Value should the return type be
          ReturnTypes.TIMESTAMP_LTZ,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  // CURRENT_TIMESTAMP is already supported inside Calcite and will be picked
  // up automatically. No need to implement it again here

  // LOCALTIMESTAMP is already supported inside Calcite and will be picked
  // up automatically. No need to implement it again here

  public static final SqlFunction SYSTIMESTAMP = GETDATE.withName("SYSTIMESTAMP");

  public static final SqlFunction NOW = GETDATE.withName("NOW");

  // CURRENT_TIME is already supported inside Calcite and will be picked
  // up automatically. No need to implement it again here

  // LOCALTIME is already supported inside Calcite and will be picked
  // up automatically. No need to implement it again here

  public static final SqlBasicFunction UTC_TIMESTAMP =
      SqlBasicFunction.create(
          "UTC_TIMESTAMP",
          // What Value should the return type be
          ReturnTypes.TIMESTAMP,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction UTC_DATE = UTC_TIMESTAMP.withName("UTC_DATE");

  public static final SqlFunction SYSDATE = UTC_TIMESTAMP.withName("SYSDATE");

  public static final SqlFunction MICROSECOND =
      new SqlDatePartFunction("MICROSECOND", TimeUnit.MICROSECOND);

  // NOTE: This is unique to Bodo
  public static final SqlFunction NANOSECOND =
      new SqlDatePartFunction("NANOSECOND", TimeUnit.NANOSECOND);

  public static final SqlFunction WEEKOFYEAR = new SqlDatePartFunction("WEEKOFYEAR", TimeUnit.WEEK);

  public static final SqlFunction WEEKISO = new SqlDatePartFunction("WEEKISO", TimeUnit.WEEK);

  public static final SqlFunction DAYNAME =
      SqlBasicFunction.create(
          "DAYNAME",
          // This always returns a 3 letter value.
          BodoReturnTypes.VARCHAR_3_NULLABLE,
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
      SqlBasicFunction.create(
          "MONTHS_BETWEEN",
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          /// What Input Types does the function accept.
          OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.DATETIME),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction ADD_MONTHS =
      SqlBasicFunction.create(
          "ADD_MONTHS",
          // What Value should the return type be
          ReturnTypes.ARG0_NULLABLE,
          OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction CURDATE =
      SqlBasicFunction.create(
          "CURDATE",
          // What Value should the return type be
          ReturnTypes.DATE,
          // What Input Types does the function accept.
          OperandTypes.NILADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATE_FORMAT =
      SqlBasicFunction.create(
          "DATE_FORMAT",
          // Precision cannot be statically determined.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
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
      SqlBasicFunction.create(
          "MAKEDATE",
          // What Value should the return type be
          ReturnTypes.DATE_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.sequence(
              "MAKEDATE(INTEGER, INTEGER)", OperandTypes.INTEGER, OperandTypes.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction WEEKDAY =
      SqlBasicFunction.create(
          "WEEKDAY",
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction YEARWEEK =
      SqlBasicFunction.create(
          "YEARWEEK",
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.TIMESTAMP,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATE_TRUNC =
      SqlNullPolicyFunction.createAnyPolicy(
          "DATE_TRUNC",
          // What Value should the return type be
          opBinding -> datetruncReturnType(opBinding),
          // What Input Types does the function accept.
          OperandTypes.sequence(
              "DATE_TRUNC(UNIT, DATETIME)", OperandTypes.ANY, OperandTypes.DATETIME),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIME_SLICE =
      SqlBasicFunction.create(
          "TIME_SLICE",
          // What Value should the return type be
          ReturnTypes.ARG0_NULLABLE,
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
      SqlBasicFunction.create(
          "TRUNC",
          opBinding -> truncReturnType(opBinding),
          // What Input Types does the function accept.
          OperandTypes.sequence("TRUNC(UNIT, DATETIME)", OperandTypes.ANY, OperandTypes.DATETIME)
              .or(argumentRange(1, SqlTypeFamily.NUMERIC, SqlTypeFamily.INTEGER)),
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction YEAROFWEEK =
      SqlBasicFunction.create(
          "YEAROFWEEK",
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);
  public static final SqlFunction YEAROFWEEKISO =
      SqlBasicFunction.create(
          "YEAROFWEEKISO",
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction NEXT_DAY =
      SqlBasicFunction.create(
          "NEXT_DAY",
          // What Value should the return type be
          ReturnTypes.DATE_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.CHARACTER)
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction PREVIOUS_DAY =
      SqlBasicFunction.create(
          "PREVIOUS_DAY",
          // What Value should the return type be
          ReturnTypes.DATE_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.family(SqlTypeFamily.DATETIME, SqlTypeFamily.CHARACTER)
              .or(OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.CHARACTER)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DAY = new SqlDatePartFunction("DAY", TimeUnit.DAY);

  public static final SqlFunction CONVERT_TIMEZONE =
      SqlBasicFunction.create(
          "CONVERT_TIMEZONE",
          CONVERT_TIMEZONE_RETURN_TYPE.andThen(SqlTypeTransforms.TO_NULLABLE),
          OperandTypes.CHARACTER_CHARACTER_DATETIME.or(
              OperandTypes.family(SqlTypeFamily.CHARACTER, SqlTypeFamily.DATETIME)),
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction EPOCH_SECOND =
      SqlBasicFunction.create(
          "EPOCH_SECOND",
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction EPOCH_MILLISECOND =
      SqlBasicFunction.create(
          "EPOCH_MILLISECOND",
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction EPOCH_MICROSECOND =
      SqlBasicFunction.create(
          "EPOCH_MICROSECOND",
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction EPOCH_NANOSECOND =
      SqlBasicFunction.create(
          "EPOCH_NANOSECOND",
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMEZONE_HOUR =
      SqlBasicFunction.create(
          "TIMEZONE_HOUR",
          // Note: This is the max type the SF return precision.
          // It seems like a tinyint should be possible.
          BodoReturnTypes.SMALLINT_NULLABLE,
          // What Input Types does the function accept.
          OperandTypes.DATETIME,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction TIMEZONE_MINUTE =
      SqlBasicFunction.create(
          "TIMEZONE_MINUTE",
          // Note: This is the max type the SF return precision.
          // It seems like a tinyint should be possible.
          BodoReturnTypes.SMALLINT_NULLABLE,
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
          TIMEFROMPARTS,
          TIME_FROM_PARTS,
          TIME_SLICE,
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
