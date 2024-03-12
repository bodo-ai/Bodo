/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.bodosql.calcite.sql.func;

import static java.util.Objects.requireNonNull;
import static org.apache.calcite.sql.validate.SqlNonNullableAccessors.getOperandLiteralValueOrThrow;
import static org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE;

import com.bodosql.calcite.application.operatorTables.DatetimeFnUtils;
import java.util.Locale;
import org.apache.calcite.avatica.util.TimeUnit;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.SqlReturnTypeInference;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;

/**
 * The <code>TIMESTAMPADD</code> function, which adds an interval to a datetime (TIMESTAMP, TIME or
 * DATE).
 *
 * <p>The SQL syntax is
 *
 * <blockquote>
 *
 * <code>TIMESTAMPADD(<i>timestamp interval</i>, <i>quantity</i>,
 * <i>datetime</i>)</code>
 *
 * </blockquote>
 *
 * Accepts both quoted and unquoted time units now.
 *
 * <p>Returns modified datetime.
 */
public class BodoSqlTimestampAddFunction extends SqlFunction {

  private static final SqlReturnTypeInference RETURN_TYPE_INFERENCE =
      opBinding -> {
        final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
        RelDataType arg0Type = opBinding.getOperandType(0);

        DatetimeFnUtils.DateTimePart arg0timeUnit =
            getOperandLiteralValueOrThrow(opBinding, 0, DatetimeFnUtils.DateTimePart.class);
        RelDataType ret;
        try {
          ret =
              deduceType(
                  typeFactory,
                  arg0timeUnit,
                  opBinding.getOperandType(1),
                  opBinding.getOperandType(2));
        } catch (RuntimeException e) {
          String errMsg = requireNonNull(e.getMessage());
          // This can be called in the convertlet. If so we won't have a SqlCallBinding,
          // but also validation shouldn't fail.
          if (opBinding instanceof SqlCallBinding) {
            SqlCallBinding opBindingWithCast = (SqlCallBinding) opBinding;
            throw opBindingWithCast
                .getValidator()
                .newValidationError(
                    opBindingWithCast.getCall(),
                    BODO_SQL_RESOURCE.wrongTimeUnit("TIMESTAMPADD", errMsg));
          } else {
            throw e;
          }
        }
        return ret;
      };

  /**
   * Helper function that verifies and standardizes the time unit input.
   *
   * @param fnName the function which takes this time unit as input
   * @param inputTimeStr the input time unit string
   * @param isTime if this time unit should fit with Bodo.Time, which means smaller or equal to hour
   * @return the standardized time unit string
   */
  public static TimeUnit standardizeTimeUnit(String fnName, String inputTimeStr, boolean isTime) {
    TimeUnit unit;
    switch (inputTimeStr.toLowerCase(Locale.ROOT)) {
      case "\"year\"":
      case "\"y\"":
      case "\"yy\"":
      case "\"yyy\"":
      case "\"yyyy\"":
      case "\"yr\"":
      case "\"years\"":
      case "\"yrs\"":
      case "year":
      case "y":
      case "yy":
      case "yyy":
      case "yyyy":
      case "yr":
      case "years":
      case "yrs":
      case "sql_tsi_year":
        if (isTime) {
          throw new RuntimeException(
              "Unsupported " + fnName + " unit for TIME input: " + inputTimeStr);
        }
        unit = TimeUnit.YEAR;
        break;

      case "\"month\"":
      case "\"mm\"":
      case "\"mon\"":
      case "\"mons\"":
      case "\"months\"":
      case "month":
      case "mm":
      case "mon":
      case "mons":
      case "months":
      case "sql_tsi_month":
        if (isTime) {
          throw new RuntimeException(
              "Unsupported " + fnName + " unit for TIME input: " + inputTimeStr);
        }
        unit = TimeUnit.MONTH;
        break;

      case "\"day\"":
      case "\"d\"":
      case "\"dd\"":
      case "\"days\"":
      case "\"dayofmonth\"":
      case "day":
      case "d":
      case "dd":
      case "days":
      case "dayofmonth":
      case "sql_tsi_day":
        if (isTime) {
          throw new RuntimeException(
              "Unsupported " + fnName + " unit for TIME input: " + inputTimeStr);
        }
        unit = TimeUnit.DAY;
        break;

      case "\"week\"":
      case "\"w\"":
      case "\"wk\"":
      case "\"weekofyear\"":
      case "\"woy\"":
      case "\"wy\"":
      case "week":
      case "w":
      case "wk":
      case "weekofyear":
      case "woy":
      case "wy":
      case "sql_tsi_week":
        if (isTime) {
          throw new RuntimeException(
              "Unsupported " + fnName + " unit for TIME input: " + inputTimeStr);
        }
        unit = TimeUnit.WEEK;
        break;

      case "\"quarter\"":
      case "\"q\"":
      case "\"qtr\"":
      case "\"qtrs\"":
      case "\"quarters\"":
      case "quarter":
      case "q":
      case "qtr":
      case "qtrs":
      case "quarters":
      case "sql_tsi_quarter":
        if (isTime) {
          throw new RuntimeException(
              "Unsupported " + fnName + " unit for TIME input: " + inputTimeStr);
        }
        unit = TimeUnit.QUARTER;
        break;

      case "\"hour\"":
      case "\"h\"":
      case "\"hh\"":
      case "\"hr\"":
      case "\"hours\"":
      case "\"hrs\"":
      case "hour":
      case "h":
      case "hh":
      case "hr":
      case "hours":
      case "hrs":
      case "sql_tsi_hour":
        unit = TimeUnit.HOUR;
        break;

      case "\"minute\"":
      case "\"m\"":
      case "\"mi\"":
      case "\"min\"":
      case "\"minutes\"":
      case "\"mins\"":
      case "minute":
      case "m":
      case "mi":
      case "min":
      case "minutes":
      case "mins":
      case "sql_tsi_minute":
        unit = TimeUnit.MINUTE;
        break;

      case "\"second\"":
      case "\"s\"":
      case "\"sec\"":
      case "\"seconds\"":
      case "\"secs\"":
      case "second":
      case "s":
      case "sec":
      case "seconds":
      case "secs":
      case "sql_tsi_second":
        unit = TimeUnit.SECOND;
        break;

      case "\"millisecond\"":
      case "\"ms\"":
      case "\"msec\"":
      case "\"milliseconds\"":
      case "millisecond":
      case "ms":
      case "msec":
      case "milliseconds":
        unit = TimeUnit.MILLISECOND;
        break;

      case "\"microsecond\"":
      case "\"us\"":
      case "\"usec\"":
      case "\"microseconds\"":
      case "microsecond":
      case "us":
      case "usec":
      case "microseconds":
      case "frac_second":
      case "sql_tsi_microsecond":
        unit = TimeUnit.MICROSECOND;
        break;

      case "\"nanosecond\"":
      case "\"ns\"":
      case "\"nsec\"":
      case "\"nanosec\"":
      case "\"nsecond\"":
      case "\"nanoseconds\"":
      case "\"nanosecs\"":
      case "\"nseconds\"":
      case "nanosecond":
      case "ns":
      case "nsec":
      case "nanosec":
      case "nsecond":
      case "nanoseconds":
      case "nanosecs":
      case "nseconds":
      case "sql_tsi_frac_second":
        unit = TimeUnit.NANOSECOND;
        break;

      default:
        throw new RuntimeException("Unsupported " + fnName + " unit: " + inputTimeStr);
    }
    return unit;
  }

  public static RelDataType deduceType(
      RelDataTypeFactory typeFactory,
      DatetimeFnUtils.DateTimePart timeUnit,
      RelDataType operandType1,
      RelDataType operandType2) {

    // https://docs.snowflake.com/en/sql-reference/functions/timestampadd
    // Based on my reading of this:

    // Given a Date input, we get a Date output if the unit value added is larger than a day,
    // and Timestamp otherwise

    // Given a time input, we get a Time output. if the unit value added is larger than a day, we
    // error

    // Given a timestamp input, we get a timestamp output always (Precision determined by arg2
    // type and timeunit size)

    boolean timeUnitSmallerThanDay;

    switch (timeUnit) {
      case HOUR:
      case MINUTE:
      case SECOND:
      case MILLISECOND:
      case MICROSECOND:
      case NANOSECOND:
        timeUnitSmallerThanDay = true;
        break;
      default:
        timeUnitSmallerThanDay = false;
    }

    final RelDataType outputType;
    switch (operandType2.getSqlTypeName()) {
      case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
      case TIMESTAMP_TZ:
      case TIMESTAMP:
        outputType = operandType2;
        break;
      case TIME_WITH_LOCAL_TIME_ZONE:
      case TIME:
        if (!timeUnitSmallerThanDay) {
          throw new RuntimeException(
              "When arg2 is a time," + " the specified time unit must be smaller than day.");
        }
        outputType = operandType2;
        break;
      case DATE:
        if (timeUnitSmallerThanDay) {
          outputType = typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
        } else {
          outputType = operandType2;
        }
        break;
      default:
        // We already checked input type of arg2, this should never be thrown
        throw new RuntimeException("Input type of arg2 must be DATE, TIME or TIMESTAMP.");
    }

    return typeFactory.createTypeWithNullability(
        outputType, operandType1.isNullable() || operandType2.isNullable());
  }

  /** Creates a SqlTimestampAddFunction. */
  BodoSqlTimestampAddFunction() {
    super(
        "TIMESTAMPADD",
        SqlKind.TIMESTAMP_ADD,
        RETURN_TYPE_INFERENCE,
        null,
        OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.NUMERIC, SqlTypeFamily.DATETIME),
        SqlFunctionCategory.TIMEDATE);
  }
}
