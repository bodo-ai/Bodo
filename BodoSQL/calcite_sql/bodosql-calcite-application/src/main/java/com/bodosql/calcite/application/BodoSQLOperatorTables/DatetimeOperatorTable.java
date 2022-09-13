package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.avatica.util.TimeUnit;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.fun.SqlDatePartFunction;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.validate.SqlNameMatcher;

public final class DatetimeOperatorTable implements SqlOperatorTable {

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
          ReturnTypes.TIMESTAMP_NULLABLE,
          null,
          OperandTypes.sequence(
              "DATEADD(DATETIME_OR_DATETIME_STRING, INTERVAL_OR_INTEGER)",
              OperandTypes.or(OperandTypes.DATETIME, OperandTypes.STRING),
              OperandTypes.or(OperandTypes.INTERVAL, OperandTypes.INTEGER)),
          SqlFunctionCategory.TIMEDATE);

  // TODO: Extend the Library Operator and use the builtin Libraries
  public static final SqlFunction DATE_ADD =
      new SqlFunction(
          "DATE_ADD",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP_NULLABLE,
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

  // TODO: make this actually coerce to cast, so that Calcite can properly typecheck it at compile
  // time as of right now, we do this check in our code.
  public static final SqlFunction TO_DATE =
      new SqlFunction(
          "TO_DATE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          OperandTypes.ANY,
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  public static final SqlFunction DATE_SUB =
      new SqlFunction(
          "DATE_SUB",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP_NULLABLE,
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
          ReturnTypes.TIMESTAMP_NULLABLE,
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
          ReturnTypes.TIMESTAMP_NULLABLE,
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
          OperandTypes.sequence(
              "DATEDIFF(TIMESTAMP, TIMESTAMP)", OperandTypes.TIMESTAMP, OperandTypes.TIMESTAMP),
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

  public static final SqlFunction NOW =
      new SqlFunction(
          "NOW",
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

  public static final SqlFunction CURDATE =
      new SqlFunction(
          "CURDATE",
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

  public static final SqlFunction CURRENT_DATE =
      new SqlFunction(
          "CURRENT_DATE",
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
          ReturnTypes.TIMESTAMP_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.sequence(
              "DATE_TRUNC(STRING_LITERAL, TIMESTAMP)",
              OperandTypes.LITERAL,
              OperandTypes.TIMESTAMP),
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

  public static final SqlFunction PREVIOUS_DAY =
      new SqlFunction(
          "PREVIOUS_DAY",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.TIMESTAMP_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          OperandTypes.sequence(
              "PREVIOUS_DAY(DATETIME_OR_DATETIME_STRING, STRING_LITERAL)",
              OperandTypes.or(OperandTypes.DATETIME, OperandTypes.STRING),
              OperandTypes.or(OperandTypes.STRING, OperandTypes.LITERAL)),
          // What group of functions does this fall into?
          SqlFunctionCategory.TIMEDATE);

  private List<SqlOperator> functionList =
      Arrays.asList(
          DATEADD,
          DATE_ADD,
          DATE_SUB,
          DATEDIFF,
          STR_TO_DATE,
          LOCALTIMESTAMP,
          NOW,
          UTC_TIMESTAMP,
          UTC_DATE,
          DAYNAME,
          MONTHNAME,
          MICROSECOND,
          WEEKOFYEAR,
          WEEKISO,
          CURDATE,
          CURRENT_DATE,
          DATE_FORMAT,
          MAKEDATE,
          ADDDATE,
          SUBDATE,
          YEARWEEK,
          WEEKDAY,
          TO_DATE,
          DATE_TRUNC,
          YEAROFWEEKISO,
          PREVIOUS_DAY);

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
