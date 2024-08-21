package com.bodosql.calcite.application.operatorTables;

import static org.apache.calcite.sql.type.BodoReturnTypes.ArrayAggReturnType;
import static org.apache.calcite.sql.type.BodoReturnTypes.BOOL_AGG_RET_TYPE;
import static org.apache.calcite.sql.type.BodoReturnTypes.bitX_ret_type;

import java.util.Arrays;
import java.util.List;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlRankFunction;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.fun.SqlBasicAggFunction;
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeTransforms;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.apache.calcite.util.Optionality;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Operator table used for aggregate functions. This needs access to package private methods. */
public class AggOperatorTable implements SqlOperatorTable {
  private static @Nullable AggOperatorTable instance;

  /** Returns the Aggregation operator table, creating it if necessary. */
  public static synchronized AggOperatorTable instance() {
    AggOperatorTable instance = AggOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new AggOperatorTable();
      AggOperatorTable.instance = instance;
    }
    return instance;
  }

  public static final SqlAggFunction ARRAY_AGG =
      SqlBasicAggFunction.create(
              "ARRAY_AGG",
              SqlKind.ARRAY_AGG,
              opBinding -> ArrayAggReturnType(opBinding),
              OperandTypes.ANY)
          .withGroupOrder(Optionality.OPTIONAL)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  public static final SqlAggFunction ARRAY_UNIQUE_AGG =
      SqlBasicAggFunction.create(
              "ARRAY_UNIQUE_AGG", SqlKind.OTHER_FUNCTION, ReturnTypes.TO_ARRAY, OperandTypes.ANY)
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  //  Returns the logical (boolean) OR value of all non-NULL boolean records in a group.
  //  BOOLOR_AGG returns ‘true’ if at least one record in the group evaluates to ‘true’.
  //  Numeric values, Decimal, and floating point values are converted to ‘true’ if they are
  // different from zero.
  //  Character/text types are not supported as they cannot be converted to Boolean.
  public static final SqlAggFunction BOOLOR_AGG =
      SqlBasicAggFunction.create(
              "BOOLOR_AGG",
              SqlKind.OTHER,
              // Spec (https://docs.snowflake.com/en/sql-reference/functions/boolor_agg) says
              // ...if the group is empty, the function returns NULL.
              BOOL_AGG_RET_TYPE,
              OperandTypes.BOOLEAN.or(OperandTypes.NUMERIC))
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  // The same as BOOLAND_AGG except that it returns true if all of the inputs are true
  public static final SqlAggFunction BOOLAND_AGG =
      SqlBasicAggFunction.create(
              "BOOLAND_AGG",
              SqlKind.OTHER,
              // see BOOLOR_AGG for nullability note
              BOOL_AGG_RET_TYPE,
              OperandTypes.BOOLEAN.or(OperandTypes.NUMERIC))
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  // The same as BOOLAND_AGG except that it returns true if exactly one of the inputs  is are true
  public static final SqlAggFunction BOOLXOR_AGG =
      SqlBasicAggFunction.create(
              "BOOLXOR_AGG",
              SqlKind.OTHER,
              // see BOOLXOR_AGG for nullability note
              BOOL_AGG_RET_TYPE,
              OperandTypes.BOOLEAN.or(OperandTypes.NUMERIC))
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  public static final SqlBasicAggFunction CONDITIONAL_TRUE_EVENT =
      SqlBasicAggFunction.create(
          "CONDITIONAL_TRUE_EVENT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.INTEGER,
          OperandTypes.BOOLEAN);

  public static final SqlBasicAggFunction CONDITIONAL_CHANGE_EVENT =
      SqlBasicAggFunction.create(
          "CONDITIONAL_CHANGE_EVENT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.INTEGER,
          OperandTypes.ANY);

  public static final SqlBasicAggFunction COUNT_IF =
      SqlBasicAggFunction.create(
          "COUNT_IF", SqlKind.OTHER_FUNCTION, ReturnTypes.INTEGER, OperandTypes.BOOLEAN);

  // MIN_ROW_NUMBER_FILTER is an internal function created as an optimization
  // on ROW_NUMBER = 1
  public static final SqlRankFunction MIN_ROW_NUMBER_FILTER =
      // NOTE: There's a ReturnTypes.BOOLEAN_NOT_NULL and a ReturnTypes.BOOLEAN.
      // No other type has an equivalent _NOT_NULL return type.
      // Through reading of the code, we've confirmed that both are functionally equivalent.
      // I'm going to use ReturnTypes.BOOLEAN_NOT_NULL for maximum safety, and better readability,
      // but either would be fine here.
      new SqlRankFunction(SqlKind.MIN_ROW_NUMBER_FILTER, ReturnTypes.BOOLEAN_NOT_NULL, true);

  public static final SqlBasicAggFunction VARIANCE_POP =
      SqlBasicAggFunction.create(
          "VARIANCE_POP",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          OperandTypes.NUMERIC);

  public static final SqlBasicAggFunction VARIANCE_SAMP =
      SqlBasicAggFunction.create(
          "VARIANCE_SAMP",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          OperandTypes.NUMERIC);

  public static final SqlBasicAggFunction CORR =
      SqlBasicAggFunction.create(
              "CORR",
              SqlKind.CORR,
              ReturnTypes.DOUBLE.andThen(SqlTypeTransforms.FORCE_NULLABLE),
              OperandTypes.NUMERIC_NUMERIC)
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  public static final SqlAggFunction APPROX_PERCENTILE =
      SqlBasicAggFunction.create(
              "APPROX_PERCENTILE",
              SqlKind.OTHER_FUNCTION,
              ReturnTypes.DOUBLE_NULLABLE,
              OperandTypes.NUMERIC_NUMERIC)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  public static final SqlAggFunction MEDIAN =
      SqlBasicAggFunction.create(
              "MEDIAN", SqlKind.MEDIAN, ReturnTypes.ARG0_NULLABLE_IF_EMPTY, OperandTypes.NUMERIC)
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  public static final SqlAggFunction KURTOSIS =
      SqlBasicAggFunction.create(
              "KURTOSIS", SqlKind.OTHER_FUNCTION, ReturnTypes.DOUBLE_NULLABLE, OperandTypes.NUMERIC)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  public static final SqlAggFunction SKEW =
      SqlBasicAggFunction.create(
              "SKEW", SqlKind.OTHER_FUNCTION, ReturnTypes.DOUBLE_NULLABLE, OperandTypes.NUMERIC)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  public static final SqlAggFunction RATIO_TO_REPORT =
      SqlBasicAggFunction.create(
          "RATIO_TO_REPORT",
          SqlKind.OTHER_FUNCTION,
          // Can output null in the case that the sum within the group
          // evaluates to 0
          BodoReturnTypes.DOUBLE_FORCE_NULLABLE,
          OperandTypes.NUMERIC);

  public static final SqlAggFunction BITOR_AGG =
      SqlBasicAggFunction.create(
              "BITOR_AGG",
              SqlKind.BIT_OR,
              sqlOperatorBinding -> bitX_ret_type(sqlOperatorBinding),
              OperandTypes.EXACT_NUMERIC)
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  public static final SqlAggFunction BITAND_AGG =
      SqlBasicAggFunction.create(
              "BITAND_AGG",
              SqlKind.BIT_AND,
              sqlOperatorBinding -> bitX_ret_type(sqlOperatorBinding),
              OperandTypes.EXACT_NUMERIC)
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  public static final SqlAggFunction BITXOR_AGG =
      SqlBasicAggFunction.create(
              "BITXOR_AGG",
              SqlKind.BIT_XOR,
              sqlOperatorBinding -> bitX_ret_type(sqlOperatorBinding),
              OperandTypes.NUMERIC.or(OperandTypes.STRING))
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  public static final SqlAggFunction OBJECT_AGG =
      SqlBasicAggFunction.create(
              "OBJECT_AGG",
              SqlKind.OTHER_FUNCTION,
              ReturnTypes.ARG1.andThen(BodoReturnTypes.TO_MAP),
              OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.ANY))
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  private List<SqlOperator> aggOperatorList =
      Arrays.asList(
          ARRAY_AGG,
          ARRAY_UNIQUE_AGG,
          BOOLOR_AGG,
          BOOLAND_AGG,
          BOOLXOR_AGG,
          CONDITIONAL_TRUE_EVENT,
          CONDITIONAL_CHANGE_EVENT,
          COUNT_IF,
          MIN_ROW_NUMBER_FILTER,
          VARIANCE_POP,
          VARIANCE_SAMP,
          CORR,
          APPROX_PERCENTILE,
          MEDIAN,
          KURTOSIS,
          SKEW,
          RATIO_TO_REPORT,
          BITOR_AGG,
          BITAND_AGG,
          BITXOR_AGG,
          OBJECT_AGG);

  @Override
  public void lookupOperatorOverloads(
      SqlIdentifier opName,
      @Nullable SqlFunctionCategory category,
      SqlSyntax syntax,
      List<SqlOperator> operatorList,
      SqlNameMatcher nameMatcher) {
    // Heavily copied from Calcite:
    // https://github.com/apache/calcite/blob/4bc916619fd286b2c0cc4d5c653c96a68801d74e/core/src/main/java/org/apache/calcite/sql/util/ListSqlOperatorTable.java#L57
    for (SqlOperator operator : aggOperatorList) {
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
    return aggOperatorList;
  }
}
