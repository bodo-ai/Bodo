package com.bodosql.calcite.application.operatorTables;

import static org.apache.calcite.sql.type.BodoReturnTypes.BOOL_AGG_RET_TYPE;

import java.util.Arrays;
import java.util.List;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlFunction;
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
import org.apache.calcite.sql.type.SqlSingleOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeTransforms;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.apache.calcite.util.Optionality;
import org.checkerframework.checker.nullness.qual.Nullable;

public class CondOperatorTable implements SqlOperatorTable {
  private static @Nullable CondOperatorTable instance;

  // Type for a function with a boolean and then two matching types
  public static final SqlSingleOperandTypeChecker BOOLEAN_SAME_SAME =
      new SameOperandTypeExceptFirstOperandChecker(3, "BOOLEAN");

  // Type for a function with a boolean and then two matching types
  public static final SqlSingleOperandTypeChecker DECODE_VARIADIC = new DecodeOperandChecker();

  /** Returns the Datetime operator table, creating it if necessary. */
  public static synchronized CondOperatorTable instance() {
    CondOperatorTable instance = CondOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new CondOperatorTable();
      CondOperatorTable.instance = instance;
    }
    return instance;
  }

  public static final SqlFunction REGR_VALX =
      new SqlFunction(
          "REGR_VALX",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. It accepts two doubles
          OperandTypes.NUMERIC_NUMERIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction REGR_VALY =
      new SqlFunction(
          "REGR_VALY",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. It accepts two doubles
          OperandTypes.NUMERIC_NUMERIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  // TODO: Extend the Library Operator and use the builtin Libraries
  public static final SqlFunction IF_FUNC =
      new SqlFunction(
          "IF",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // The return type consists of the least restrictive of arguments 1 and 2
          // and if either is null.
          // null is considered false for this function, so we don't need to consider its
          // nullability
          // when determining output nullability
          BodoReturnTypes.leastRestrictiveSubsetNullable(1, 3),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts
          // a boolean arg0 and two matching args
          BOOLEAN_SAME_SAME,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  // TODO: Extend the Library Operator and use the builtin Libraries
  public static final SqlFunction IFF_FUNC =
      new SqlFunction(
          "IFF",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // The return type consists of the least restrictive of arguments 1 and 2
          // and if either is null.
          // null is considered false for this function, so we don't need to consider its
          // nullability
          // when determining output nullability
          BodoReturnTypes.leastRestrictiveSubsetNullable(1, 3),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts
          // a boolean arg0 and two matching args
          BOOLEAN_SAME_SAME,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction BOOLAND =
      new SqlFunction(
          "BOOLAND",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BOOLEAN_NULLABLE,
          null,
          OperandTypes.NUMERIC_NUMERIC,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction BOOLOR =
      new SqlFunction(
          "BOOLOR",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BOOLEAN_NULLABLE,
          null,
          OperandTypes.NUMERIC_NUMERIC,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction BOOLXOR =
      new SqlFunction(
          "BOOLXOR",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BOOLEAN_NULLABLE,
          null,
          OperandTypes.NUMERIC_NUMERIC,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction BOOLNOT =
      new SqlFunction(
          "BOOLNOT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BOOLEAN_NULLABLE,
          null,
          OperandTypes.NUMERIC,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction EQUAL_NULL =
      new SqlFunction(
          "EQUAL_NULL",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BOOLEAN,
          null,
          OperandTypes.SAME_SAME,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction IFNULL_FUNC =
      new SqlFunction(
          "IFNULL",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // LEAST_RESTRICTIVE will cast the return type to the least restrictive union of the
          // Two input types, and LEAST_NULLABLE will cast that type to a nullable type
          // If both of the two inputs are a nullable type.
          ReturnTypes.LEAST_RESTRICTIVE.andThen(SqlTypeTransforms.LEAST_NULLABLE),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts two
          // matching input types
          OperandTypes.SAME_SAME,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction NULLIFZERO =
      new SqlFunction(
          "NULLIFZERO",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // The output type is the same as the input type, but nullable
          ReturnTypes.ARG0_FORCE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts two
          // matching input types
          OperandTypes.NUMERIC,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction NVL =
      new SqlFunction(
          "NVL",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // LEAST_RESTRICTIVE will cast the return type to the least restrictive union of the
          // Two input types, and LEAST_NULLABLE will cast that type to a nullable type
          // If both of the two inputs are a nullable type.
          ReturnTypes.LEAST_RESTRICTIVE.andThen(SqlTypeTransforms.LEAST_NULLABLE),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts two
          // matching input types
          OperandTypes.SAME_SAME,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction NVL2 =
      new SqlFunction(
          "NVL2",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // The return type consists of the least restrictive of arguments 1 and 2
          // and if either is null.
          BodoReturnTypes.leastRestrictiveSubsetNullable(1, 3),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts two
          // matching input types
          OperandTypes.SAME_SAME_SAME,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ZEROIFNULL =
      new SqlFunction(
          "ZEROIFNULL",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // The output type is the same as the input type
          ReturnTypes.ARG0.andThen(SqlTypeTransforms.TO_NOT_NULLABLE),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts two
          // matching input types
          OperandTypes.NUMERIC,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  // TODO: Move aggregate function to their own operator table.

  public static final SqlBasicAggFunction CONDITIONAL_TRUE_EVENT =
      SqlBasicAggFunction.create(
          "CONDITIONAL_TRUE_EVENT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.INTEGER,
          OperandTypes.BOOLEAN);

  public static final SqlFunction DECODE =
      new SqlFunction(
          "DECODE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          BodoReturnTypes.DECODE_RETURN_TYPE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. See DecodeOperandChecker
          // for the rules
          DECODE_VARIADIC,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction HASH =
      new SqlFunction(
          "HASH",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BIGINT,
          null,
          OperandTypes.VARIADIC,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

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

  private List<SqlOperator> functionList =
      Arrays.asList(
          BOOLOR_AGG,
          BOOLAND_AGG,
          BOOLXOR_AGG,
          CONDITIONAL_TRUE_EVENT,
          COUNT_IF,
          CONDITIONAL_CHANGE_EVENT,
          REGR_VALX,
          REGR_VALY,
          IF_FUNC,
          IFF_FUNC,
          BOOLAND,
          BOOLOR,
          BOOLXOR,
          BOOLNOT,
          EQUAL_NULL,
          IFNULL_FUNC,
          NULLIFZERO,
          NVL,
          NVL2,
          ZEROIFNULL,
          DECODE,
          HASH,
          MIN_ROW_NUMBER_FILTER);

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
      // All Cond Operators are functions so far.
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
