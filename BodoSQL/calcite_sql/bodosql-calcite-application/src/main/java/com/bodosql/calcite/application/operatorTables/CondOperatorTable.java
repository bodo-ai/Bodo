package com.bodosql.calcite.application.operatorTables;

import java.util.Arrays;
import java.util.List;
import org.apache.calcite.sql.SqlBasicFunction;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlSingleOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeTransforms;
import org.apache.calcite.sql.validate.SqlNameMatcher;
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

  public static final SqlBasicFunction REGR_VALX =
      SqlBasicFunction.create(
          "REGR_VALX",
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What Input Types does the function accept. It accepts two doubles
          OperandTypes.NUMERIC_NUMERIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.STRING);

  public static final SqlFunction REGR_VALY = REGR_VALX.withName("REGR_VALY");

  // TODO: Extend the Library Operator and use the builtin Libraries
  public static final SqlBasicFunction IF_FUNC =
      SqlBasicFunction.create(
          "IF",
          // The return type consists of the least restrictive of arguments 1 and 2
          // and if either is null.
          // null is considered false for this function, so we don't need to consider its
          // nullability
          // when determining output nullability
          BodoReturnTypes.leastRestrictiveSubsetNullable(1, 3),
          // What Input Types does the function accept. This function accepts
          // a boolean arg0 and two matching args
          BOOLEAN_SAME_SAME,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  // TODO: Extend the Library Operator and use the builtin Libraries
  public static final SqlFunction IFF_FUNC = IF_FUNC.withName("IFF");

  public static final SqlBasicFunction BOOLAND =
      SqlBasicFunction.create(
          "BOOLAND",
          ReturnTypes.BOOLEAN_NULLABLE,
          OperandTypes.NUMERIC_NUMERIC,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction BOOLOR = BOOLAND.withName("BOOLOR");

  public static final SqlFunction BOOLXOR = BOOLAND.withName("BOOLXOR");

  public static final SqlFunction BOOLNOT =
      SqlBasicFunction.create(
          "BOOLNOT",
          ReturnTypes.BOOLEAN_NULLABLE,
          OperandTypes.NUMERIC,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction EQUAL_NULL =
      SqlBasicFunction.create(
          "EQUAL_NULL",
          ReturnTypes.BOOLEAN,
          OperandTypes.SAME_SAME,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction NULLIFZERO =
      SqlBasicFunction.create(
          "NULLIFZERO",
          // The output type is the same as the input type, but nullable
          ReturnTypes.ARG0_FORCE_NULLABLE,
          // What Input Types does the function accept. This function accepts two
          // matching input types
          OperandTypes.NUMERIC,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlBasicFunction NVL =
      SqlBasicFunction.create(
          "NVL",
          // LEAST_RESTRICTIVE will cast the return type to the least restrictive union of the
          // Two input types, and LEAST_NULLABLE will cast that type to a nullable type
          // If both of the two inputs are a nullable type.
          ReturnTypes.LEAST_RESTRICTIVE.andThen(SqlTypeTransforms.LEAST_NULLABLE),
          // What Input Types does the function accept. This function accepts two
          // matching input types
          OperandTypes.SAME_SAME,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction IFNULL_FUNC = NVL.withName("IFNULL");

  public static final SqlFunction NVL2 =
      SqlBasicFunction.create(
          "NVL2",
          // The return type consists of the least restrictive of arguments 1 and 2
          // and if either is null.
          BodoReturnTypes.leastRestrictiveSubsetNullable(1, 3),
          // What Input Types does the function accept. This function accepts two
          // matching input types
          OperandTypes.SAME_SAME_SAME,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ZEROIFNULL =
      SqlBasicFunction.create(
          "ZEROIFNULL",
          // The output type is the same as the input type
          ReturnTypes.ARG0.andThen(SqlTypeTransforms.TO_NOT_NULLABLE),
          // What Input Types does the function accept. This function accepts two
          // matching input types
          OperandTypes.NUMERIC,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction DECODE =
      SqlBasicFunction.create(
          "DECODE",
          BodoReturnTypes.DECODE_RETURN_TYPE,
          // What Input Types does the function accept. See DecodeOperandChecker
          // for the rules
          DECODE_VARIADIC,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction HASH =
      SqlBasicFunction.create(
          "HASH",
          ReturnTypes.BIGINT,
          OperandTypes.VARIADIC,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  private List<SqlOperator> functionList =
      Arrays.asList(
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
          HASH);

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
