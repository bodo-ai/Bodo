package com.bodosql.calcite.application.operatorTables;

import static com.bodosql.calcite.application.operatorTables.OperatorTableUtils.argumentRange;
import static org.apache.calcite.sql.type.BodoReturnTypes.bitX_ret_type;

import com.bodosql.calcite.sql.func.SqlRandomOperator;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import org.apache.calcite.plan.Strong;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlBasicFunction;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.fun.SqlBasicAggFunction;
import org.apache.calcite.sql.fun.SqlLibraryOperators;
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SameOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeTransforms;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.apache.calcite.sql.validate.implicit.TypeCoercion;
import org.apache.calcite.util.Optionality;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;

public final class NumericOperatorTable implements SqlOperatorTable {

  private static @Nullable NumericOperatorTable instance;

  /** Returns the Datetime operator table, creating it if necessary. */
  public static synchronized NumericOperatorTable instance() {
    NumericOperatorTable instance = NumericOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new NumericOperatorTable();
      NumericOperatorTable.instance = instance;
    }
    return instance;
  }

  // TODO: Extend the Library Operator and use the builtin Libraries

  public static final SqlNullPolicyFunction BITAND =
      SqlNullPolicyFunction.createAnyPolicy(
          "BITAND",
          ReturnTypes.ARG0_NULLABLE,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction BITOR = BITAND.withName("BITOR");

  public static final SqlFunction BITXOR = BITAND.withName("BITXOR");

  public static final SqlFunction BITNOT =
      SqlNullPolicyFunction.createAnyPolicy(
          "BITNOT", ReturnTypes.ARG0_NULLABLE, OperandTypes.INTEGER, SqlFunctionCategory.NUMERIC);

  public static final SqlFunction BITSHIFTLEFT = BITAND.withName("BITSHIFTLEFT");

  public static final SqlFunction BITSHIFTRIGHT = BITAND.withName("BITSHIFTRIGHT");

  public static final SqlFunction GETBIT = BITAND.withName("GETBIT");

  public static final SqlNullPolicyFunction SNOWFLAKE_CEIL =
      SqlNullPolicyFunction.createAnyPolicy(
          "CEIL",
          ReturnTypes.ARG0_NULLABLE,
          argumentRange(1, SqlTypeFamily.NUMERIC, SqlTypeFamily.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction SNOWFLAKE_FLOOR = SNOWFLAKE_CEIL.withName("FLOOR");

  public static final SqlFunction CEILING =
      SqlNullPolicyFunction.createAnyPolicy(
          "CEILING",
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What Input Types does the function accept. This function accepts one numeric argument
          OperandTypes.NUMERIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction DIV0 =
      SqlNullPolicyFunction.createAnyPolicy(
          "DIV0",
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What Input Types does the function accept. This function accepts two numerics
          // arguments
          OperandTypes.NUMERIC_NUMERIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction HAVERSINE =
      SqlNullPolicyFunction.createAnyPolicy(
          "HAVERSINE",
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What Input Types does the function accept. This function accepts four numeric arguments
          OperandTypes.family(
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction LOG =
      SqlNullPolicyFunction.createAnyPolicy(
          "LOG",
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What Input Types does the function accept.
          // This function one or two numeric argument
          OperandTypes.NUMERIC.or(OperandTypes.NUMERIC_NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction LOG2 =
      SqlNullPolicyFunction.createAnyPolicy(
          "LOG2",
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What Input Types does the function accept. This function accepts one numeric argument
          OperandTypes.NUMERIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction POW =
      SqlNullPolicyFunction.createAnyPolicy(
          "POW",
          ReturnTypes.DOUBLE_NULLABLE,
          OperandTypes.NUMERIC_NUMERIC,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction CONV =
      SqlNullPolicyFunction.createAnyPolicy(
          "CONV",
          // precision cannot be statically determined
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          // What Input Types does the function accept. This function accepts one string,
          // and two numeric arguments
          OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction WIDTH_BUCKET =
      SqlNullPolicyFunction.createAnyPolicy(
          "WIDTH_BUCKET",
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What Input Types does the function accept. This function accepts three numeric
          // arguments
          OperandTypes.family(
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlNullPolicyFunction ACOSH =
      SqlNullPolicyFunction.createAnyPolicy(
          "ACOSH", ReturnTypes.DOUBLE_NULLABLE, OperandTypes.NUMERIC, SqlFunctionCategory.NUMERIC);

  public static final SqlFunction ASINH = ACOSH.withName("ASINH");

  public static final SqlFunction ATANH = ACOSH.withName("ATANH");

  public static final SqlFunction FACTORIAL =
      SqlNullPolicyFunction.createAnyPolicy(
          "FACTORIAL",
          ReturnTypes.BIGINT_NULLABLE,
          OperandTypes.INTEGER,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction SQUARE =
      SqlNullPolicyFunction.createAnyPolicy(
          "SQUARE", ReturnTypes.DOUBLE_NULLABLE, OperandTypes.NUMERIC, SqlFunctionCategory.NUMERIC);

  public static final SqlFunction COSH = SqlLibraryOperators.COSH;
  public static final SqlFunction SINH = SqlLibraryOperators.SINH;
  public static final SqlFunction TANH = SqlLibraryOperators.TANH;

  public static final SqlFunction UNIFORM =
      SqlBasicFunction.create(
          "UNIFORM",
          ReturnTypes.LEAST_RESTRICTIVE,
          OperandTypes.sequence(
              "UNIFORM(NUMERIC, NUMERIC, NUMERIC)",
              OperandTypes.NUMERIC,
              OperandTypes.NUMERIC,
              OperandTypes.NUMERIC),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction GREATEST =
      new SqlLeastGreatestFunction("GREATEST", SqlKind.GREATEST);

  public static final SqlFunction LEAST = new SqlLeastGreatestFunction("LEAST", SqlKind.LEAST);

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
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          OperandTypes.NUMERIC_NUMERIC);

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
              OperandTypes.NUMERIC.or(OperandTypes.STRING))
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  public static final SqlAggFunction BITAND_AGG =
      SqlBasicAggFunction.create(
              "BITAND_AGG",
              SqlKind.BIT_AND,
              sqlOperatorBinding -> bitX_ret_type(sqlOperatorBinding),
              OperandTypes.NUMERIC.or(OperandTypes.STRING))
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

  public static final SqlFunction RANDOM = new SqlRandomOperator();

  private List<SqlOperator> functionList =
      Arrays.asList(
          ACOSH,
          ASINH,
          ATANH,
          APPROX_PERCENTILE,
          COSH,
          SINH,
          TANH,
          BITAND,
          BITOR,
          BITXOR,
          BITNOT,
          BITSHIFTLEFT,
          BITSHIFTRIGHT,
          GETBIT,
          BITOR_AGG,
          BITAND_AGG,
          BITXOR_AGG,
          CEILING,
          SNOWFLAKE_CEIL,
          SNOWFLAKE_FLOOR,
          DIV0,
          HAVERSINE,
          LOG,
          LOG2,
          CORR,
          MEDIAN,
          KURTOSIS,
          SKEW,
          RATIO_TO_REPORT,
          POW,
          CONV,
          FACTORIAL,
          SQUARE,
          WIDTH_BUCKET,
          UNIFORM,
          RANDOM,
          GREATEST,
          LEAST,
          VARIANCE_POP,
          VARIANCE_SAMP);

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

  private static class SqlLeastGreatestFunction extends SqlFunction {
    public SqlLeastGreatestFunction(String name, SqlKind kind) {
      super(
          name,
          kind,
          ReturnTypes.LEAST_RESTRICTIVE.andThen(SqlTypeTransforms.TO_NULLABLE),
          null,
          LeastGreatestOperandTypeChecker.INSTANCE,
          SqlFunctionCategory.SYSTEM);
    }

    @Override
    public @Nullable SqlOperator reverse() {
      if (getKind() == SqlKind.GREATEST) {
        return LEAST;
      } else {
        return GREATEST;
      }
    }

    /**
     * Returns the {@link Strong.Policy} strategy for this operator, or null if there is no
     * particular strategy, in which case this policy will be deducted from the operator's {@link
     * SqlKind}.
     *
     * @see Strong
     */
    @Override
    @Pure
    public @Nullable Supplier<Strong.Policy> getStrongPolicyInference() {
      // TODO: Define a new policy ALL, that means all inputs must be nullable.
      return () -> Strong.Policy.ANY;
    }

    private static class LeastGreatestOperandTypeChecker extends SameOperandTypeChecker {
      public LeastGreatestOperandTypeChecker() {
        super(-1);
      }

      public static LeastGreatestOperandTypeChecker INSTANCE =
          new LeastGreatestOperandTypeChecker();

      @Override
      public boolean checkOperandTypes(SqlCallBinding callBinding, boolean throwOnFailure) {
        if (callBinding.isTypeCoercionEnabled()) {
          TypeCoercion typeCoercion = callBinding.getValidator().getTypeCoercion();
          List<RelDataType> operandTypes = callBinding.collectOperandTypes();
          RelDataType type = typeCoercion.getWiderTypeFor(operandTypes, true);
          if (null != type) {
            ArrayList<SqlTypeFamily> expectedFamilies = new ArrayList<>();
            for (int i = 0; i < operandTypes.size(); i++) {
              expectedFamilies.add(type.getSqlTypeName().getFamily());
            }
            typeCoercion.builtinFunctionCoercion(callBinding, operandTypes, expectedFamilies);
          }
        }
        return super.checkOperandTypes(callBinding, throwOnFailure);
      }
    }
  }
}
