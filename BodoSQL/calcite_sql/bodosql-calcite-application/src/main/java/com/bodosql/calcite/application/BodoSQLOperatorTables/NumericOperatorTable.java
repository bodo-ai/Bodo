package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.fun.SqlBasicAggFunction;
import org.apache.calcite.sql.fun.SqlCoalesceFunction;
import org.apache.calcite.sql.fun.SqlLibraryOperators;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SameOperandTypeChecker;
import org.apache.calcite.sql.type.SqlOperandTypeInference;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeTransforms;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.apache.calcite.sql.validate.implicit.TypeCoercion;
import org.apache.calcite.util.Optionality;

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

  public static final SqlFunction BITAND =
      new SqlFunction(
          "BITAND",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction BITOR =
      new SqlFunction(
          "BITOR",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction BITXOR =
      new SqlFunction(
          "BITXOR",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction BITNOT =
      new SqlFunction(
          "BITNOT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.INTEGER,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction BITSHIFTLEFT =
      new SqlFunction(
          "BITSHIFTLEFT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction BITSHIFTRIGHT =
      new SqlFunction(
          "BITSHIFTRIGHT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction GETBIT =
      new SqlFunction(
          "GETBIT",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0_NULLABLE,
          null,
          OperandTypes.family(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction CEILING =
      new SqlFunction(
          "CEILING",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BIGINT_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts one numeric argument
          OperandTypes.NUMERIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction DIV0 =
      new SqlFunction(
          "DIV0",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts two numerics
          // arguments
          OperandTypes.NUMERIC_NUMERIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction HAVERSINE =
      new SqlFunction(
          "HAVERSINE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts four numeric arguments
          OperandTypes.family(
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction LOG =
      new SqlFunction(
          "LOG",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept.
          // This function one or two numeric argument
          OperandTypes.or(OperandTypes.NUMERIC, OperandTypes.NUMERIC_NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction LOG2 =
      new SqlFunction(
          "LOG2",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.DOUBLE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts one numeric argument
          OperandTypes.NUMERIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction POW =
      new SqlFunction(
          "POW",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          (SqlOperandTypeInference) null,
          OperandTypes.NUMERIC_NUMERIC,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction TRUNC =
      new SqlFunction(
          "TRUNC",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          (SqlOperandTypeInference) null,
          OperandTypes.NUMERIC_INTEGER,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction CONV =
      new SqlFunction(
          "CONV",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.VARCHAR_2000,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts one string,
          // and two numeric arguments
          OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.NUMERIC, SqlTypeFamily.NUMERIC),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction WIDTH_BUCKET =
      new SqlFunction(
          "WIDTH_BUCKET",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts three numeric
          // arguments
          OperandTypes.family(
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.NUMERIC,
              SqlTypeFamily.INTEGER),
          // What group of functions does this fall into?
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction ACOSH =
      new SqlFunction(
          "ACOSH",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          null,
          OperandTypes.NUMERIC,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction ASINH =
      new SqlFunction(
          "ASINH",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          null,
          OperandTypes.NUMERIC,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction ATANH =
      new SqlFunction(
          "ATANH",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          null,
          OperandTypes.NUMERIC,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction FACTORIAL =
      new SqlFunction(
          "FACTORIAL",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BIGINT_NULLABLE,
          null,
          OperandTypes.INTEGER,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction SQUARE =
      new SqlFunction(
          "SQUARE",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.DOUBLE_NULLABLE,
          null,
          OperandTypes.NUMERIC,
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction COSH = SqlLibraryOperators.COSH;
  public static final SqlFunction SINH = SqlLibraryOperators.SINH;
  public static final SqlFunction TANH = SqlLibraryOperators.TANH;

  public static final SqlFunction GREATEST =
      new SqlLeastGreatestFunction("GREATEST", SqlKind.GREATEST);

  public static final SqlFunction LEAST = new SqlLeastGreatestFunction("LEAST", SqlKind.LEAST);

  public static final SqlBasicAggFunction VARIANCE_POP =
      SqlBasicAggFunction.create(
          "VARIANCE_POP", SqlKind.OTHER_FUNCTION, ReturnTypes.INTEGER, OperandTypes.NUMERIC);

  public static final SqlBasicAggFunction VARIANCE_SAMP =
      SqlBasicAggFunction.create(
          "VARIANCE_SAMP", SqlKind.OTHER_FUNCTION, ReturnTypes.INTEGER, OperandTypes.NUMERIC);

  public static final SqlBasicAggFunction CORR =
      SqlBasicAggFunction.create(
          "CORR", SqlKind.OTHER_FUNCTION, ReturnTypes.INTEGER, OperandTypes.NUMERIC_NUMERIC);

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
          ReturnTypes.DOUBLE_NULLABLE,
          OperandTypes.NUMERIC);

  public static final SqlAggFunction BITOR_AGG =
      SqlBasicAggFunction.create(
              "BITOR_AGG", SqlKind.BIT_OR, ReturnTypes.ARG0_NULLABLE, OperandTypes.NUMERIC)
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  // TODO: Add support for `precision` and `scale` arguments
  public static final SqlFunction TO_NUMBER =
      new SqlFunction(
          "TO_NUMBER",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BIGINT_NULLABLE,
          null,
          OperandTypes.or(OperandTypes.STRING, OperandTypes.NUMERIC),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction TO_NUMERIC =
      new SqlFunction(
          "TO_NUMERIC",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BIGINT_NULLABLE,
          null,
          OperandTypes.or(OperandTypes.STRING, OperandTypes.NUMERIC),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction TO_DECIMAL =
      new SqlFunction(
          "TO_DECIMAL",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BIGINT_NULLABLE,
          null,
          OperandTypes.or(OperandTypes.STRING, OperandTypes.NUMERIC),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction TRY_TO_NUMBER =
      new SqlFunction(
          "TRY_TO_NUMBER",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BIGINT_NULLABLE,
          null,
          OperandTypes.or(OperandTypes.STRING, OperandTypes.NUMERIC),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction TRY_TO_NUMERIC =
      new SqlFunction(
          "TRY_TO_NUMERIC",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BIGINT_NULLABLE,
          null,
          OperandTypes.or(OperandTypes.STRING, OperandTypes.NUMERIC),
          SqlFunctionCategory.NUMERIC);

  public static final SqlFunction TRY_TO_DECIMAL =
      new SqlFunction(
          "TRY_TO_DECIMAL",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.BIGINT_NULLABLE,
          null,
          OperandTypes.or(OperandTypes.STRING, OperandTypes.NUMERIC),
          SqlFunctionCategory.NUMERIC);

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
          CEILING,
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
          TRUNC,
          WIDTH_BUCKET,
          GREATEST,
          LEAST,
          VARIANCE_POP,
          VARIANCE_SAMP,
          TO_NUMBER,
          TO_NUMERIC,
          TO_DECIMAL,
          TRY_TO_NUMBER,
          TRY_TO_NUMERIC,
          TRY_TO_DECIMAL);

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

    private static class LeastGreatestOperandTypeChecker extends SameOperandTypeChecker {
      public LeastGreatestOperandTypeChecker() {
        super(-1);
      }

      public static LeastGreatestOperandTypeChecker INSTANCE = new LeastGreatestOperandTypeChecker();

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
