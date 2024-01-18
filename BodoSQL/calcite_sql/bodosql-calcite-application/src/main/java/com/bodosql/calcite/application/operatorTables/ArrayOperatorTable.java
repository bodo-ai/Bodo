package com.bodosql.calcite.application.operatorTables;

import java.util.Arrays;
import java.util.List;
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
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeTransforms;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.checkerframework.checker.nullness.qual.Nullable;

public class ArrayOperatorTable implements SqlOperatorTable {
  private static @Nullable ArrayOperatorTable instance;

  /** Returns the operator table, creating it if necessary. */
  public static synchronized ArrayOperatorTable instance() {
    ArrayOperatorTable instance = ArrayOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new ArrayOperatorTable();
      ArrayOperatorTable.instance = instance;
    }
    return instance;
  }

  public static final SqlFunction ARRAY_COMPACT =
      SqlNullPolicyFunction.createAnyPolicy(
          "ARRAY_COMPACT",
          ReturnTypes.ARG0_NULLABLE,
          OperandTypes.ARRAY,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ARRAY_CONSTRUCT =
      SqlBasicFunction.create(
          "ARRAY_CONSTRUCT",
          // What Value should the return type be
          // TODO: This function can return an arary with different types, see
          //  https://docs.snowflake.com/en/sql-reference/functions/array_construct
          //  I'm not sure how we're handling this, so for now we're just disallowing
          //  anything that doesn't coerce to a common type
          ReturnTypes.LEAST_RESTRICTIVE.andThen(SqlTypeTransforms.TO_ARRAY),
          // The input can be any data type, any number of times.
          // We require there to be at least one input type, otherwise
          // ReturnTypes.LEAST_RESTRICTIVE will throw an error. This isn't an easy
          // fix for other reasons detailed in the ticket here:
          // https://bodo.atlassian.net/browse/BSE-2111
          OperandTypes.ONE_OR_MORE,
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ARRAY_CONSTRUCT_COMPACT =
      SqlBasicFunction.create(
          "ARRAY_CONSTRUCT_COMPACT",
          ReturnTypes.LEAST_RESTRICTIVE
              .andThen(SqlTypeTransforms.TO_ARRAY)
              .andThen(SqlTypeTransforms.TO_NULLABLE),
          OperandTypes.ONE_OR_MORE,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ARRAY_TO_STRING =
      SqlBasicFunction.create(
          "ARRAY_TO_STRING",
          // Final precision cannot be statically determined.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          // The input can be any data type.
          OperandTypes.sequence(
              "ARRAY_TO_STRING(ARRAY, STRING)", OperandTypes.ARRAY, OperandTypes.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  // GET(val, idx) syntax
  public static final SqlFunction ARRAY_MAP_GET = new BodoSqlItemOperatorGetFnSyntax();

  // val[idx] syntax
  public static final SqlOperator ARRAY_MAP_GET_BRACKET = new BodoSqlItemOperatorBracketsSyntax();

  /** Nulls are dropped by arrayAgg, so return a non-null array of the input type. */
  public static RelDataType ArrayAggReturnType(SqlOperatorBinding binding) {
    RelDataTypeFactory typeFactory = binding.getTypeFactory();
    RelDataType inputType = binding.collectOperandTypes().get(0);
    return typeFactory.createArrayType(typeFactory.createTypeWithNullability(inputType, false), -1);
  }

  public static final SqlFunction ARRAY_CONTAINS =
      SqlBasicFunction.create(
          "ARRAY_CONTAINS",
          ReturnTypes.BOOLEAN.andThen(BodoReturnTypes.TO_NULLABLE_ARG1),
          OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.ARRAY),
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ARRAYS_OVERLAP =
      SqlBasicFunction.create(
          "ARRAYS_OVERLAP",
          // What Value should the return type be
          ReturnTypes.BOOLEAN_NULLABLE,
          // The input can be any data type.
          OperandTypes.family(SqlTypeFamily.ARRAY, SqlTypeFamily.ARRAY),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ARRAY_POSITION =
      SqlBasicFunction.create(
          "ARRAY_POSITION",
          // What Value should the return type be
          // return value is null if the value doesn't exist in the array, so FORCE_NULLABLE is
          // needed
          BodoReturnTypes.INTEGER_FORCE_NULLABLE,
          // The input can be any data type.
          OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.ARRAY),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ARRAY_SIZE =
      SqlBasicFunction.create(
          "ARRAY_SIZE",
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // The input can be any data type.
          OperandTypes.ARRAY,
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ARRAY_REMOVE =
      SqlBasicFunction.create(
          "ARRAY_REMOVE",
          ReturnTypes.ARG0_NULLABLE,
          OperandTypes.sequence("ARRAY_REMOVE(ARRAY, ANY)", OperandTypes.ARRAY, OperandTypes.ANY),
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ARRAY_REMOVE_AT =
      SqlBasicFunction.create(
          "ARRAY_REMOVE_AT",
          ReturnTypes.ARG0_NULLABLE,
          OperandTypes.sequence(
              "ARRAY_REMOVE_AT(ARRAY, INTEGER)", OperandTypes.ARRAY, OperandTypes.INTEGER),
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlBasicFunction ARRAY_EXCEPT =
      SqlBasicFunction.create(
          "ARRAY_EXCEPT",
          // What Value should the return type be
          ReturnTypes.ARG0_NULLABLE,
          // The input can be any data type.
          OperandTypes.sequence(
              "ARRAY_EXCEPT(ARRAY, ARRAY)", OperandTypes.ARRAY, OperandTypes.ARRAY),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ARRAY_INTERSECTION = ARRAY_EXCEPT.withName("ARRAY_INTERSECTION");

  public static final SqlFunction ARRAY_CAT = ARRAY_EXCEPT.withName("ARRAY_CAT");

  public static final SqlFunction ARRAY_SLICE =
      SqlBasicFunction.create(
          "ARRAY_SLICE",
          ReturnTypes.ARG0_NULLABLE,
          OperandTypes.sequence(
              "ARRAY_SLICE(ARRAY, INTEGER, INTEGER)",
              OperandTypes.ARRAY,
              OperandTypes.INTEGER,
              OperandTypes.INTEGER),
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  private List<SqlOperator> functionList =
      Arrays.asList(
          ARRAY_COMPACT,
          ARRAY_CONSTRUCT,
          ARRAY_CONSTRUCT_COMPACT,
          ARRAY_EXCEPT,
          ARRAY_INTERSECTION,
          ARRAY_CAT,
          ARRAY_TO_STRING,
          ARRAY_SIZE,
          ARRAY_REMOVE,
          ARRAY_REMOVE_AT,
          ARRAY_SLICE,
          ARRAYS_OVERLAP,
          ARRAY_CONTAINS,
          ARRAY_POSITION,
          ARRAY_MAP_GET,
          ARRAY_MAP_GET_BRACKET);

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
      if (operator instanceof SqlFunction) {
        // All String Operators added are functions so far.
        SqlFunction func = (SqlFunction) operator;
        if (syntax != func.getSyntax()) {
          continue;
        }
        // Check that the name matches the desired names.
        if (!opName.isSimple() || !nameMatcher.matches(func.getName(), opName.getSimple())) {
          continue;
        }
      }
      if (operator.getSyntax().family != syntax) {
        continue;
      }
      // TODO: Check the category. The Lexing currently thinks
      //  all of these functions are user defined functions.
      operatorList.add(operator);
    }
  }

  @Override
  public List<SqlOperator> getOperatorList() {
    return functionList;
  }
}
