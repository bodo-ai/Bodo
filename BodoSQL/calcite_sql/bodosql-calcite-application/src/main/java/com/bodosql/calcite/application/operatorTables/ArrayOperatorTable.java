package com.bodosql.calcite.application.operatorTables;

import static org.apache.calcite.sql.type.BodoReturnTypes.toArrayReturnType;

import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.fun.SqlBasicAggFunction;
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.apache.calcite.util.Optionality;

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

  public static final SqlFunction ARRAY_CONSTRUCT =
      new SqlFunction(
          "ARRAY_CONSTRUCT",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          // TODO: This function can return an arary with different types, see
          //  https://docs.snowflake.com/en/sql-reference/functions/array_construct
          //  I'm not sure how we're handling this, so for now we're just disallowing
          //  anything that doesn't coerce to a common type
          // TODO: this function should also be able to accept array inputs to create
          // nested arrays. See https://bodo.atlassian.net/browse/BSE-1782
          ReturnTypes.LEAST_RESTRICTIVE.andThen(BodoReturnTypes.WRAP_TYPE_TO_ARRAY),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // The input can be any data type, any number of times.
          OperandTypes.VARIADIC,
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction TO_ARRAY =
      new SqlFunction(
          "TO_ARRAY",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          opBinding -> toArrayReturnType(opBinding),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // The input can be any data type.
          OperandTypes.ANY,
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ARRAY_TO_STRING =
      new SqlFunction(
          "ARRAY_TO_STRING",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // Final precision cannot be statically determined.
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // The input can be any data type.
          OperandTypes.sequence(
              "ARRAY_TO_STRING(ARRAY, STRING)", OperandTypes.ARRAY, OperandTypes.STRING),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlAggFunction ARRAY_AGG =
      SqlBasicAggFunction.create(
              "ARRAY_AGG",
              SqlKind.ARRAY_AGG,
              opBinding -> ArrayAggReturnType(opBinding),
              OperandTypes.ANY)
          .withGroupOrder(Optionality.OPTIONAL)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  /** Nulls are dropped by arrayAgg, so return a non-null array of the input type. */
  public static RelDataType ArrayAggReturnType(SqlOperatorBinding binding) {
    RelDataTypeFactory typeFactory = binding.getTypeFactory();
    RelDataType inputType = binding.collectOperandTypes().get(0);
    return typeFactory.createArrayType(typeFactory.createTypeWithNullability(inputType, false), -1);
  }

  public static final SqlFunction ARRAYS_OVERLAP =
      new SqlFunction(
          "ARRAYS_OVERLAP",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.BOOLEAN_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // The input can be any data type.
          OperandTypes.family(SqlTypeFamily.ARRAY, SqlTypeFamily.ARRAY),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ARRAY_POSITION =
      new SqlFunction(
          "ARRAY_POSITION",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          // return value is null if the value doesn't exist in the array, so FORCE_NULLABLE is
          // needed
          BodoReturnTypes.INTEGER_FORCE_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // The input can be any data type.
          OperandTypes.family(SqlTypeFamily.ANY, SqlTypeFamily.ARRAY),
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction ARRAY_SIZE =
      new SqlFunction(
          "ARRAY_SIZE",
          // What SqlKind should match?
          // TODO: Extend SqlKind with our own functions
          SqlKind.OTHER_FUNCTION,
          // What Value should the return type be
          ReturnTypes.INTEGER_NULLABLE,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // The input can be any data type.
          OperandTypes.ARRAY,
          // What group of functions does this fall into?
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  private List<SqlOperator> functionList =
      Arrays.asList(
          TO_ARRAY,
          ARRAY_CONSTRUCT,
          ARRAY_TO_STRING,
          ARRAY_AGG,
          ARRAY_SIZE,
          ARRAYS_OVERLAP,
          ARRAY_POSITION);

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
