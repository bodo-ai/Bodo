package com.bodosql.calcite.application.operatorTables;

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
import org.apache.calcite.sql.type.ArraySqlType;
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlTypeName;
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

  /**
   * Determine the return type of the TO_ARRAY function
   *
   * @param binding The operand bindings for the function signature.
   * @return The return type of the function
   */
  public static RelDataType toArrayReturnType(SqlOperatorBinding binding) {
    RelDataTypeFactory typeFactory = binding.getTypeFactory();
    RelDataType inputType = binding.getOperandType(0);
    if (inputType.getSqlTypeName().equals(SqlTypeName.NULL))
      // if the input is null, TO_ARRAY will return NULL, not an array of NULL
      return inputType;
    if (inputType instanceof ArraySqlType)
      // if the input is an array, just return it
      return inputType;
    return typeFactory.createArrayType(inputType, -1);
  }

  public static final SqlAggFunction ARRAY_AGG =
      SqlBasicAggFunction.create(
              "ARRAY_AGG", SqlKind.ARRAY_AGG, ReturnTypes.TO_ARRAY, OperandTypes.ANY)
          .withGroupOrder(Optionality.OPTIONAL)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

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
      Arrays.asList(TO_ARRAY, ARRAY_TO_STRING, ARRAY_AGG, ARRAY_SIZE);

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
