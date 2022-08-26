package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.fun.SqlBasicAggFunction;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlSingleOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeTransforms;
import org.apache.calcite.sql.validate.SqlNameMatcher;

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

          // Return type needs to match arg1
          // TODO: this currently fails when inputs are timestamp/str, need
          // better restriction on input operands, see BS-581
          ReturnTypes.ARG1_NULLABLE,
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

          // Return type needs to match arg1
          // TODO: this currently fails when inputs are timestamp/str, need
          // better restriction on input operands, see BS-581
          ReturnTypes.ARG1_NULLABLE,
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
          ReturnTypes.BOOLEAN_NULLABLE,
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
          ReturnTypes.ARG0_NULLABLE,
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
          // LEAST_RESTRICTIVE will cast the return type to the least restrictive union of the
          // Three input types, and LEAST_NULLABLE will cast that type to a nullable type
          // If both of the three inputs are a nullable type.
          ReturnTypes.LEAST_RESTRICTIVE.andThen(SqlTypeTransforms.LEAST_NULLABLE),
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
          ReturnTypes.ARG0,
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. This function accepts two
          // matching input types
          OperandTypes.NUMERIC,
          // TODO: Add a proper category
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  /**
   * Takes in the arguments to a DECODE call and extracts a subset of the argument types that
   * correspond to outputs. For example:
   *
   * <p>DECODE(A, B, C, D, E) --> [C, E]
   *
   * <p>DECODE(A, B, C, D, E, F) --> [C, E, F]
   *
   * @param binding a container for all the operands of the DECODE function call
   * @return a list of all the output types corresponding to output arguments of DECODE
   */
  public static List<RelDataType> collectOutputTypes(SqlOperatorBinding binding) {
    List<RelDataType> operandTypes = binding.collectOperandTypes();
    List<RelDataType> outputTypes = new ArrayList<RelDataType>();
    int count = binding.getOperandCount();
    for (int i = 2; i < count; i++) {
      if (i % 2 == 0) {
        outputTypes.add(operandTypes.get(i));
      }
    }
    if (count > 3 && count % 2 == 0) {
      outputTypes.add(operandTypes.get(count - 1));
    }
    return outputTypes;
  }

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
          // Obtains the least restructive union of all the argument types
          // corresponding to outputs in the key-value pairs of arguments
          // (plus the optional default value argument)
          opBinding -> opBinding.getTypeFactory().leastRestrictive(collectOutputTypes(opBinding)),
          // What should be used to infer operand types. We don't use
          // this so we set it to None.
          null,
          // What Input Types does the function accept. See DecodeOperandChecker
          // for the rules
          DECODE_VARIADIC,
          // TODO: Add a proper category
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

  private List<SqlOperator> functionList =
      Arrays.asList(
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
          DECODE);

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
