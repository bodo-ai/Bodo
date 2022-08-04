package com.bodosql.calcite.application.BodoSQLOperatorTables;

import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
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

  private List<SqlOperator> functionList =
      Arrays.asList(
          REGR_VALX, REGR_VALY, IF_FUNC, IFF_FUNC, IFNULL_FUNC, NULLIFZERO, NVL, NVL2, ZEROIFNULL);

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
