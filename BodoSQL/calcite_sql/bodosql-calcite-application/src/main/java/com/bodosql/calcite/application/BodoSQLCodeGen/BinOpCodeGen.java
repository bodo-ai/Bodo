package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLExprType.meet_elementwise_op;
import static com.bodosql.calcite.application.Utils.Utils.generateNullCheck;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLExprType;
import java.util.HashSet;
import java.util.List;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;

/**
 * Class that returns the generated code for a BinOp expression after all inputs have been visited.
 */
public class BinOpCodeGen {

  /**
   * Function that return the necessary generated code for a BinOp Call. This function may have more
   * than two arguments because repeated use of the same operator are grouped into the same Node.
   *
   * @param args The arguments to the binop.
   * @param exprTypes The exprType of each argument.
   * @param binOp The Binary operator to apply to each pair of arguments.
   * @param isScalar Is this function used inside an apply and should always generate scalar code.
   * @return The code generated for the BinOp call.
   */
  public static String generateBinOpCode(
      List<String> args,
      List<BodoSQLExprType.ExprType> exprTypes,
      SqlOperator binOp,
      boolean isScalar) {
    return generateBinOpCodeHelper(args, exprTypes, binOp.getKind(), binOp.getName(), isScalar);
  }
  /**
   * Helper function that returns the necessary generated code for a BinOp Call. This function may
   * have more than two arguments because repeated use of the same operator are grouped into the
   * same Node.
   *
   * @param args The arguments to the binop.
   * @param exprTypes The exprType of each argument.
   * @param binOpKind The SqlKind of the binary operator
   * @param binOpName The name of the binary operator
   * @param isScalar Is this function used inside an apply and should always generate scalar code.
   * @return The code generated for the BinOp call.
   */
  public static String generateBinOpCodeHelper(
      List<String> args,
      List<BodoSQLExprType.ExprType> exprTypes,
      SqlKind binOpKind,
      String binOpName,
      boolean isScalar) {
    String columnOperator;
    String scalarOperator;
    // Is the column operator generated as a function call
    boolean columnIsFunction = false;
    // Is the scalar operator generated as a function call
    boolean scalarIsFunction = false;
    switch (binOpKind) {
      case EQUALS:
        columnOperator = "==";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_equal";
        break;
      case NULL_EQUALS:
        columnIsFunction = true;
        columnOperator = "bodosql.libs.sql_operators.sql_null_equal_column";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.null_handling.sql_null_equal_scalar";
        break;
      case NOT_EQUALS:
        columnOperator = "!=";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_not_equal";
        break;
      case LESS_THAN:
        columnOperator = "<";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_less_than";
        break;
      case GREATER_THAN:
        columnOperator = ">";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_greater_than";
        break;
      case PLUS:
        columnOperator = "+";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_addition";
        break;
      case MINUS:
        columnOperator = "-";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_subtraction";
        break;
      case TIMES:
        columnOperator = "*";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_multiplication";
        break;
      case LESS_THAN_OR_EQUAL:
        columnOperator = "<=";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_less_than_or_equal";
        break;
      case GREATER_THAN_OR_EQUAL:
        columnOperator = ">=";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_greater_than_or_equal";
        break;
      case AND:
        columnOperator = "&";
        scalarOperator = "and";
        break;
      case DIVIDE:
        columnOperator = "/";
        scalarIsFunction = true;
        scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_true_division";
        break;
      case OTHER_FUNCTION:
        switch (binOpName) {
          case "CONCAT_WS":
          case "CONCAT":
            columnOperator = "+";
            scalarIsFunction = true;
            scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_addition";
            break;
          default:
            throw new BodoSQLCodegenException(
                "Unsupported Operator, " + binOpName.toString() + " specified in query.");
        }
        break;
      case OTHER:
        switch (binOpName) {
          case "||":
            columnOperator = "+";
            scalarIsFunction = true;
            scalarOperator = "bodosql.libs.generated_lib.sql_null_checking_addition";
            break;
          default:
            throw new BodoSQLCodegenException(
                "Unsupported Operator, " + binOpName.toString() + " specified in query.");
        }
        break;
      default:
        throw new BodoSQLCodegenException(
            "Unsupported Operator, " + binOpKind.toString() + " specified in query.");
    }
    StringBuilder newOp = new StringBuilder();
    StringBuilder functionStack = new StringBuilder();
    newOp.append("(");
    BodoSQLExprType.ExprType exprType = exprTypes.get(0);
    String operator;
    boolean wasFunction;
    boolean isFunction = false;
    /**
     * Iterate through the arguments to generate the necessary code. For operators that map to
     * operators in Pandas, generate ARG0 OP ARG1. For operators that map to functions generate
     * OP(ARG0, ARG1)
     *
     * <p>To keep inserting function calls to front, we use two StringBuilders so we can keep
     * finding the starting position.
     */
    for (int i = 0; i < args.size() - 1; i++) {
      exprType = meet_elementwise_op(exprType, exprTypes.get(i + 1));
      wasFunction = isFunction;
      if (isScalar || exprType == BodoSQLExprType.ExprType.SCALAR) {
        operator = scalarOperator;
        isFunction = scalarIsFunction;
      } else {
        operator = columnOperator;
        isFunction = columnIsFunction;
      }
      if (isFunction) {
        functionStack.insert(0, operator + "(");
        functionStack.append(args.get(i));
        if (wasFunction) {
          functionStack.append(")");
        }
        functionStack.append(", ");
      } else {
        if (wasFunction) {
          newOp.append(functionStack).append(args.get(i)).append(")");
          // Clear the function stack
          functionStack = new StringBuilder();
        } else {
          newOp.append(args.get(i));
        }
        newOp.append(" ").append(operator).append(" ");
      }
    }
    newOp.append(functionStack);
    newOp.append(args.get(args.size() - 1)).append(")");
    if (isFunction) {
      newOp.append(")");
    }
    return newOp.toString();
  }

  /**
   * Function that returns the generated name for a Binop Call. This function may have more than two
   * arguments because repeated use of the same operator are grouped into the same Node.
   *
   * @param names The names of the arguments to the binop
   * @param binOp The Binary operator to apply to each pair of arguments.
   * @return The name generated that matches the Binop expression.
   */
  public static String generateBinOpName(List<String> names, SqlOperator binOp) {
    StringBuilder nameBuilder = new StringBuilder();
    for (int i = 0; i < names.size(); i++) {
      nameBuilder.append(binOp.toString()).append("(");
    }
    nameBuilder.append(names.get(0));
    for (int i = 1; i < names.size(); i++) {
      nameBuilder.append(", ").append(names.get(i)).append(")");
    }
    return nameBuilder.toString();
  }

  /**
   * Function that generates code for a single argument in OR.
   *
   * @param argCode The code generate for that single argument
   * @param appendOperator Should the operator be added to generated string
   * @param inputVar The name of the input table which columns reference
   * @param nullSet Set of columns that may need to be treated as null. If this argument is treated
   *     as column then the set will be empty.
   * @param isScalar Is this function used inside an apply and should always generate scalar code.
   * @return The code generated for that single argument.
   */
  public static String generateOrCode(
      String argCode,
      boolean appendOperator,
      String inputVar,
      HashSet<String> nullSet,
      boolean isScalar) {
    StringBuilder newArg = new StringBuilder();
    // This generates code as (False if (pd.isna(col0) or ... pd.isna(coln)) else argCode)
    // There may be ways to refactor this to generate more efficient code.
    newArg.append(generateNullCheck(inputVar, nullSet, "False", argCode));
    if (appendOperator) {
      if (isScalar) {
        newArg.append(" or ");
      } else {
        newArg.append(" | ");
      }
    }
    return newArg.toString();
  }
}
