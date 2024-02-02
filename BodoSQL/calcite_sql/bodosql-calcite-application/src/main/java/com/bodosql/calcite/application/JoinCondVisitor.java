package com.bodosql.calcite.application;

import static com.bodosql.calcite.application.utils.Utils.escapePythonQuotes;
import static com.bodosql.calcite.application.utils.Utils.makeQuoted;

import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import org.apache.calcite.rel.core.JoinInfo;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlBinaryOperator;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlPostfixOperator;
import org.apache.calcite.sql.SqlPrefixOperator;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.Pair;

/**
 * Visitor Class for Join Conditions. This special visitor class is used to avoid special handling
 * for Join conditions inside the general PandasCodeGenVisitor as no node will have side effects.
 * All expressions in this form should match the support on conditions inside Bodo and are
 * restricted to simple constants (string, int, float), arithmetic expressions, comparison operators
 * excluding nulleq, logical operators, and column accesses.
 *
 * <p>Returns a Expr containing the generated code.
 */
public class JoinCondVisitor {
  public static Expr visitJoinCond(
      RexNode joinNode,
      List<String> leftColNames,
      List<String> rightColNames,
      HashSet<String> mergeCols) {
    /** General function for join conditions. */
    Expr outputString;
    if (joinNode instanceof RexInputRef) {
      outputString = visitJoinInputRef((RexInputRef) joinNode, leftColNames, rightColNames);
    } else if (joinNode instanceof RexLiteral) {
      outputString = visitJoinLiteral((RexLiteral) joinNode);
    } else if (joinNode instanceof RexCall) {
      outputString = visitJoinCall((RexCall) joinNode, leftColNames, rightColNames, mergeCols);
    } else {
      throw new BodoSQLCodegenException(
          "Internal Error: Calcite Plan Produced an unsupported Join Condition");
    }
    return outputString;
  }

  /**
   * API Equivalent to visitJoinCond, but called from the nonEquiConditions of a join info.
   *
   * @param conds List of conditions that should be AND together.
   * @param leftColNames The column names from the left table.
   * @param rightColNames The column names from the right table.
   * @return A Expr that is the body of a string to pass to the Bodo Runtime
   */
  public static Expr visitNonEquiConditions(
      List<RexNode> conds, List<String> leftColNames, List<String> rightColNames) {
    if (conds.size() == 0) {
      return new Expr.Raw("");
    }
    // Create the hashset to enable calling visitJoinCond but the output
    // will be unused.
    HashSet<String> unused = new HashSet<>();
    Expr result = visitJoinCond(conds.get(0), leftColNames, rightColNames, unused);
    for (int i = 1; i < conds.size(); i++) {
      Expr newCond = visitJoinCond(conds.get(i), leftColNames, rightColNames, unused);
      result = new Expr.Binary("&", result, newCond);
    }
    return result;
  }

  public static Variable visitJoinInputRef(
      RexInputRef joinNode, List<String> leftColNames, List<String> rightColNames) {
    /**
     * Visit node for an input ref within a join condition. This should output either left.`column`
     * or right.`column`. We use ` ` to avoid non-identifier column issues.
     */
    int colNum = joinNode.getIndex();
    if (colNum < leftColNames.size()) {
      String leftColName = leftColNames.get(colNum);
      leftColName = "`" + leftColName + "`";
      return new Variable(String.format("left.%s", leftColName));
    } else {
      String rightColName = rightColNames.get(colNum - leftColNames.size());
      rightColName = "`" + rightColName + "`";
      return new Variable(String.format("right.%s", rightColName));
    }
  }

  public static Expr visitJoinLiteral(RexLiteral joinNode) {
    /**
     * Visit node for a literal within a join condition. This should output the literal value for
     * literals that can be supported within a join condition.
     */
    SqlTypeName typeName = joinNode.getType().getSqlTypeName();
    // TODO: Refactor this to use literal expressions
    // I'm not doing it right now because there's a fair amount of wierd casting/string manipulation
    // going on, and I don't want to make changes that have the potential to actually change
    // correctness.
    switch (typeName) {
      case TINYINT:
      case SMALLINT:
      case INTEGER:
      case BIGINT:
      case FLOAT:
      case REAL:
      case DOUBLE:
      case DECIMAL:
        return new Expr.Raw(joinNode.getValue().toString());
      case CHAR:
      case VARCHAR:
        // extract value without specific sql type info.
        return new Expr.Raw(makeQuoted(escapePythonQuotes(joinNode.getValue2().toString())));
      case BOOLEAN:
        String boolName = joinNode.toString();
        return new Expr.Raw(boolName.substring(0, 1).toUpperCase() + boolName.substring(1));
      default:
        throw new BodoSQLCodegenException(
            String.format(
                "Internal Error: Literal Type %s not supported within a join condition.",
                typeName));
    }
  }

  public static Expr visitJoinCall(
      RexCall joinNode,
      List<String> leftColNames,
      List<String> rightColNames,
      HashSet<String> mergeCols) {
    /**
     * Visit node for a call within a join condition. This should be limited to logical operators,
     * comparison operators (excluding <=>), arithmetic operators, and POW().
     *
     * <p>Returns a Pair<Str, Bool> containing the generated code and if there is an equality
     * expression.
     */
    if (joinNode.getOperator() instanceof SqlBinaryOperator) {
      SqlOperator binOp = joinNode.getOperator();
      // Add () to ensure operator precedence
      String operator;
      switch (binOp.getKind()) {
        case EQUALS:
          operator = "==";
          break;
        case NOT_EQUALS:
          operator = "!=";
          break;
        case GREATER_THAN:
          operator = ">";
          break;
        case GREATER_THAN_OR_EQUAL:
          operator = ">=";
          break;
        case LESS_THAN:
          operator = "<";
          break;
        case LESS_THAN_OR_EQUAL:
          operator = "<=";
          break;
        case AND:
          operator = "&";
          break;
        case OR:
          operator = "|";
          break;
        case PLUS:
          operator = "+";
          break;
        case MINUS:
          operator = "-";
          break;
        case TIMES:
          operator = "*";
          break;
        case DIVIDE:
          operator = "/";
          break;
        default:
          throw new BodoSQLCodegenException(
              String.format(
                  "Internal Error: Call with operator %s not supported within a join condition.",
                  joinNode.getOperator()));
      }
      List<RexNode> operands = joinNode.getOperands();
      final Expr val1 = visitJoinCond(operands.get(0), leftColNames, rightColNames, mergeCols);

      // curExpr will be updated for each iteration of the loop
      // when the loop terminates, it will be an expression
      // equivalent to the full join call.
      Expr curExpr = val1;
      for (int i = 1; i < operands.size(); i++) {
        Expr val2 = visitJoinCond(operands.get(i), leftColNames, rightColNames, mergeCols);
        curExpr = new Expr.Binary(operator, curExpr, val2);

        // If we have an equality with two InputRefs that are the same
        // mark the columns as merged.
        if (binOp.getKind() == SqlKind.EQUALS
            && operands.get(i) instanceof RexInputRef
            && operands.get(0) instanceof RexInputRef) {
          String col1 = val1.emit().split("\\.")[1];
          // need to trim here to remove whitespace
          // TODO: remove technical debt.
          col1 = col1.substring(1, col1.length() - 1);
          String col2 = val2.emit().split("\\.")[1];
          // need to trim here to remove whitespace
          // TODO: remove technical debt.
          col2 = col2.substring(1, col2.length() - 1);
          if (col1.equals(col2)) {
            mergeCols.add(col1);
          }
        }
      }

      return curExpr;
    } else if (joinNode.getOperator() instanceof SqlPrefixOperator) {
      SqlOperator prefixOp = joinNode.getOperator();
      if (prefixOp.getKind() == SqlKind.NOT) {
        Expr val1 = visitJoinCond(joinNode.operands.get(0), leftColNames, rightColNames, mergeCols);
        Expr notCond = new Expr.Unary("~", val1);
        return notCond;
      }
    } else if (joinNode.getOperator() instanceof SqlPostfixOperator) {
      SqlOperator postfixOp = joinNode.getOperator();
      if (postfixOp.getKind() == SqlKind.IS_NOT_TRUE) {
        Expr val1 = visitJoinCond(joinNode.operands.get(0), leftColNames, rightColNames, mergeCols);
        Expr notCond = new Expr.Unary("~", val1);
        return notCond;
      }
    } else if (joinNode.getOperator() instanceof SqlFunction
        && joinNode.getOperator().toString().equals("POW")) {
      // TODO[BE-4274]: support all possible functions
      List<RexNode> operands = joinNode.getOperands();
      Expr val1 = visitJoinCond(operands.get(0), leftColNames, rightColNames, mergeCols);
      Expr val2 = visitJoinCond(operands.get(1), leftColNames, rightColNames, mergeCols);
      Expr powerExpr = new Expr.Call("pow", List.of(val1, val2), List.of());
      return powerExpr;
    }
    throw new BodoSQLCodegenException(
        String.format(
            "Internal Error: Call with operator %s not supported within a join condition.",
            joinNode.getOperator()));
  }

  /**
   * For a Bodo join lower two global variables, one for the build table and one for the probe table
   * containing the indices of the columns used for each comparison.
   *
   * @param info The join equality information.
   * @param visitor The visitor used for lowering global variables
   * @return A pair of variables that contain the selected indices.
   */
  public static Pair<Variable, Variable> getStreamingJoinKeyIndices(
      JoinInfo info, PandasCodeGenVisitor visitor) {
    // Define list of indices
    List<Integer> leftKeys = info.leftKeys;
    List<Integer> rightKeys = info.rightKeys;
    List<Expr.IntegerLiteral> leftIndices = new ArrayList<>();
    List<Expr.IntegerLiteral> rightIndices = new ArrayList<>();
    for (int i = 0; i < leftKeys.size(); i++) {
      leftIndices.add(new Expr.IntegerLiteral(leftKeys.get(i)));
      rightIndices.add(new Expr.IntegerLiteral(rightKeys.get(i)));
    }
    // Convert the lists to globals and return them.
    Variable leftVar = visitor.lowerAsMetaType(new Expr.Tuple(leftIndices));
    Variable rightVar = visitor.lowerAsMetaType(new Expr.Tuple(rightIndices));
    return new Pair<>(leftVar, rightVar);
  }
}
