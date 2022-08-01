package com.bodosql.calcite.application;

import static com.bodosql.calcite.application.Utils.Utils.escapePythonQuotes;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import java.util.HashSet;
import java.util.List;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlBinaryOperator;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
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
 * <p>Returns a Pair<Str, Bool> containing the generated code and if there is an equality
 * expression.
 */
public class JoinCondVisitor {
  public static Pair<String, Boolean> visitJoinCond(
      RexNode joinNode,
      List<String> leftColNames,
      List<String> rightColNames,
      HashSet<String> mergeCols) {
    /** General function for join conditions. */
    String outputString;
    boolean hasEquals = false;
    if (joinNode instanceof RexInputRef) {
      outputString = visitJoinInputRef((RexInputRef) joinNode, leftColNames, rightColNames);
    } else if (joinNode instanceof RexLiteral) {
      outputString = visitJoinLiteral((RexLiteral) joinNode);
    } else if (joinNode instanceof RexCall) {
      Pair<String, Boolean> callResult =
          visitJoinCall((RexCall) joinNode, leftColNames, rightColNames, mergeCols);
      outputString = callResult.getKey();
      hasEquals = callResult.getValue();
    } else {
      throw new BodoSQLCodegenException(
          "Internal Error: Calcite Plan Produced an unsupported Join Condition");
    }
    return new Pair<>(outputString, hasEquals);
  }

  public static String visitJoinInputRef(
      RexInputRef joinNode, List<String> leftColNames, List<String> rightColNames) {
    /**
     * Visit node for an input ref within a join condition. This should output either left.`column`
     * or right.`column`. We use ` ` to avoid non-identifier column issues.
     */
    int colNum = joinNode.getIndex();
    if (colNum < leftColNames.size()) {
      String leftColName = leftColNames.get(colNum);
      leftColName = "`" + leftColName + "`";
      return String.format("left.%s", leftColName);
    } else {
      String rightColName = rightColNames.get(colNum - leftColNames.size());
      rightColName = "`" + rightColName + "`";
      return String.format("right.%s", rightColName);
    }
  }

  public static String visitJoinLiteral(RexLiteral joinNode) {
    /**
     * Visit node for a literal within a join condition. This should output the literal value for
     * literals that can be supported within a join condition.
     */
    SqlTypeName typeName = joinNode.getType().getSqlTypeName();
    switch (typeName) {
      case TINYINT:
      case SMALLINT:
      case INTEGER:
      case BIGINT:
      case FLOAT:
      case REAL:
      case DOUBLE:
      case DECIMAL:
        return joinNode.getValue().toString();
      case CHAR:
      case VARCHAR:
        // extract value without specific sql type info.
        return makeQuoted(escapePythonQuotes(joinNode.getValue2().toString()));
      case BOOLEAN:
        String boolName = joinNode.toString();
        return boolName.substring(0, 1).toUpperCase() + boolName.substring(1);
      default:
        throw new BodoSQLCodegenException(
            String.format(
                "Internal Error: Literal Type %s not supported within a join condition.",
                typeName));
    }
  }

  public static Pair<String, Boolean> visitJoinCall(
      RexCall joinNode,
      List<String> leftColNames,
      List<String> rightColNames,
      HashSet<String> mergeCols) {
    /**
     * Visit node for a call within a join condition. This should be limited to logical operators,
     * comparison operators (excluding <=>), and arithmetic operators.
     *
     * <p>Returns a Pair<Str, Bool> containing the generated code and if there is an equality
     * expression.
     */
    if (joinNode.getOperator() instanceof SqlBinaryOperator) {
      boolean hasEquals = false;
      SqlOperator binOp = joinNode.getOperator();
      StringBuilder binOpCond = new StringBuilder();
      // Add () to ensure operator precedence
      binOpCond.append("(");
      String operator;
      switch (binOp.getKind()) {
        case EQUALS:
          operator = "==";
          hasEquals = true;
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
        default:
          throw new BodoSQLCodegenException(
              String.format(
                  "Internal Error: Call with operator %s not supported within a join condition.",
                  joinNode.getOperator()));
      }
      List<RexNode> operands = joinNode.getOperands();
      Pair<String, Boolean> val1Info =
          visitJoinCond(operands.get(0), leftColNames, rightColNames, mergeCols);
      String val1 = val1Info.getKey();
      // OR is always treated as false because we can't handle it in the engine.
      if (binOp.getKind() != SqlKind.OR) {
        hasEquals = hasEquals || val1Info.getValue();
      }
      binOpCond.append(val1);
      for (int i = 1; i < operands.size(); i++) {
        binOpCond.append(" ").append(operator).append(" ");
        Pair<String, Boolean> val2Info =
            visitJoinCond(operands.get(i), leftColNames, rightColNames, mergeCols);
        String val2 = val2Info.getKey();
        // OR is always treated as false because we can't handle it in the engine.
        if (binOp.getKind() != SqlKind.OR) {
          hasEquals = hasEquals || val2Info.getValue();
        }
        binOpCond.append(val2);
        // If we have an equality with two InputRefs that are the same
        // mark the columns as merged.
        if (binOp.getKind() == SqlKind.EQUALS && operands.get(i) instanceof RexInputRef) {
          String col1 = val1.split("\\.")[1];
          // need to trim here to remove whitespace
          // TODO: remove technical debt.
          col1 = col1.substring(1, col1.length() - 1);
          String col2 = val2.split("\\.")[1];
          // need to trim here to remove whitespace
          // TODO: remove technical debt.
          col2 = col2.substring(1, col2.length() - 1);
          if (col1.equals(col2)) {
            mergeCols.add(col1);
          }
        }
        val1 = val2;
      }
      binOpCond.append(")");
      return new Pair<>(binOpCond.toString(), hasEquals);
    } else if (joinNode.getOperator() instanceof SqlPrefixOperator) {
      SqlOperator prefixOp = joinNode.getOperator();
      if (prefixOp.getKind() == SqlKind.NOT) {
        Pair<String, Boolean> val1Info =
            visitJoinCond(joinNode.operands.get(0), leftColNames, rightColNames, mergeCols);
        String notCond = "(~" + val1Info.getKey() + ")";
        return new Pair<>(notCond, val1Info.getValue());
      }
    }
    throw new BodoSQLCodegenException(
        String.format(
            "Internal Error: Call with operator %s not supported within a join condition.",
            joinNode.getOperator()));
  }
}
