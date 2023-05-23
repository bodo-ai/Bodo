package com.bodosql.calcite.application;

import static com.bodosql.calcite.application.Utils.Utils.escapePythonQuotes;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import java.util.HashSet;
import java.util.List;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexVisitorImpl;
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
 * <p>Returns a Pair<Str, Bool> containing the generated code and if there is an equality
 * expression.
 */
public class JoinCondVisitor {
  public static Pair<Expr, Boolean> visitJoinCond(
      RexNode joinNode,
      List<String> leftColNames,
      List<String> rightColNames,
      HashSet<String> mergeCols) {
    /** General function for join conditions. */
    Expr outputString;
    boolean hasEquals = false;
    if (joinNode instanceof RexInputRef) {
      outputString = visitJoinInputRef((RexInputRef) joinNode, leftColNames, rightColNames);
    } else if (joinNode instanceof RexLiteral) {
      outputString = visitJoinLiteral((RexLiteral) joinNode);
    } else if (joinNode instanceof RexCall) {
      Pair<Expr, Boolean> callResult =
          visitJoinCall((RexCall) joinNode, leftColNames, rightColNames, mergeCols);
      outputString = callResult.getKey();
      hasEquals = callResult.getValue();
    } else {
      throw new BodoSQLCodegenException(
          "Internal Error: Calcite Plan Produced an unsupported Join Condition");
    }
    return new Pair<>(outputString, hasEquals);
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

  public static Pair<Expr, Boolean> visitJoinCall(
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
      boolean hasEquals = false;
      SqlOperator binOp = joinNode.getOperator();
      // Add () to ensure operator precedence
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
      Pair<Expr, Boolean> val1Info =
          visitJoinCond(operands.get(0), leftColNames, rightColNames, mergeCols);
      final Expr val1 = val1Info.getKey();
      // OR is always treated as false because we can't handle it in the engine.
      if (binOp.getKind() != SqlKind.OR) {
        hasEquals = hasEquals || val1Info.getValue();
      }

      // curExpr will be updated for each iteration of the loop
      // when the loop terminates, it will be an expression
      // equivalent to the full join call.
      Expr curExpr = val1;
      for (int i = 1; i < operands.size(); i++) {
        Pair<Expr, Boolean> val2Info =
            visitJoinCond(operands.get(i), leftColNames, rightColNames, mergeCols);
        Expr val2 = val2Info.getKey();
        // OR is always treated as false because we can't handle it in the engine.
        if (binOp.getKind() != SqlKind.OR) {
          hasEquals = hasEquals || val2Info.getValue();
        }
        curExpr = new Expr.Binary(operator, curExpr, val2);

        // If we have an equality with two InputRefs that are the same
        // mark the columns as merged.
        if (binOp.getKind() == SqlKind.EQUALS && operands.get(i) instanceof RexInputRef) {
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

      return new Pair<>(curExpr, hasEquals);
    } else if (joinNode.getOperator() instanceof SqlPrefixOperator) {
      SqlOperator prefixOp = joinNode.getOperator();
      if (prefixOp.getKind() == SqlKind.NOT) {
        Pair<Expr, Boolean> val1Info =
            visitJoinCond(joinNode.operands.get(0), leftColNames, rightColNames, mergeCols);
        Expr notCond = new Expr.Unary("~", val1Info.getKey());
        return new Pair<>(notCond, val1Info.getValue());
      }
    } else if (joinNode.getOperator() instanceof SqlPostfixOperator) {
      SqlOperator postfixOp = joinNode.getOperator();
      if (postfixOp.getKind() == SqlKind.IS_NOT_TRUE) {
        Pair<Expr, Boolean> val1Info =
            visitJoinCond(joinNode.operands.get(0), leftColNames, rightColNames, mergeCols);
        Expr notCond = new Expr.Unary("~", val1Info.getKey());
        return new Pair<>(notCond, val1Info.getValue());
      }
    } else if (joinNode.getOperator() instanceof SqlFunction
        && joinNode.getOperator().toString().equals("POW")) {
      // TODO[BE-4274]: support all possible functions
      List<RexNode> operands = joinNode.getOperands();
      Pair<Expr, Boolean> val1Info =
          visitJoinCond(operands.get(0), leftColNames, rightColNames, mergeCols);
      Expr val1 = val1Info.getKey();
      Pair<Expr, Boolean> val2Info =
          visitJoinCond(operands.get(1), leftColNames, rightColNames, mergeCols);
      Expr val2 = val2Info.getKey();
      boolean hasEquals = val1Info.getValue() || val2Info.getValue();
      Expr powerExpr = new Expr.Call("pow", List.of(val1, val2), List.of());
      return new Pair<>(powerExpr, hasEquals);
    }
    throw new BodoSQLCodegenException(
        String.format(
            "Internal Error: Call with operator %s not supported within a join condition.",
            joinNode.getOperator()));
  }

  public static boolean enableStreamingJoin = false;

  /** Determine if a condition for a join equates to a HashJoin inside Bodo. */
  public static boolean isBodoHashJoin(RexNode condition, int numLeftCols) {
    if (!enableStreamingJoin) {
      return false;
    }
    Boolean result =
        condition.accept(
            // We don't need deep traversal
            new RexVisitorImpl<>(false) {
              @Override
              public Boolean visitCall(RexCall call) {
                /**
                 * We can only have a hash join if we either have an equality or AND of several
                 * equalities.
                 */
                if (call.getOperator() instanceof SqlBinaryOperator) {
                  if (call.getKind().equals(SqlKind.AND)) {
                    // And is true if any value is valid. Note we can't
                    // use visitArrayOr because it doesn't handle NULL properly
                    for (RexNode operand : call.getOperands()) {
                      Boolean b = operand.accept(this);
                      // Evaluate null as False
                      boolean isTrue = b != null && b;
                      if (isTrue) {
                        return true;
                      }
                    }
                  } else if (call.getKind().equals(SqlKind.EQUALS)) {
                    // Equals is valid if we have two RexInputRefs and they are from different
                    // tables
                    List<RexNode> operands = call.getOperands();
                    if (operands.size() == 2) {
                      RexNode first = operands.get(0);
                      RexNode second = operands.get(1);
                      if (first instanceof RexInputRef && second instanceof RexInputRef) {
                        RexInputRef firstRef = (RexInputRef) first;
                        RexInputRef secondRef = (RexInputRef) second;
                        int firstColNum = firstRef.getIndex();
                        int secondColNum = secondRef.getIndex();
                        // We have a valid hash join condition if they come from different tables.
                        return (firstColNum < numLeftCols && secondColNum >= numLeftCols)
                            || (firstColNum >= numLeftCols && secondColNum < numLeftCols);
                      }
                    }
                  }
                  // All other cases are not supported. If there is code like (A == B) OR (A == B)
                  // it will
                  // be simplified by the plan.
                }
                return false;
              }
            });

    return result != null && result;
  }
}
