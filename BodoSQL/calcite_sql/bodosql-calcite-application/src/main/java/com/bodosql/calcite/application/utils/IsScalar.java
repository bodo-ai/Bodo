package com.bodosql.calcite.application.utils;

import static org.apache.calcite.rex.RexVisitorImpl.visitArrayAnd;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexCorrelVariable;
import org.apache.calcite.rex.RexDynamicParam;
import org.apache.calcite.rex.RexFieldAccess;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLambda;
import org.apache.calcite.rex.RexLambdaRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexLocalRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexOver;
import org.apache.calcite.rex.RexPatternFieldRef;
import org.apache.calcite.rex.RexRangeRef;
import org.apache.calcite.rex.RexSubQuery;
import org.apache.calcite.rex.RexTableInputRef;
import org.apache.calcite.rex.RexVisitor;

public class IsScalar implements RexVisitor<Boolean> {

  @Override
  public Boolean visitInputRef(RexInputRef inputRef) {
    return false;
  }

  @Override
  public Boolean visitLocalRef(RexLocalRef localRef) {
    return false;
  }

  @Override
  public Boolean visitLiteral(RexLiteral literal) {
    return true;
  }

  @Override
  public Boolean visitOver(RexOver over) {
    return false;
  }

  @Override
  public Boolean visitCorrelVariable(RexCorrelVariable correlVariable) {
    throw unsupportedNode();
  }

  @Override
  public Boolean visitCall(RexCall call) {
    if (call.getOperator().getName().equals("RANDOM")) {
      return false;
    } else {
      return visitArrayAnd(this, call.operands);
    }
  }

  @Override
  public Boolean visitDynamicParam(RexDynamicParam dynamicParam) {
    return true;
  }

  @Override
  public Boolean visitRangeRef(RexRangeRef rangeRef) {
    throw unsupportedNode();
  }

  @Override
  public Boolean visitFieldAccess(RexFieldAccess fieldAccess) {
    // Field access is scalar if the reference expression is scalar as
    // we are always just selecting an element from the expression.
    return fieldAccess.getReferenceExpr().accept(this);
  }

  @Override
  public Boolean visitSubQuery(RexSubQuery subQuery) {
    throw unsupportedNode();
  }

  @Override
  public Boolean visitTableInputRef(RexTableInputRef ref) {
    throw unsupportedNode();
  }

  @Override
  public Boolean visitPatternFieldRef(RexPatternFieldRef fieldRef) {
    throw unsupportedNode();
  }

  @Override
  public Boolean visitLambda(RexLambda lambda) {
    throw unsupportedNode();
  }

  @Override
  public Boolean visitLambdaRef(RexLambdaRef lambdaRef) {
    throw unsupportedNode();
  }

  public static boolean isScalar(RexNode node) {
    return node.accept(new IsScalar());
  }

  protected BodoSQLCodegenException unsupportedNode() {
    return new BodoSQLCodegenException(
        "Internal Error: Calcite Plan Produced an Unsupported RexNode");
  }
}
