package org.apache.calcite.plan;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelShuttleImpl;
import org.apache.calcite.rel.RelVisitor;
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.logical.LogicalCalc;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rex.RexCorrelVariable;
import org.apache.calcite.rex.RexFieldAccess;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexSubQuery;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.rex.RexVisitorImpl;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql2rel.SqlToRelConverter;

import com.bodosql.calcite.sql.func.SqlBodoOperatorTable;
import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.HashSet;
import java.util.Set;

public class BodoRelOptUtil {
  /**
   * Returns {@link SqlOperator} for given {@link SqlKind} or returns {@code operator}
   * when {@link SqlKind} is not known.
   * <p>
   * TODO(jsternberg): This method is mostly identical to {@link RelOptUtil#op}
   * but includes support for {@link SqlBodoOperatorTable#NULL_EQUALS}.
   * <p>
   * The proper way to support this would be to give NULL_EQUALS its own
   * custom {@link org.apache.calcite.sql.fun.SqlQuantifyOperator} so we
   * don't have to modify the upstream version and then use convertlets
   * to perform the conversion similar to {@link com.bodosql.calcite.sql.func.SqlLikeQuantifyOperator}.
   * That's not necessarily straightforward and easy right now so copying
   * this method and utilizing it in {@link SqlToRelConverter} is more
   * straightforward.
   *
   * @param kind input kind
   * @param operator default operator value
   * @return SqlOperator for the given kind
   * @see RexUtil#op(SqlKind)
   */
  public static SqlOperator op(SqlKind kind, SqlOperator operator) {
    switch (kind) {
    case EQUALS:
      return SqlStdOperatorTable.EQUALS;
    case NOT_EQUALS:
      return SqlStdOperatorTable.NOT_EQUALS;
    case NULL_EQUALS:
      return SqlBodoOperatorTable.NULL_EQUALS;
    case GREATER_THAN:
      return SqlStdOperatorTable.GREATER_THAN;
    case GREATER_THAN_OR_EQUAL:
      return SqlStdOperatorTable.GREATER_THAN_OR_EQUAL;
    case LESS_THAN:
      return SqlStdOperatorTable.LESS_THAN;
    case LESS_THAN_OR_EQUAL:
      return SqlStdOperatorTable.LESS_THAN_OR_EQUAL;
    case IS_DISTINCT_FROM:
      return SqlStdOperatorTable.IS_DISTINCT_FROM;
    case IS_NOT_DISTINCT_FROM:
      return SqlStdOperatorTable.IS_NOT_DISTINCT_FROM;
    default:
      return operator;
    }
  }

  /**
   * Find all correlation variables that are used in a RexNode.
   */
  public static class CorrelationRexFinder extends RexVisitorImpl<Void> {

    private Set<CorrelationId> seenIds;
    public CorrelationRexFinder() {
      super(true);
      seenIds = new HashSet<>();
    }

    @Override public Void visitSubQuery(RexSubQuery subQuery) {
      // Iterate over the operands
      for (RexNode operand : subQuery.operands) {
        operand.accept(this);
      }
      // Iterate over the body of the sub query.
      subQuery.rel.accept(new CorrelationRelFinder(this));
      return null;
    }

    @Override
    public Void visitCorrelVariable(RexCorrelVariable correlVariable) {
      seenIds.add(correlVariable.id);
      return null;
    }

    public Set<CorrelationId> getSeenIds() {
      return seenIds;
    }
  }

  /**
   * Find all correlation variables that are used in a plan.
   * This applies the CorrelationRexFinder to a Filter,
   * Projection, or Join. If additional nodes can accept correlation
   * variables this should be updated.
   */
  public static class CorrelationRelFinder extends RelShuttleImpl {

    private final CorrelationRexFinder rexFinder;
    CorrelationRelFinder(CorrelationRexFinder rexFinder) {
      this.rexFinder = rexFinder;
    }

    @Override public RelNode visit(LogicalFilter filter) {
      filter.getCondition().accept(rexFinder);
      return super.visit(filter);
    }

    @Override public RelNode visit(LogicalProject project) {
      project.getProjects().stream().forEach(x -> x.accept(rexFinder));
      return super.visit(project);
    }

    @Override public RelNode visit(LogicalJoin join) {
      if (join.getCondition() != null) {
        join.getCondition().accept(rexFinder);
      }
      return super.visit(join);
    }
  }
}
