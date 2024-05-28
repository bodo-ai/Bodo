package org.apache.calcite.plan;

import java.util.Stack;
import org.apache.calcite.plan.volcano.RelSubset;
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
import org.apache.calcite.util.Pair;
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

  /**
   * Determine if there is a cycle in a RelNode.
   * This is intended for use with the volcano planner,
   * so it supports RelSubset.
   */
  public static class CycleFinder {

    // Make the constructor private so no one creates an instance of this class.
    // We can't cache results because especially with RelSubset, the plan can change.
    private CycleFinder() {}
    /**
     * Check for a cycle by implementing an iterative version of DFS.
     * @param p The root of the RelNode tree.
     * @return True if a cycle is found, false otherwise.
     */
    public static boolean determineCycle(RelNode p) {
      final Set<Integer> startedRelNodes = new HashSet();
      final Set<Integer> finishedRelNodes = new HashSet();
      // Keep a stack of both the upcoming nodes and the nodes that have been visited.
      // Here True indicates a second visit to the node, whereas False means this is our
      // first visit.
      Stack<Pair<RelNode, Boolean>> nodeStack = new Stack<>();
      nodeStack.add(new Pair(p, false));
      while (!nodeStack.isEmpty()) {
        Pair<RelNode, Boolean> pair = nodeStack.pop();
        RelNode node = pair.left;
        Boolean secondVisit = pair.right;
        if (finishedRelNodes.contains(node.getId())) {
          // Do nothing if we already finished this node.
        } else if (secondVisit) {
          // Mark this node as finished. It does not contain a cycle.
          finishedRelNodes.add(node.getId());
          startedRelNodes.remove(node.getId());
        } else if (startedRelNodes.contains(node.getId())) {
          // We found a cycle.
          return true;
        } else {
          startedRelNodes.add(node.getId());
          // Add the node so the LIFO order can be used to detect cycles.
          nodeStack.add(new Pair(node, true));
          if (node instanceof RelSubset) {
            RelSubset s = (RelSubset) node;
            RelNode best = s.getBest();
            if (best != null && !finishedRelNodes.contains(best.getId())) {
              nodeStack.add(new Pair(best, false));
            }
          } else {
            for (RelNode input : node.getInputs()) {
              if (!finishedRelNodes.contains(input.getId())) {
                nodeStack.add(new Pair(input, false));
              }
            }
          }
        }
      }
      return false;
    }
  }
}
