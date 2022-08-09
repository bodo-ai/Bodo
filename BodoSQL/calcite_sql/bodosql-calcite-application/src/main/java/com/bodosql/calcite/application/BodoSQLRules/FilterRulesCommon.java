package com.bodosql.calcite.application.BodoSQLRules;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.Pair;

/** Class containing common static helpers shared by 2 or Rules involving filters. */
public class FilterRulesCommon {

  /**
   * Determines if the given filter RexNode contains an OR. This is used to determine if certain
   * rule optimizations should be attempted.
   *
   * @param cond The RexNode to check for OR.
   * @return If cond contains an OR.
   */
  public static boolean filterContainsOr(RexNode cond) {
    if (cond.getKind() == SqlKind.OR) {
      return true;
    } else if (cond.getKind() == SqlKind.AND) {
      RexCall andCond = (RexCall) cond;
      for (int i = 0; i < andCond.operands.size(); i++) {
        if (filterContainsOr(andCond.operands.get(i))) {
          return true;
        }
      }
    }
    // If the expression is not an AND or an OR it cannot
    // contain an OR
    return false;
  }

  /**
   * Takes a RexNode with the common operations and returns a refactored RexNode extracting the
   * expression from commonOPs. Since the shared expressions change upon reaching each OR, this
   * function is applied to the subtree containing an OR. If a node is unchanged the original node
   * may be returned.
   *
   * <p>For example if the expressions was: OR( AND(A > 1, B < 2), AND(A > 1, A < 10), ) Then the
   * second argument should be the specified common operation to extract, which would be A > 1.
   *
   * <p>Then this function returns the result of rewriting the expression with this pruning, which
   * for this example would be AND(A > 1, OR(B < 2, A < 10)).
   *
   * @param builder Rexbuilder that creates the new RexNodes when the code is changed.
   * @param cond RexNode that corresponds to a subtree of the condition to update.
   * @param commonOps Hashmap containing the common operations that should be extracted.
   * @return The RexNode after pruning common expressions.
   */
  private static RexNode pruneConditionExtractCommon(
      RelBuilder builder, RexNode cond, HashSet<RexNode> commonOps) {
    if (cond.getKind() == SqlKind.AND) {
      List<RexNode> operands = new ArrayList<>();
      RexCall callCond = (RexCall) cond;
      for (int i = 0; i < callCond.operands.size(); i++) {
        RexNode newNode = pruneConditionExtractCommon(builder, callCond.operands.get(i), commonOps);
        if (newNode != null) {
          operands.add(newNode);
        }
      }
      if (operands.size() > 0) {
        // AND together operands
        return builder.and(operands);
      } else {
        // If no nodes are kept return null. We will prune
        // the whole condition.
        return null;
      }
    } else if (commonOps.contains(cond)) {
      return null;
    } else {
      return cond;
    }
  }

  /**
   * Takes a RexNode contain that is part of a conditional expression containing at least 1 OR
   * (either as a child or a parent) and extracts any expressions that can be an AND across all
   * parts of the OR. Once those are found (if any), the RexNode is rewritten to prune those
   * expressions from the original condition and converted to an AND of the common expressions with
   * the remaining OR.
   *
   * <p>For example if the expressions was: OR( AND(A > 1, B < 2), AND(A > 1, A < 10), )
   *
   * <p>Then this would be converted to AND( A > 1, OR(B < 2, A < 10) )
   *
   * <p>If possible this will remove the OR. An AND expression may also be updated either adding or
   * removing parts.
   *
   * @param builder Rexbuilder that creates the new RexNodes when the code is changed.
   * @param cond RexNode that corresponds to a subtree of the condition.
   * @param commonOps Hashmap that is filled in this function to track all RexNodes that are shared
   *     with a subexpression (AND/OR). To enable nested ANDs/ORs (if necessary), this map is
   *     recreated at the start of each OR expression and the result is passed back to the original
   *     map.
   * @return A pair of values containing the new condition RexNode and if it was changed.
   */
  public static Pair<RexNode, Boolean> updateConditionsExtractCommon(
      RelBuilder builder, RexNode cond, HashSet<RexNode> commonOps) {
    List<RexNode> nodes = new ArrayList<>();
    if (cond.getKind() == SqlKind.AND) {
      // If we have an AND we UNION together any of
      // the binary operators
      RexCall andCond = (RexCall) cond;
      boolean changed = false;
      for (int i = 0; i < andCond.operands.size(); i++) {
        Pair<RexNode, Boolean> nodePair =
            updateConditionsExtractCommon(builder, andCond.operands.get(i), commonOps);
        changed = nodePair.getValue() || changed;
        nodes.add(nodePair.getKey());
      }
      if (changed) {
        // If the result has been changed, we create a new AND that comes from the
        // result of the combining the children of this node.
        return new Pair<>(builder.and(nodes), true);
      } else {
        return new Pair<>(cond, false);
      }

    } else if (cond.getKind() == SqlKind.OR) {
      // If we have an OR we INSERT together the subexpressions
      // from each child. Any binary expression that remains can be
      // removed from the OR and we can rewrite the whole RexNode
      // as AND(saved, PRUNE_OR).
      RexCall orCond = (RexCall) cond;
      HashSet<RexNode> sharedOps = new HashSet<>();
      Pair<RexNode, Boolean> nodePair =
          updateConditionsExtractCommon(builder, orCond.operands.get(0), sharedOps);
      nodes.add(nodePair.getKey());
      for (int i = 1; i < orCond.operands.size(); i++) {
        HashSet<RexNode> otherOps = new HashSet<>();
        nodePair = updateConditionsExtractCommon(builder, orCond.operands.get(i), otherOps);
        nodes.add(nodePair.getKey());
        sharedOps.retainAll(otherOps);
      }
      if (!sharedOps.isEmpty()) {
        // Update the ops for recursion
        commonOps.addAll(sharedOps);
        // Prune the existing RexNode
        List<RexNode> keptNodes = new ArrayList<>();
        for (RexNode node : nodes) {
          RexNode updatedNode = pruneConditionExtractCommon(builder, node, sharedOps);
          if (updatedNode != null) {
            keptNodes.add(updatedNode);
          }
        }
        List<RexNode> totalNodes = new ArrayList<>();
        // Create an OR with the leftover nodes
        if (nodes.size() > 0) {
          totalNodes.add(builder.or(keptNodes));
        }
        // AND together the remaining conditions
        for (RexNode commonCond : sharedOps) {
          totalNodes.add(commonCond);
        }
        return new Pair<>(builder.and(totalNodes), true);
      } else {
        return new Pair<>(cond, false);
      }
    } else {
      // If we any other node (BINOP, single columns, etc) then
      // we add to the set.
      commonOps.add(cond);
      return new Pair<>(cond, false);
    }
  }
}
