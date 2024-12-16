package com.bodosql.calcite.application.logicalRules;

import static com.bodosql.calcite.application.logicalRules.FilterRulesCommon.rexNodeContainsCase;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.FilterBase;
import java.util.*;
import org.apache.calcite.plan.*;
import org.apache.calcite.rel.*;
import org.apache.calcite.rel.core.*;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.rex.*;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.*;
import org.apache.calcite.tools.*;
import org.immutables.value.*;

/**
 * Rule that takes a filter that contains 1 or more case statements in the condition and replaces
 * the filter with a projection that computes all of the case statements. This transforms a filter
 * into a project then filter then projection. For example assume this filter is called on a table
 * with two columns.
 *
 * <p>Filter(cond=[(Case when $0 then 1 else 2 end) = 1])
 *
 * <p>This becomes
 *
 * <p>Project(col0=$0, col1=$1)
 *
 * <p>Filter(cond=[$2=1])
 *
 * <p>Project(col0=$0, col1=$1, $f2=Case when $0 then 1 else 2 end)
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class FilterExtractCaseRule<C extends BodoSQLReduceExpressionsRule.Config>
    extends RelRule<FilterExtractCaseRule.Config> implements TransformationRule {

  /** Creates a FilterExtractCaseRule. */
  protected FilterExtractCaseRule(FilterExtractCaseRule.Config config) {
    super(config);
  }

  /**
   * Returns if the filter contains a case statement. This is used for selecting this rule.
   *
   * @param filter The filter
   * @return Does it contain 1 or more cases in its condition.
   */
  private static boolean filterContainsCase(Filter filter) {
    return rexNodeContainsCase(filter.getCondition());
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Filter filter = call.rel(0);
    call.transformTo(apply(call, filter));
  }

  private static List<RexNode> getConditionCaseStatement(RexNode node) {
    List<RexNode> caseStatements = new ArrayList<>();
    node.accept(
        new RexVisitorImpl<Void>(true) {
          @Override
          public Void visitCall(RexCall call) {
            if (call.getOperator() instanceof SqlCaseOperator) {
              caseStatements.add(call);
            }
            // Note: We may not want to visit the child for nested case
            // statements, but in the worst case that should just generate
            // an extra column that will be pruned later.
            return super.visitCall(call);
          }
        });
    return caseStatements;
  }

  /**
   * Implementation of RexVisitorImpl that uses the case map and builder to remove case statements
   * from a filter node.
   */
  private static class ReplaceCase extends RexShuttle {
    private final Map<RexNode, Integer> caseMap;
    private final RelBuilder builder;

    ReplaceCase(Map<RexNode, Integer> caseMap, RelBuilder builder) {
      super();
      this.caseMap = caseMap;
      this.builder = builder;
    }

    @Override
    public RexNode visitOver(RexOver over) {
      RexWindow window = over.getWindow();
      // Boolean to act as a pointer to indicate if our children have changed.
      boolean[] update = {false};
      List<RexNode> newPartitionKeys = visitList(window.partitionKeys, update);
      // Accumulate the order by Keys to recurse on.
      int orderKeySize = window.orderKeys.size();
      List<RexNode> oldKeyRexNodes = new ArrayList(orderKeySize);
      for (int i = 0; i < orderKeySize; i++) {
        oldKeyRexNodes.add(i, window.orderKeys.get(i).getKey());
      }
      List<RexNode> newKeyRexNodes = visitList(oldKeyRexNodes, update);
      if (!update[0]) {
        return over;
      }
      // Ensure the final result contains the correct collation information.
      List<RexNode> newOrderKeys = new ArrayList<>(orderKeySize);
      for (int i = 0; i < orderKeySize; i++) {
        RexNode child = newKeyRexNodes.get(i);
        Set<SqlKind> kinds = window.orderKeys.get(i).getValue();
        if (kinds.contains(SqlKind.NULLS_FIRST)) {
          child = builder.nullsFirst(child);
        }
        if (kinds.contains(SqlKind.NULLS_LAST)) {
          child = builder.nullsLast(child);
        }
        if (kinds.contains(SqlKind.DESCENDING)) {
          child = builder.desc(child);
        }
        newOrderKeys.add(child);
      }
      RelBuilder.OverCall baseNode =
          builder
              .aggregateCall(over.getAggOperator(), over.getOperands())
              .distinct(over.isDistinct())
              .ignoreNulls(over.ignoreNulls())
              .over()
              .partitionBy(newPartitionKeys)
              .orderBy(newOrderKeys);
      if (window.isRows()) {
        return baseNode.rowsBetween(window.getLowerBound(), window.getUpperBound()).toRex();
      } else {
        return baseNode.rangeBetween(window.getLowerBound(), window.getUpperBound()).toRex();
      }
    }

    @Override
    public RexNode visitCall(RexCall call) {
      if (call.getOperator() instanceof SqlCaseOperator) {
        return builder.field(caseMap.get(call));
      }
      // Boolean to act as a pointer to indicate if our children have changed.
      boolean[] update = {false};
      List<RexNode> newOperands = visitList(call.getOperands(), update);
      if (!update[0]) {
        return call;
      }
      if (call.getOperator() instanceof SqlCastFunction) {
        // TODO: Replace with a simpler cast API instead in Calcite
        return builder.getCluster().getRexBuilder().makeCast(call.getType(), newOperands.get(0));
      } else {
        return builder.call(call.getOperator(), newOperands);
      }
    }
  }

  private static RexNode updateRexNode(
      RexNode node, Map<RexNode, Integer> caseMap, RelBuilder builder) {
    // XXX: In the future we can replace this with RexVisitorImpl
    ReplaceCase visitor = new ReplaceCase(caseMap, builder);
    return node.accept(visitor);
  }

  private static RexNode generateFilterCondition(
      RexNode cond, List<RexNode> caseStatements, int originalSize, RelBuilder builder) {
    // Create a map from each case statement to its inputRef column number.
    Map<RexNode, Integer> caseMap = new HashMap<>();
    for (int i = 0; i < caseStatements.size(); i++) {
      caseMap.put(caseStatements.get(i), originalSize + i);
    }
    return updateRexNode(cond, caseMap, builder);
  }

  private static RelNode apply(RelOptRuleCall call, Filter filter) {
    RelBuilder builder = call.builder();
    RelDataType originalType = filter.getRowType();
    List<String> fieldNames = originalType.getFieldNames();
    // Push the input so we can generate a new projection.
    builder.push(filter.getInput());
    // Step 1: Generate a new projection under the filter. This needs to contain
    // all nodes from the input and any case statements.
    RexNode condition = filter.getCondition();
    List<RexNode> caseStatements = getConditionCaseStatement(condition);
    List<RexNode> allColumns = new ArrayList<>();
    for (int i = 0; i < fieldNames.size(); i++) {
      allColumns.add(builder.field(i));
    }
    for (RexNode caseStatement : caseStatements) {
      allColumns.add(caseStatement);
    }
    builder.project(allColumns);
    // Step 2: Generate a new filter by replacing each of the case statements
    // with the new input refs.
    RexNode newCondition =
        generateFilterCondition(filter.getCondition(), caseStatements, fieldNames.size(), builder);
    builder.filter(newCondition);
    // Step 3: Generate a new final projection with the original input types.
    List<RexNode> inputRefs = new ArrayList<>();
    for (int i = 0; i < fieldNames.size(); i++) {
      inputRefs.add(builder.field(i));
    }
    builder.project(inputRefs, fieldNames);
    return builder.build();
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    FilterExtractCaseRule.Config DEFAULT =
        ImmutableFilterExtractCaseRule.Config.of().withOperandFor(FilterBase.class);

    @Override
    default FilterExtractCaseRule toRule() {
      return new FilterExtractCaseRule(this);
    }

    /** Defines an operand tree for the given classes. */
    default FilterExtractCaseRule.Config withOperandFor(Class<? extends Filter> filterClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(filterClass)
                      .predicate(FilterExtractCaseRule::filterContainsCase)
                      .anyInputs())
          .as(FilterExtractCaseRule.Config.class);
    }
  }
}
