package com.bodosql.calcite.application.bodo_sql_rules;

import static com.bodosql.calcite.application.bodo_sql_rules.FilterRulesCommon.rexNodeContainsCase;

import com.bodosql.calcite.application.Utils.BodoSQLStyleImmutable;
import java.util.*;
import org.apache.calcite.plan.*;
import org.apache.calcite.rel.*;
import org.apache.calcite.rel.core.*;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.rex.*;
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

  private static RexNode updateRexNode(
      RexNode node, Map<RexNode, Integer> caseMap, RelBuilder builder) {
    // XXX: In the future we can replace this with RexVisitorImpl
    if (node instanceof RexCall) {
      RexCall callNode = ((RexCall) node);
      // If we find a case statement replace it with its
      // new RexInputRef.
      if (callNode.getOperator() instanceof SqlCaseOperator) {
        return builder.field(caseMap.get(callNode));
      }
      // Call expressions need to have their children traversed. No other RexNodes
      // should have children.
      List<RexNode> oldOperands = callNode.getOperands();
      List<RexNode> newOperands = new ArrayList<>();
      boolean replaceNode = false;
      for (RexNode oldOperand : oldOperands) {
        RexNode newOperand = updateRexNode(oldOperand, caseMap, builder);
        newOperands.add(newOperand);
        replaceNode = replaceNode || !newOperand.equals(oldOperand);
      }
      if (replaceNode) {
        if (callNode.getOperator() instanceof SqlCastFunction) {
          // TODO: Replace with a simpler cast API instead in Calcite
          return builder
              .getCluster()
              .getRexBuilder()
              .makeCast(callNode.getType(), newOperands.get(0));
        } else {
          return builder.call(callNode.getOperator(), newOperands);
        }
      }
    }
    return node;
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
        ImmutableFilterExtractCaseRule.Config.of().withOperandFor(Filter.class);

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
