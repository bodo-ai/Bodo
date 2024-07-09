package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.rules.SubstitutionRule;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.tools.RelBuilder;
import org.immutables.value.Value;

/**
 * Rule that takes a join between a Values with a single row and another input and converts it into
 * a projection of the other input with the values from the single row added as new columns.
 *
 * <p>If there is a join condition then this generates a filter on top of the projection assuming
 * it's not an outer join.
 */
@BodoSQLStyleImmutable
public class PruneSingleRowJoinRules {

  protected abstract static class PruneSingleRowJoinRule
      extends RelRule<PruneSingleRowJoinRule.Config> implements SubstitutionRule {
    protected PruneSingleRowJoinRule(PruneSingleRowJoinRule.Config config) {
      super(config);
    }

    @Override
    public boolean autoPruneOld() {
      return true;
    }

    /** Rule configuration. */
    public interface Config extends RelRule.Config {
      @Override
      PruneSingleRowJoinRule toRule();
    }
  }

  public static boolean isSingleRowValues(Values node) {
    return node.getTuples().size() == 1;
  }

  /** Configuration for rule that prunes a join if its left input is a single row. */
  @Value.Immutable
  public interface PruneLeftSingleRowJoinRuleConfig extends PruneSingleRowJoinRule.Config {
    PruneLeftSingleRowJoinRuleConfig DEFAULT =
        ImmutablePruneLeftSingleRowJoinRuleConfig.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Join.class)
                        .predicate(
                            join ->
                                join.getJoinType() == JoinRelType.INNER
                                    || (join.getJoinType() == JoinRelType.RIGHT
                                        && join.getCondition().isAlwaysTrue()))
                        .inputs(
                            b1 ->
                                b1.operand(Values.class)
                                    .predicate(v -> isSingleRowValues(v))
                                    .noInputs(),
                            b2 -> b2.operand(RelNode.class).anyInputs()))
            .withDescription("PruneSingleRowJoin(left)");

    @Override
    default PruneSingleRowJoinRule toRule() {
      return new PruneSingleRowJoinRule(this) {
        @Override
        public void onMatch(RelOptRuleCall call) {
          final Join join = call.rel(0);
          final Values left = call.rel(1);
          final RelNode right = call.rel(2);
          final RelBuilder relBuilder = call.builder();
          final RexBuilder rexBuilder = relBuilder.getRexBuilder();
          relBuilder.push(right);
          List<RexNode> projects = new ArrayList<>();
          // Append the values.
          ImmutableList<RexLiteral> row = left.getTuples().get(0);
          List<RelDataTypeField> fieldList = join.getRowType().getFieldList();
          for (int i = 0; i < row.size(); i++) {
            RexLiteral literal = row.get(i);
            RelDataType type = fieldList.get(i).getType();
            final RexNode value;
            if (literal.getType() != type) {
              // We may need to cast nullability.
              value = rexBuilder.makeCast(type, literal, true, false);
            } else {
              value = literal;
            }
            projects.add(value);
          }
          // Append the input refs
          int rightFieldCount = right.getRowType().getFieldCount();
          for (int i = 0; i < rightFieldCount; i++) {
            projects.add(relBuilder.field(i));
          }
          relBuilder.project(projects);
          RexNode joinCondition = join.getCondition();
          if (!joinCondition.isAlwaysTrue()) {
            relBuilder.filter(joinCondition);
          }
          RelNode newNode = relBuilder.build();
          call.transformTo(newNode);
        }
      };
    }
  }

  /** Configuration for rule that prunes a join if its right input is a single row. */
  @Value.Immutable
  public interface PruneRightSingleRowJoinRuleConfig extends PruneSingleRowJoinRule.Config {
    PruneRightSingleRowJoinRuleConfig DEFAULT =
        ImmutablePruneRightSingleRowJoinRuleConfig.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Join.class)
                        .predicate(
                            join ->
                                join.getJoinType() == JoinRelType.INNER
                                    || (join.getJoinType() == JoinRelType.LEFT
                                        && join.getCondition().isAlwaysTrue()))
                        .inputs(
                            b1 -> b1.operand(RelNode.class).anyInputs(),
                            b2 ->
                                b2.operand(Values.class)
                                    .predicate(v -> isSingleRowValues(v))
                                    .noInputs()))
            .withDescription("PruneSingleRowJoin(right)");

    @Override
    default PruneSingleRowJoinRule toRule() {
      return new PruneSingleRowJoinRule(this) {
        @Override
        public void onMatch(RelOptRuleCall call) {
          final Join join = call.rel(0);
          final RelNode left = call.rel(1);
          final Values right = call.rel(2);
          final RelBuilder relBuilder = call.builder();
          final RexBuilder rexBuilder = relBuilder.getRexBuilder();
          relBuilder.push(left);
          List<RexNode> projects = new ArrayList<>();
          // Append the input refs
          int leftFieldCount = left.getRowType().getFieldCount();
          for (int i = 0; i < leftFieldCount; i++) {
            projects.add(relBuilder.field(i));
          }
          // Append the values.
          ImmutableList<RexLiteral> row = right.getTuples().get(0);
          List<RelDataTypeField> fieldList = join.getRowType().getFieldList();
          int offset = left.getRowType().getFieldCount();
          for (int i = 0; i < row.size(); i++) {
            RexLiteral literal = row.get(i);
            RelDataType type = fieldList.get(i + offset).getType();
            final RexNode value;
            if (literal.getType() != type) {
              // We may need to cast nullability.
              value = rexBuilder.makeCast(type, literal, true, false);
            } else {
              value = literal;
            }
            projects.add(value);
          }
          relBuilder.project(projects);
          RexNode joinCondition = join.getCondition();
          if (!joinCondition.isAlwaysTrue()) {
            relBuilder.filter(joinCondition);
          }
          RelNode newNode = relBuilder.build();
          call.transformTo(newNode);
        }
      };
    }
  }
}
