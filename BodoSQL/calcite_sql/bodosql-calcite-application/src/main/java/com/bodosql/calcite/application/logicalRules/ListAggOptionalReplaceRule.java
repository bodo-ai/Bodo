package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilder;
import org.immutables.value.Value;

/**
 * Rule that converts LISTAGG(A) calls to the equivalent expression LISTAGG(A, ''). This is needed
 * because the LISTAGG implementation currently requires the optional separator argument to be a
 * column instead of a scalar. Therefore, this rule is required in order to produce a calcite plan
 * where the separator is passed into the aggregate node as a column.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class ListAggOptionalReplaceRule extends RelRule<ListAggOptionalReplaceRule.Config>
    implements TransformationRule {

  /** Creates a ListAggOptionalReplaceRule. */
  protected ListAggOptionalReplaceRule(ListAggOptionalReplaceRule.Config config) {
    super(config);
  }

  private boolean needsTransform(List<AggregateCall> aggregateCallList) {
    for (int i = 0; i < aggregateCallList.size(); i++) {
      AggregateCall curCall = aggregateCallList.get(i);
      if (curCall.getAggregation().getKind() == SqlKind.LISTAGG
          && curCall.getArgList().size() < 2) {
        return true;
      }
    }
    return false;
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Aggregate origAggRel = call.rel(0);
    final RelNode origInput = origAggRel.getInput();
    final RexBuilder rexBuilder = origAggRel.getCluster().getRexBuilder();
    final RelBuilder relBuilder = call.builder();

    if (!needsTransform(origAggRel.getAggCallList())) {
      return;
    }

    int emptyStrIdx = origInput.getRowType().getFieldCount();

    final List<RexNode> newProjects = new ArrayList<>();
    for (int i = 0; i < emptyStrIdx; i++) {
      RelDataType typ = origInput.getRowType().getFieldList().get(i).getType();
      newProjects.add(new RexInputRef(i, typ));
    }
    newProjects.add(rexBuilder.makeLiteral(""));

    RelNode newInput = relBuilder.push(origAggRel.getInput()).project(newProjects).build();

    List<AggregateCall> origAggregateCallList = origAggRel.getAggCallList();
    List<AggregateCall> newAggregateCallList = new ArrayList<>();

    for (int i = 0; i < origAggRel.getAggCallList().size(); i++) {
      AggregateCall origAggCall = origAggregateCallList.get(i);
      if (origAggCall.getAggregation().getKind() == SqlKind.LISTAGG
          && origAggCall.getArgList().size() < 2) {
        List<Integer> newArgsList = new ArrayList<>(origAggCall.getArgList());
        newArgsList.add(emptyStrIdx);
        newAggregateCallList.add(origAggCall.copy(newArgsList));
      } else {
        newAggregateCallList.add(origAggCall);
      }
    }

    Aggregate newAggRel =
        origAggRel.copy(
            origAggRel.getTraitSet(),
            newInput,
            origAggRel.getGroupSet(),
            origAggRel.getGroupSets(),
            newAggregateCallList);

    call.transformTo(newAggRel);
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    ListAggOptionalReplaceRule.Config DEFAULT =
        ImmutableListAggOptionalReplaceRule.Config.of().withOperandFor(Aggregate.class);

    @Override
    default ListAggOptionalReplaceRule toRule() {
      return new ListAggOptionalReplaceRule(this);
    }

    /** Defines an operand tree for the given 2 classes. */
    default ListAggOptionalReplaceRule.Config withOperandFor(
        Class<? extends Aggregate> aggregateClass) {
      // Bodo Change: Add a requirement that there are no window functions in the filter.
      return withOperandSupplier(b0 -> b0.operand(aggregateClass).anyInputs())
          .as(ListAggOptionalReplaceRule.Config.class);
    }
  }
}
