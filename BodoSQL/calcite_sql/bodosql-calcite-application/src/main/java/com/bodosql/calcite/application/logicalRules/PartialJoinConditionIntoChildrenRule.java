package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoJoinConditionUtil;
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.logical.BodoLogicalJoin;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import kotlin.Pair;
import kotlin.Triple;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexPermuteInputsShuttle;
import org.apache.calcite.sql.validate.SqlValidatorUtil;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.mapping.Mappings;
import org.immutables.value.Value;

/**
 * Rule that converts a join with an unsupported condition into Join atop a projection that pushes
 * the additional compute. This rule should only fire if the join can be completely transformed into
 * a supported version.
 *
 * <p>For example this would take this join:
 *
 * <blockquote>
 *
 * <pre>
 *  Join(AND(=($1, $2), =(NULLIF($0, ''), NULLIF($3, ''))
 *    ...
 *    ...
 * </pre>
 *
 * </blockquote>
 *
 * and convert it into
 *
 * <blockquote>
 *
 * <pre>
 *  Project($0, $1, $3)
 *    Join(AND(=($1, $3), =($2, $4))
 *      Project($0, $1, NULLIF($0, ''))
 *        ...
 *      Project($0, NULLIF($0, ''))
 *        ...
 * </pre>
 *
 * </blockquote>
 *
 * This is done by finding the unsupported sections that operate on only 1 table and tranposing
 * those into an underlying projection and then updating the references.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class PartialJoinConditionIntoChildrenRule
    extends RelRule<PartialJoinConditionIntoChildrenRule.Config> implements TransformationRule {

  /** Creates a PartialJoinConditionIntoChildrenRule. */
  protected PartialJoinConditionIntoChildrenRule(
      PartialJoinConditionIntoChildrenRule.Config config) {
    super(config);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final RelBuilder builder = call.builder();
    final Join join = call.rel(0);
    final RelNode left = join.getLeft();
    final RelNode right = join.getRight();
    // Step 1: Build the information about the join sides.
    int leftFieldCount = left.getRowType().getFieldCount();
    int rightFieldCount = right.getRowType().getFieldCount();
    int totalFieldCount = leftFieldCount + rightFieldCount;
    Mappings.TargetMapping leftMapping = Mappings.createIdentity(leftFieldCount);
    Mappings.TargetMapping rightMapping =
        Mappings.createShiftMapping(totalFieldCount, 0, leftFieldCount, rightFieldCount);
    // Note: The left shuttle is for consistency and shouldn't be needed.
    final RexPermuteInputsShuttle leftShuttle = new RexPermuteInputsShuttle(leftMapping, left);
    final RexPermuteInputsShuttle rightShuttle = new RexPermuteInputsShuttle(rightMapping, right);
    // Step 2: Find the condition parts that will be pushed down.
    Triple<List<RexNode>, List<RexNode>, Map<RexNode, Pair<Boolean, Integer>>> pushInfo =
        BodoJoinConditionUtil.extractPushableFunctions(
            join.getCondition(), leftFieldCount, totalFieldCount, leftShuttle, rightShuttle);
    List<RexNode> leftColumns = pushInfo.component1();
    List<RexNode> rightColumns = pushInfo.component2();
    Map<RexNode, Pair<Boolean, Integer>> nodeCache = pushInfo.component3();
    if (leftColumns.isEmpty() && rightColumns.isEmpty()) {
      // If both sides are empty then we can't perform this transformation.
      // This can be reached if we have an inner join and need to "pull up"
      // the whole filter above the join.
      return;
    }
    // Step 3: Build the left side.
    final RelNode newLeft;
    if (!leftColumns.isEmpty()) {
      builder.push(left);
      List<RexNode> newFields = new ArrayList<>();
      for (int i = 0; i < leftFieldCount; i++) {
        newFields.add(builder.field(i));
      }
      newFields.addAll(leftColumns);
      newLeft = builder.project(newFields).build();
    } else {
      newLeft = left;
    }
    // Step 4: Build the right side.
    final RelNode newRight;
    if (!rightColumns.isEmpty()) {
      builder.push(right);
      List<RexNode> newFields = new ArrayList<>();
      for (int i = 0; i < rightFieldCount; i++) {
        newFields.add(builder.field(i));
      }
      newFields.addAll(rightColumns);
      newRight = builder.project(newFields).build();
    } else {
      newRight = right;
    }
    int leftAppendedFieldCount = leftFieldCount + leftColumns.size();
    // Step 5: Create a new shuttle and update the join condition + map InputRefs.
    // We have added new inputs and may therefore need to update the values of existing
    // input refs (although this should only be necessary if we update the left child).
    final Mappings.TargetMapping updatedRightMapping =
        Mappings.createShiftMapping(
            totalFieldCount + leftColumns.size() + rightColumns.size(),
            leftAppendedFieldCount,
            leftFieldCount,
            rightFieldCount);
    final Mappings.TargetMapping conditionMapping =
        Mappings.merge(leftMapping, updatedRightMapping);
    final RexPermuteInputsShuttle conditionShuttle =
        new RexPermuteInputsShuttle(conditionMapping, right);
    // Regenerate the condition
    final RexNode updatedCondition = join.getCondition().accept(conditionShuttle);
    // Regenerate the cache
    RelDataType inputNodeType =
        SqlValidatorUtil.createJoinType(
            builder.getTypeFactory(), newLeft.getRowType(), newRight.getRowType(), null, List.of());
    // Step 6: Replace the pushed down sections by preparing the arguments
    // for the RexReplacer and then calling it as a visitor.
    List<RexNode> oldExprs = new ArrayList();
    List<RexNode> newExprs = new ArrayList();
    List<Boolean> addCasts = new ArrayList();
    for (RexNode node : nodeCache.keySet()) {
      oldExprs.add(node.accept(conditionShuttle));
      Pair<Boolean, Integer> cachedInfo = nodeCache.get(node);
      boolean isRightTable = cachedInfo.component1();
      int tableIndex = cachedInfo.component2();
      final int index;
      if (isRightTable) {
        index = tableIndex + leftAppendedFieldCount;
      } else {
        index = tableIndex;
      }
      newExprs.add(
          builder
              .getRexBuilder()
              .makeInputRef(inputNodeType.getFieldList().get(index).getType(), index));
      addCasts.add(false);
    }
    final RexNode finalCondition =
        updatedCondition.accept(
            new BodoSQLReduceExpressionsRule.RexReplacer(
                builder.getRexBuilder(), oldExprs, newExprs, addCasts));
    // Step 7: Build the actual join.
    builder.push(newLeft);
    builder.push(newRight);
    builder.join(join.getJoinType(), finalCondition);
    // Step 8: Wrap the result in a final projection. This is necessary
    // to ensure the result is type stable.
    List<RexNode> finalFields = new ArrayList<>();
    for (int i = 0; i < leftFieldCount; i++) {
      finalFields.add(builder.field(i));
    }
    for (int i = 0; i < rightFieldCount; i++) {
      finalFields.add(builder.field(i + leftAppendedFieldCount));
    }
    builder.project(finalFields, join.getRowType().getFieldNames());
    RelNode result = builder.build();
    call.transformTo(result);
  }

  @Value.Immutable
  public interface Config extends RelRule.Config {
    PartialJoinConditionIntoChildrenRule.Config DEFAULT =
        ImmutablePartialJoinConditionIntoChildrenRule.Config.of()
            .withOperandFor(BodoLogicalJoin.class);

    @Override
    default PartialJoinConditionIntoChildrenRule toRule() {
      return new PartialJoinConditionIntoChildrenRule(this);
    }

    /** Defines an operand tree for the given 3 classes. */
    default PartialJoinConditionIntoChildrenRule.Config withOperandFor(
        Class<? extends Join> joinClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(joinClass)
                      .predicate(f -> BodoJoinConditionUtil.requiresTransformationToValid(f))
                      .anyInputs())
          .as(PartialJoinConditionIntoChildrenRule.Config.class);
    }
  }
}
