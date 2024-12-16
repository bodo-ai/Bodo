package com.bodosql.calcite.prepare;

import com.bodosql.calcite.rel.core.BodoLogicalRelFactories;
import com.bodosql.calcite.rel.logical.BodoLogicalFilter;
import com.bodosql.calcite.rel.logical.BodoLogicalJoin;
import com.bodosql.calcite.rel.logical.BodoLogicalProject;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.rules.FilterMultiJoinMergeRule;
import org.apache.calcite.rel.rules.JoinToMultiJoinRule;
import org.apache.calcite.rel.rules.MultiJoin;
import org.apache.calcite.rel.rules.MultiJoinProjectTransposeRule;
import org.apache.calcite.rel.rules.ProjectMultiJoinMergeRule;

/**
 * Contains all of our initialized multi-join rules. These are moved here because
 * withOperandSupplier is frustrating to write in kotlin and allows for simpler organization.
 */
public class MultiJoinRules {

  /** Gather join nodes into a single multi join for optimization. */
  public static final RelOptRule JOIN_TO_MULTI_JOIN =
      JoinToMultiJoinRule.Config.DEFAULT
          // Disable Join to MultiJoin conversion if the join has hints
          .withOperandSupplier(
              b0 ->
                  b0.operand(BodoLogicalJoin.class)
                      .predicate(join -> join.getHints().isEmpty())
                      .inputs(
                          b1 -> b1.operand(RelNode.class).anyInputs(),
                          b2 -> b2.operand(RelNode.class).anyInputs()))
          .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
          .toRule();

  /** Push projections into Multi-Joins to expand the scope of the multi join. */
  public static final RelOptRule PROJECT_MULTI_JOIN_MERGE =
      ProjectMultiJoinMergeRule.Config.DEFAULT
          .withOperandFor(BodoLogicalProject.class, MultiJoin.class)
          .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
          .toRule();

  /** Push Projects between a join and two lower multi-joins */
  public static final RelOptRule MULTI_JOIN_BOTH_PROJECT =
      MultiJoinProjectTransposeRule.Config.BOTH_PROJECT
          .withOperandSupplier(
              b0 ->
                  b0.operand(BodoLogicalJoin.class)
                      .inputs(
                          b1 ->
                              b1.operand(BodoLogicalProject.class)
                                  .oneInput(b2 -> b2.operand(MultiJoin.class).anyInputs()),
                          b3 ->
                              b3.operand(BodoLogicalProject.class)
                                  .oneInput(b4 -> b4.operand(MultiJoin.class).anyInputs())))
          .withDescription("MultiJoinProjectTransposeRule: with two LogicalProject children")
          .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
          .toRule();

  /** Push Projects between a join and a left multi-join */
  public static final RelOptRule MULTI_JOIN_LEFT_PROJECT =
      MultiJoinProjectTransposeRule.Config.LEFT_PROJECT
          .withOperandSupplier(
              b0 ->
                  b0.operand(BodoLogicalJoin.class)
                      .inputs(
                          b1 ->
                              b1.operand(BodoLogicalProject.class)
                                  .oneInput(b2 -> b2.operand(MultiJoin.class).anyInputs())))
          .withDescription("MultiJoinProjectTransposeRule: with LogicalProject on left")
          .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
          .toRule();

  /** Push Projects between a join and a right multi-join */
  public static final RelOptRule MULTI_JOIN_RIGHT_PROJECT =
      MultiJoinProjectTransposeRule.Config.RIGHT_PROJECT
          .withOperandSupplier(
              b0 ->
                  b0.operand(BodoLogicalJoin.class)
                      .inputs(
                          b1 -> b1.operand(RelNode.class).anyInputs(),
                          b2 ->
                              b2.operand(BodoLogicalProject.class)
                                  .oneInput(b3 -> b3.operand(MultiJoin.class).anyInputs())))
          .withDescription("MultiJoinProjectTransposeRule: with LogicalProject on right")
          .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
          .toRule();

  /** Push projections into Multi-Joins to expand the scope of the multi join. */
  public static final RelOptRule FILTER_MULTI_JOIN_MERGE =
      FilterMultiJoinMergeRule.Config.DEFAULT
          .withOperandSupplier(
              b0 ->
                  b0.operand(BodoLogicalFilter.class)
                      .predicate(f -> !f.containsOver())
                      .oneInput(b1 -> b1.operand(MultiJoin.class).anyInputs()))
          .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
          .toRule();
}
