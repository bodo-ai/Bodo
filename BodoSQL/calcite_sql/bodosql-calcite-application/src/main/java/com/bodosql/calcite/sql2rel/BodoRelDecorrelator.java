package com.bodosql.calcite.sql2rel;

import static java.util.Objects.requireNonNull;

import com.bodosql.calcite.application.logicalRules.BodoFilterCorrelateRule;
import com.bodosql.calcite.application.logicalRules.BodoJoinProjectTransposeNoCSEUndoRule;
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.prepare.BodoPrograms;
import com.bodosql.calcite.prepare.BodoRules;
import com.bodosql.calcite.rel.core.Flatten;
import com.bodosql.calcite.tools.BodoRelBuilder;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.plan.Context;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgram;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelVisitor;
import org.apache.calcite.rel.core.Correlate;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.rules.CoreRules;
import org.apache.calcite.rel.rules.FilterFlattenCorrelatedConditionRule;
import org.apache.calcite.rel.rules.FilterJoinRule;
import org.apache.calcite.rel.rules.FilterProjectTransposeRule;
import org.apache.calcite.rel.rules.SingleValuesOptimizationRules;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexCorrelVariable;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexPermuteInputsShuttle;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.rex.RexVisitorImpl;
import org.apache.calcite.sql.SqlExplainFormat;
import org.apache.calcite.sql.SqlExplainLevel;
import org.apache.calcite.sql2rel.CorrelateProjectExtractor;
import org.apache.calcite.sql2rel.RelDecorrelator;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.mapping.Mapping;
import org.apache.calcite.util.mapping.MappingType;
import org.apache.calcite.util.mapping.Mappings;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.immutables.value.Value;

/**
 * See <a
 * href="https://bodo.atlassian.net/wiki/spaces/B/pages/1469480977/Flatten+Operator+Design#Decorrelation-Overview">this
 * confluence doc</a> for more details on decorrelation and how we use it.
 */
@BodoSQLStyleImmutable
public class BodoRelDecorrelator extends RelDecorrelator {
  protected BodoRelDecorrelator(CorelMap cm, Context context, RelBuilder relBuilder) {
    super(cm, context, relBuilder);
  }

  @Override
  protected RelBuilderFactory relBuilderFactory() {
    return BodoRelBuilder.proto(relBuilder);
  }

  @Override
  protected RelNode decorrelate(RelNode root) {
    // first adjust count() expression if any
    final RelBuilderFactory f = relBuilderFactory();
    HepProgram program =
        HepProgram.builder()
            .addRuleInstance(AdjustProjectForCountAggregateRule.config(false, this, f).toRule())
            .addRuleInstance(AdjustProjectForCountAggregateRule.config(true, this, f).toRule())
            .addRuleInstance(
                FilterJoinRule.FilterIntoJoinRule.FilterIntoJoinRuleConfig.DEFAULT
                    .withRelBuilderFactory(f)
                    .withOperandSupplier(
                        b0 ->
                            b0.operand(Filter.class)
                                .predicate(x -> !x.containsOver())
                                .oneInput(b1 -> b1.operand(Join.class).anyInputs()))
                    .withDescription("FilterJoinRule:filter")
                    .as(FilterJoinRule.FilterIntoJoinRule.FilterIntoJoinRuleConfig.class)
                    .withSmart(true)
                    .withPredicate((join, joinType, exp) -> true)
                    .as(FilterJoinRule.FilterIntoJoinRule.FilterIntoJoinRuleConfig.class)
                    .toRule())
            .addRuleInstance(
                CoreRules.FILTER_PROJECT_TRANSPOSE
                    .config
                    .withRelBuilderFactory(f)
                    .as(FilterProjectTransposeRule.Config.class)
                    .withOperandFor(
                        Filter.class,
                        filter ->
                            !RexUtil.containsCorrelation(filter.getCondition())
                                && !filter.containsOver(),
                        Project.class,
                        project -> !project.containsOver())
                    .withCopyFilter(true)
                    .withCopyProject(true)
                    .toRule())
            // BODO CHANGE: replacing FilterCorrelationRule with BodoFilterCorrelateRule
            .addRuleInstance(
                BodoFilterCorrelateRule.BodoConfig.DEFAULT.withRelBuilderFactory(f).toRule())
            // BODO CHANGE: adding PullFilterAboveFlattenCorrelationRule
            .addRuleInstance(PullFilterAboveFlattenCorrelationRule.config(this, f).toRule())
            .addRuleInstance(RemoveCorrelationForFlattenRule.config(this, f).toRule())
            .addRuleInstance(
                FilterFlattenCorrelatedConditionRule.Config.DEFAULT
                    .withOperandSupplier(
                        op ->
                            op.operand(Filter.class).predicate(x -> !x.containsOver()).anyInputs())
                    .withRelBuilderFactory(f)
                    .toRule())
            .build();

    HepPlanner planner = createPlanner(program);

    planner.setRoot(root);
    root = planner.findBestExp();
    if (SQL2REL_LOGGER.isDebugEnabled()) {
      SQL2REL_LOGGER.debug(
          "Plan before extracting correlated computations:\n" + RelOptUtil.toString(root));
    }
    root = root.accept(new CorrelateProjectExtractor(f));
    // Necessary to update cm (CorrelMap) since CorrelateProjectExtractor above may modify the plan
    this.cm = new CorelMapBuilder().build(root);
    if (SQL2REL_LOGGER.isDebugEnabled()) {
      SQL2REL_LOGGER.debug(
          "Plan after extracting correlated computations:\n" + RelOptUtil.toString(root));
    }
    // Perform decorrelation.
    map.clear();

    final Frame frame = getInvoke(root, false, null);
    if (frame != null) {
      // has been rewritten; apply rules post-decorrelation
      final HepProgramBuilder builder =
          HepProgram.builder()
              .addRuleInstance(CoreRules.FILTER_INTO_JOIN.config.withRelBuilderFactory(f).toRule())
              .addRuleInstance(
                  CoreRules.JOIN_CONDITION_PUSH.config.withRelBuilderFactory(f).toRule());
      if (!getPostDecorrelateRules().isEmpty()) {
        builder.addRuleCollection(getPostDecorrelateRules());
      }
      final HepProgram program2 = builder.build();

      final HepPlanner planner2 = createPlanner(program2);
      final RelNode newRoot = frame.r;
      planner2.setRoot(newRoot);
      return planner2.findBestExp();
    }

    return root;
  }

  // Copied over from RelDecorrelator since it is static, with minor changes
  public static RelNode decorrelateQuery(RelNode rootRel, RelBuilder relBuilder) {
    final CorelMap corelMap = new CorelMapBuilder().build(rootRel);
    if (!corelMap.hasCorrelation()) {
      return rootRel;
    }

    final RelOptCluster cluster = rootRel.getCluster();
    // BODO CHANGE: uses BodoRelDecorrelator instead of RelDecorrelator
    final BodoRelDecorrelator decorrelator =
        new BodoRelDecorrelator(corelMap, cluster.getPlanner().getContext(), relBuilder);

    RelNode newRootRel = decorrelator.removeCorrelationViaRule(rootRel);

    if (SQL2REL_LOGGER.isDebugEnabled()) {
      SQL2REL_LOGGER.debug(
          RelOptUtil.dumpPlan(
              "Plan after removing Correlator",
              newRootRel,
              SqlExplainFormat.TEXT,
              SqlExplainLevel.EXPPLAN_ATTRIBUTES));
    }

    // BODO CHANGE: moved the logic to a helper function
    // to contend with access modifiers.
    newRootRel = decorrelator.decorrelateStep(newRootRel);

    // Re-propagate the hints.
    newRootRel = RelOptUtil.propagateRelHints(newRootRel, true);

    return newRootRel;
  }

  // Helper function used for the BodoRelDecorrelator implementation
  // of decorrelateQuery so that protected fields can be accessed.
  private RelNode decorrelateStep(RelNode root) {
    if (!cm.getMapCorToCorRel().isEmpty()) {
      return decorrelate(root);
    }
    return root;
  }

  // Copied over from RelDecorrelator with new rules added
  @Override
  public RelNode removeCorrelationViaRule(RelNode root) {
    final RelBuilderFactory f = relBuilderFactory();
    HepProgram program =
        HepProgram.builder()
            .addRuleInstance(RemoveSingleAggregateRule.config(f).toRule())
            .addRuleInstance(RemoveCorrelationForScalarProjectRule.config(this, f).toRule())
            .addRuleInstance(RemoveCorrelationForScalarAggregateRule.config(this, f).toRule())
            // New Rules:
            .addRuleInstance(RemoveCorrelationForSingletonValuesRule.config(this, f).toRule())
            .addRuleInstance(RemoveCorrelationForFlattenRule.config(this, f).toRule())
            .addRuleInstance(PushProjectFlattenRule.config(this, f).toRule())
            // Single value can be a blocker to decorrelation handling.
            .addRuleInstance(BodoRules.SINGLE_VALUE_REMOVE_RULE)
            // Sub query expansion often creates joins with 1 row inputs,
            // which could be a blocker to decorrelation.
            .addRuleInstance(SingleValuesOptimizationRules.JOIN_LEFT_INSTANCE)
            .addRuleInstance(SingleValuesOptimizationRules.JOIN_RIGHT_INSTANCE)
            .addRuleInstance(SingleValuesOptimizationRules.JOIN_LEFT_PROJECT_INSTANCE)
            .addRuleInstance(SingleValuesOptimizationRules.JOIN_RIGHT_PROJECT_INSTANCE)
            // Bodo Change: Add project merge and remove to handle projections from
            // SINGLE_VALUE_REMOVE_RULE
            .addRuleInstance(CoreRules.PROJECT_MERGE)
            .addRuleInstance(BodoJoinProjectTransposeNoCSEUndoRule.Config.RIGHT_OUTER.toRule())
            .build();

    // Bodo Change: Run multiple planner passes. This is done
    // because a bug in the HEP planner may incorrectly view the
    // optimizer as done when each rule has finished without
    // considering the dependencies.
    RelNode lastOptimizedPlan = root;
    for (int i = 0; i < BodoPrograms.getMaxHepIterations(); i++) {
      HepPlanner planner = createPlanner(program);
      planner.setRoot(lastOptimizedPlan);
      RelNode curOptimizedPlan = planner.findBestExp();
      if (curOptimizedPlan.deepEquals(lastOptimizedPlan)) {
        return lastOptimizedPlan;
      }
      lastOptimizedPlan = curOptimizedPlan;
    }
    return lastOptimizedPlan;
  }

  // New Function: verifies that no correlations or correlated variables remain
  // anywhere in the plan
  public static void verifyNoCorrelationsRemaining(RelNode rel) {
    NoCorrelationRelVisitor relVisitor = new NoCorrelationRelVisitor();
    relVisitor.go(rel);
    if (relVisitor.foundCorrelatedVariable) {
      throw new RuntimeException(
          "Found correlation in plan:\n"
              + RelOptUtil.dumpPlan(
                  "", rel, SqlExplainFormat.TEXT, SqlExplainLevel.NON_COST_ATTRIBUTES));
    }
  }

  protected static class NoCorrelationRelVisitor extends RelVisitor {
    public boolean foundCorrelatedVariable = false;
    final NoCorrelationRexVisitor rexVisitor = new NoCorrelationRexVisitor(true);

    @Override
    public void visit(RelNode node, int ordinal, @Nullable RelNode parent) {
      if (node instanceof Correlate) {
        foundCorrelatedVariable = true;
        return;
      }
      List<RexNode> rexList = new ArrayList<RexNode>();
      if (node instanceof Project) {
        rexList.addAll(((Project) node).getProjects());
      } else if (node instanceof Filter) {
        rexList.add(((Filter) node).getCondition());
      } else if (node instanceof Join) {
        rexList.add(((Join) node).getCondition());
      }
      for (RexNode rex : rexList) {
        rex.accept(rexVisitor);
        if (rexVisitor.foundCorrelatedVariable) {
          foundCorrelatedVariable = true;
          return;
        }
      }
      super.visit(node, ordinal, parent);
    }
  }

  protected static class NoCorrelationRexVisitor extends RexVisitorImpl<Void> {
    public boolean foundCorrelatedVariable = false;

    protected NoCorrelationRexVisitor(boolean deep) {
      super(deep);
    }

    @Override
    public Void visitCorrelVariable(RexCorrelVariable correlVariable) {
      foundCorrelatedVariable = true;
      return null;
    }
  }

  /**
   * A query like this: SELECT T.A, L.B, FROM T, LATERAL (SELECT T.Z / 10 AS B) L Will be re-written
   * into the following plan:
   *
   * <blockquote>
   *
   * <pre>
   *  LogicalProject(A=$0, B=$1)
   *    LogicalCorrelate()
   *      LogicalProject(A=$0 f1=/($1, 10))
   *        TableScan(...)
   *    LogicalProject($cor0.$f1)
   *      LogicalValues({0})
   * </pre>
   *
   * </blockquote>
   *
   * This rule rewrites this structure into its simplified projection form:
   *
   * <blockquote>
   *
   * <pre>
   *  LogicalProject(A=$0, B=/($1, 10))
   *    TableScan(...)
   * </pre>
   *
   * </blockquote>
   */
  public static final class RemoveCorrelationForSingletonValuesRule
      extends RelRule<
          RemoveCorrelationForSingletonValuesRule.RemoveCorrelationForSingletonValuesRuleConfig> {
    private final BodoRelDecorrelator d;

    public static RemoveCorrelationForSingletonValuesRuleConfig config(
        BodoRelDecorrelator d, RelBuilderFactory f) {
      return ImmutableRemoveCorrelationForSingletonValuesRuleConfig.builder()
          .withRelBuilderFactory(f)
          .withDecorrelator(d)
          .withOperandSupplier(
              b0 ->
                  b0.operand(Correlate.class)
                      .inputs(
                          b1 -> b1.operand(RelNode.class).anyInputs(),
                          b2 ->
                              // Bodo Change: Don't match on window functions
                              b2.operand(Project.class)
                                  .predicate(x -> !x.containsOver())
                                  .oneInput(b3 -> b3.operand(Values.class).noInputs())))
          .build();
    }

    /** Creates a RemoveSingleAggregateRule. */
    RemoveCorrelationForSingletonValuesRule(
        RemoveCorrelationForSingletonValuesRule.RemoveCorrelationForSingletonValuesRuleConfig
            config) {
      super(config);
      this.d = (BodoRelDecorrelator) requireNonNull(config.decorrelator());
    }

    @Override
    public void onMatch(RelOptRuleCall call) {
      final Correlate correlate = call.rel(0);
      final RelNode left = call.rel(1);
      final Project project = call.rel(2);
      final Values values = call.rel(3);

      // The rule only matches if the VALUES clause is a singleton, otherwise
      // a join will be required.
      if (values.tuples.size() != 1) {
        return;
      }
      // This rule only works for inner joins on the correlation
      if (correlate.getJoinType() != JoinRelType.INNER) {
        return;
      }

      d.setCurrent(call.getPlanner().getRoot(), correlate);

      // Check corVar references are valid
      if (!d.checkCorVars(correlate, project, null, null)) {
        return;
      }

      // Verify that no entries contain inputRefs so we can prune the values clause.
      RelOptUtil.InputFinder inputFinder = new RelOptUtil.InputFinder();
      for (RexNode proj : project.getProjects()) {
        proj.accept(inputFinder);
      }
      ImmutableBitSet usedColumns = inputFinder.build();
      if (!usedColumns.isEmpty()) {
        return;
      }

      // Add every field from the lhs to the output
      RelBuilder builder = call.builder();
      RexBuilder rexBuilder = builder.getRexBuilder();

      List<RexNode> projects = new ArrayList();
      RelDataType leftType = left.getRowType();
      // Add the fields on the LHS
      for (int projIdx = 0; projIdx < leftType.getFieldCount(); projIdx++) {
        projects.add(
            rexBuilder.makeInputRef(leftType.getFieldList().get(projIdx).getType(), projIdx));
      }
      // Decorrelate the project inputs.
      for (RexNode expr : project.getProjects()) {
        projects.add(d.removeCorrelationExpr(expr, false));
      }
      RelNode newProject =
          d.relBuilder
              .push(left)
              .projectNamed(projects, correlate.getRowType().getFieldNames(), true)
              .build();

      call.transformTo(newProject);

      d.removeCorVarFromTree(correlate);
    }

    /** Rule configuration. */
    @Value.Immutable(singleton = false)
    public interface RemoveCorrelationForSingletonValuesRuleConfig
        extends BodoRelDecorrelator.Config {
      @Override
      default RemoveCorrelationForSingletonValuesRule toRule() {
        return new RemoveCorrelationForSingletonValuesRule(this);
      }
    }
  }

  /**
   * PIn a plan like the following:
   *
   * <blockquote>
   *
   * <pre>
   *  LogicalCorrelate(correlation=[$cor0], joinType=[inner], requiredColumns=[{3}])
   *    LogicalProject(A=[$0], B=[$1]])
   *      TableScan(...)
   *    LogicalProject(C=[$0], D=[$3]])
   *      BodoLogicalFlatten(Call=[FLATTEN($0)], ..., Repeated columns=[{}])
   *        BodoLogicalProject($f0=[$cor0.$B])
   *          LogicalValues(tuples=[[{ true }]])
   * </pre>
   *
   * </blockquote>
   *
   * This rule pushes the project into the flatten call to explicitly prune which columns are kept.
   *
   * </blockquote>
   */
  public static final class PushProjectFlattenRule
      extends RelRule<PushProjectFlattenRule.PushProjectFlattenRuleConfig> {
    private final BodoRelDecorrelator d;

    public static PushProjectFlattenRuleConfig config(BodoRelDecorrelator d, RelBuilderFactory f) {
      return ImmutablePushProjectFlattenRuleConfig.builder()
          .withRelBuilderFactory(f)
          .withDecorrelator(d)
          .withOperandSupplier(
              b0 ->
                  b0.operand(Correlate.class)
                      .inputs(
                          b1 -> b1.operand(RelNode.class).anyInputs(),
                          b2 ->
                              b2.operand(Project.class)
                                  .predicate(x -> !x.containsOver())
                                  .oneInput(b3 -> b3.operand(Flatten.class).anyInputs())))
          .build();
    }

    /** Creates a RemoveSingleAggregateRule. */
    PushProjectFlattenRule(PushProjectFlattenRule.PushProjectFlattenRuleConfig config) {
      super(config);
      this.d = (BodoRelDecorrelator) requireNonNull(config.decorrelator());
    }

    @Override
    public void onMatch(RelOptRuleCall call) {
      final Correlate correlate = call.rel(0);
      final RelNode left = call.rel(1);
      final Project project = call.rel(2);
      final Flatten flatten = call.rel(3);

      List<Integer> usedOutputCols = new ArrayList();
      for (RexNode proj : project.getProjects()) {
        if (proj instanceof RexInputRef) {
          int idx = ((RexInputRef) proj).getIndex();
          usedOutputCols.add(flatten.getUsedColOutputs().nth(idx));
        } else {
          return;
        }
      }

      // Build the new flatten node
      Flatten newFlatten =
          flatten.copy(
              flatten.getTraitSet(),
              flatten.getInput(),
              flatten.getCall(),
              flatten.getCallType(),
              ImmutableBitSet.of(usedOutputCols),
              flatten.getRepeatColumns());

      RelNode newCorrelate = correlate.copy(correlate.getTraitSet(), List.of(left, newFlatten));
      call.transformTo(newCorrelate);
    }

    /** Rule configuration. */
    @Value.Immutable(singleton = false)
    public interface PushProjectFlattenRuleConfig extends BodoRelDecorrelator.Config {
      @Override
      default PushProjectFlattenRule toRule() {
        return new PushProjectFlattenRule(this);
      }
    }
  }

  /**
   * A query like this: SELECT * FROM T, LATERAL FLATTEN(T.A) L Will be re-written into the
   * following plan:
   *
   * <blockquote>
   *
   * <pre>
   *  LogicalCorrelate(correlation=[$cor0], joinType=[inner], requiredColumns=[{3}])
   *    LogicalProject(A=[$0], B=[$1]])
   *      TableScan(...)
   *    BodoLogicalFlatten(Call=[FLATTEN($0)], ..., Repeated columns=[{}])
   *      BodoLogicalProject($f0=[$cor0.$B])
   *        LogicalValues(tuples=[[{ true }]])
   * </pre>
   *
   * </blockquote>
   *
   * This rule rewrites this structure into its simplified flatten form:
   *
   * <blockquote>
   *
   * <pre>
   *  BodoLogicalFlatten(Call=[FLATTEN($1)], ..., Repeated columns=[{$0}])
   *    LogicalProject(A=[$0], B=[$1]])
   *      TableScan(...)
   * </pre>
   *
   * </blockquote>
   */
  public static final class RemoveCorrelationForFlattenRule
      extends RelRule<RemoveCorrelationForFlattenRule.RemoveCorrelationForFlattenRuleConfig> {
    private final BodoRelDecorrelator d;

    public static RemoveCorrelationForFlattenRuleConfig config(
        BodoRelDecorrelator d, RelBuilderFactory f) {
      return ImmutableRemoveCorrelationForFlattenRuleConfig.builder()
          .withRelBuilderFactory(f)
          .withDecorrelator(d)
          .withOperandSupplier(
              b0 ->
                  b0.operand(Correlate.class)
                      .inputs(
                          b1 -> b1.operand(RelNode.class).anyInputs(),
                          b2 ->
                              b2.operand(Flatten.class)
                                  .oneInput(
                                      b3 ->
                                          b3.operand(Project.class)
                                              .predicate(x -> !x.containsOver())
                                              .oneInput(
                                                  b4 -> b4.operand(Values.class).noInputs()))))
          .build();
    }

    /** Creates a RemoveSingleAggregateRule. */
    RemoveCorrelationForFlattenRule(
        RemoveCorrelationForFlattenRule.RemoveCorrelationForFlattenRuleConfig config) {
      super(config);
      this.d = (BodoRelDecorrelator) requireNonNull(config.decorrelator());
    }

    @Override
    public void onMatch(RelOptRuleCall call) {
      final Correlate correlate = call.rel(0);
      final RelNode left = call.rel(1);
      final Flatten flatten = call.rel(2);
      final Project project = call.rel(3);
      final Values values = call.rel(4);

      // The rule only matches if the VALUES clause is a singleton, otherwise
      // a join will be required.
      if (values.tuples.size() != 1) {
        return;
      }

      d.setCurrent(call.getPlanner().getRoot(), correlate);

      // Check corVar references are valid
      if (!d.checkCorVars(correlate, project, null, null)) {
        return;
      }

      // The next step is to build the projection containing all of the values
      // from the left input as well as all the terms from the rhs that were
      // operands to the flatten call.
      RexCall oldFlattenCall = flatten.getCall();
      List<RexNode> flattenOperands = new ArrayList();
      List<RexNode> projectTerms = new ArrayList();
      List<Integer> repeatCols = new ArrayList();

      // Push the left input for any field generation
      d.relBuilder.push(left);

      // Add all elements from the lhs to the new projection
      for (int i = 0; i < left.getRowType().getFieldCount(); i++) {
        projectTerms.add(d.relBuilder.field(i));
        repeatCols.add(i);
      }

      // Add all operands to the flatten call to the new projection
      // by finding the field they reference in the child projection
      // and adding the decorrelated version.
      for (RexNode operand : oldFlattenCall.operands) {
        if (operand instanceof RexInputRef) {
          int projIdx = ((RexInputRef) operand).getIndex();
          RexNode input = project.getProjects().get(projIdx);
          int newProjIdx = projectTerms.size();
          RexNode decorrelatedInput = d.removeCorrelationExpr(input, false);
          projectTerms.add(decorrelatedInput);
          flattenOperands.add(new RexInputRef(newProjIdx, input.getType()));
        } else if (operand instanceof RexLiteral) {
          // Copy over literal arguments directly.
          flattenOperands.add(operand);
        } else {
          return;
        }
      }

      // Build the new projection with the combined terms
      RelNode newProject = d.relBuilder.project(projectTerms).build();

      // Construct the new call to the flatten operation
      RexCall newFlattenCall = oldFlattenCall.clone(oldFlattenCall.type, flattenOperands);

      // Build the new flatten node
      Flatten newFlatten =
          flatten.copy(
              flatten.getTraitSet(),
              newProject,
              newFlattenCall,
              flatten.getCallType(),
              flatten.getUsedColOutputs(),
              ImmutableBitSet.of(repeatCols));

      // Build a new projection on top of the flatten node that reshuffles
      // the outputs of the flatten node so the replicated columns come first
      List<RexNode> newOrder = new ArrayList();
      int numUsed = flatten.getUsedColOutputs().cardinality();
      for (int i : newFlatten.getRepeatColumns()) {
        newOrder.add(
            new RexInputRef(
                i + numUsed, newFlatten.getRowType().getFieldList().get(i + numUsed).getType()));
      }
      // Shift the used col outputs since there are now
      // repeated columns before them in the inputs.
      int nOutputs = flatten.getUsedColOutputs().cardinality();
      for (int i = 0; i < nOutputs; i++) {
        newOrder.add(new RexInputRef(i, newFlatten.getRowType().getFieldList().get(i).getType()));
      }
      RelNode reorderProject = d.relBuilder.push(newFlatten).project(newOrder).build();
      call.transformTo(reorderProject);
      d.removeCorVarFromTree(correlate);
    }

    /** Rule configuration. */
    @Value.Immutable(singleton = false)
    public interface RemoveCorrelationForFlattenRuleConfig extends BodoRelDecorrelator.Config {
      @Override
      default RemoveCorrelationForFlattenRule toRule() {
        return new RemoveCorrelationForFlattenRule(this);
      }
    }
  }

  /**
   * Pulls a filter above a flatten node above its corresponding correlation. For example, with the
   * following input plan:
   *
   * <blockquote>
   *
   * <pre>
   *  LogicalCorrelate(...)
   *    LogicalProject(A=..., B=...)
   *      ...
   *    LogicalFilter(IS NOT NULL($0))
   *       BodoLogicalFlatten(Call=[FLATTEN($0)], ...)
   *         ...
   * </pre>
   *
   * </blockquote>
   *
   * This rule rewrites this structure into a form with the filter pulled up:
   *
   * <blockquote>
   *
   * <pre>
   *  LogicalFilter(IS NOT NULL($2))
   *    LogicalCorrelate(...)
   *      LogicalProject(A=..., B=...)
   *        ...
   *    BodoLogicalFlatten(Call=[FLATTEN($0)], ...)
   *      ...
   * </pre>
   *
   * </blockquote>
   */
  public static final class PullFilterAboveFlattenCorrelationRule
      extends RelRule<
          PullFilterAboveFlattenCorrelationRule.PullFilterAboveFlattenCorrelationRuleConfig> {
    private final BodoRelDecorrelator d;

    public static PullFilterAboveFlattenCorrelationRuleConfig config(
        BodoRelDecorrelator d, RelBuilderFactory f) {
      return ImmutablePullFilterAboveFlattenCorrelationRuleConfig.builder()
          .withRelBuilderFactory(f)
          .withDecorrelator(d)
          .withOperandSupplier(
              b0 ->
                  b0.operand(Correlate.class)
                      .inputs(
                          b1 -> b1.operand(RelNode.class).anyInputs(),
                          b2 ->
                              b2.operand(Filter.class)
                                  .predicate(
                                      x ->
                                          !x.containsOver()
                                              && !RexUtil.containsCorrelation(x.getCondition())
                                              && isFilterAncestorOfFlatten(x))
                                  .oneInput(b3 -> b3.operand(RelNode.class).anyInputs())))
          .build();
    }

    /**
     * Detects cases where a chain of filters leads to a flatten node.
     *
     * @param rel The current rel node
     * @return Whether it is a flatten node or a chain of filters leading to a flatten node.
     */
    public static Boolean isFilterAncestorOfFlatten(RelNode rel) {
      if (rel instanceof Flatten) return true;
      if (rel instanceof HepRelVertex)
        return isFilterAncestorOfFlatten(((HepRelVertex) rel).getCurrentRel());
      if (rel instanceof Filter) return isFilterAncestorOfFlatten(rel.getInput(0));
      return false;
    }

    /** Creates a RemoveSingleAggregateRule. */
    PullFilterAboveFlattenCorrelationRule(
        PullFilterAboveFlattenCorrelationRule.PullFilterAboveFlattenCorrelationRuleConfig config) {
      super(config);
      this.d = (BodoRelDecorrelator) requireNonNull(config.decorrelator());
    }

    @Override
    public void onMatch(RelOptRuleCall call) {
      final Correlate correlate = call.rel(0);
      final RelNode left = call.rel(1);
      final Filter filter = call.rel(2);
      final RelNode child = call.rel(3);
      Correlate newCorrelate = correlate.copy(correlate.getTraitSet(), List.of(left, child));
      d.relBuilder.push(newCorrelate);
      int offset = left.getRowType().getFieldCount();
      int nFilterFields = filter.getRowType().getFieldCount();
      Mapping mapping =
          Mappings.create(
              MappingType.SURJECTION, nFilterFields, correlate.getRowType().getFieldCount());
      for (int i = 0; i < nFilterFields; i++) {
        mapping.set(i, i + offset);
      }
      RexNode oldCond = filter.getCondition();
      RexNode cond = oldCond.accept(new RexPermuteInputsShuttle(mapping, child));
      d.relBuilder.filter(cond);
      RelNode result = d.relBuilder.build();
      call.transformTo(result);
    }

    /** Rule configuration. */
    @Value.Immutable(singleton = false)
    public interface PullFilterAboveFlattenCorrelationRuleConfig
        extends BodoRelDecorrelator.Config {
      @Override
      default PullFilterAboveFlattenCorrelationRule toRule() {
        return new PullFilterAboveFlattenCorrelationRule(this);
      }
    }
  }
}
