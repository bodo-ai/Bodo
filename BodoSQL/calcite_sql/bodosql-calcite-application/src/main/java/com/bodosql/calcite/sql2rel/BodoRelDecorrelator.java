package com.bodosql.calcite.sql2rel;

import static java.util.Objects.requireNonNull;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.Flatten;
import java.util.ArrayList;
import java.util.List;
import org.apache.calcite.plan.Context;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgram;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelVisitor;
import org.apache.calcite.rel.core.Correlate;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexCorrelVariable;
import org.apache.calcite.rex.RexFieldAccess;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexVisitorImpl;
import org.apache.calcite.sql.SqlExplainFormat;
import org.apache.calcite.sql.SqlExplainLevel;
import org.apache.calcite.sql2rel.RelDecorrelator;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Pair;
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
            .build();

    HepPlanner planner = createPlanner(program);

    planner.setRoot(root);
    return planner.findBestExp();
  }

  // New Function: verifies that no correlations or correlated variables remain
  // anywhere in the plan
  public static void verifyNoCorrelationsRemaining(RelNode rel) throws Exception {
    NoCorrelationRelVisitor relVisitor = new NoCorrelationRelVisitor();
    relVisitor.go(rel);
    if (relVisitor.foundCorrelatedVariable) {
      throw new Exception("Found correlation in plan:\n" + RelOptUtil.toString(rel));
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
   *  LogicalProject($cor0.$f1)
   *    LogicalValues({0})
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
                          b1 -> b1.operand(Project.class).anyInputs(),
                          b2 ->
                              b2.operand(Project.class)
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
      final Project left = call.rel(1);
      final Project project = call.rel(2);
      final Values values = call.rel(3);

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

      // Verify that all the entries in the rhs are correlation variable references, as
      // opposed to a computation that may depend on the values clause.
      for (RexNode proj : project.getProjects()) {
        if (!(proj instanceof RexFieldAccess)
            || !(((RexFieldAccess) proj).getReferenceExpr() instanceof RexCorrelVariable)) {
          return;
        }
      }

      // Add every field from the lhs project to the output
      final List<RelDataTypeField> fieldList = left.getRowType().getFieldList();
      List<Pair<RexNode, String>> projects = new ArrayList<>();
      for (int projIdx = 0; projIdx < left.getProjects().size(); projIdx++) {
        projects.add(Pair.of(left.getProjects().get(projIdx), fieldList.get(projIdx).getName()));
      }
      // For each correlated ref field in the rhs project, re-add the corresponding
      // field from the lhs to the output.
      List<CorRef> refs = new ArrayList(d.cm.mapRefRelToCorRef.get(project));
      for (int projIdx = 0; projIdx < refs.size(); projIdx++) {
        CorRef ref = refs.get(projIdx);
        projects.add(
            Pair.of(
                left.getProjects().get(ref.field),
                project.getRowType().getFieldNames().get(projIdx)));
      }
      RelNode newProject =
          d.relBuilder
              .push(left.getInput(0))
              .projectNamed(Pair.left(projects), Pair.right(projects), true)
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
      List<RexNode> flattenOperands = new ArrayList<RexNode>();
      List<RexNode> projectTerms = new ArrayList<RexNode>();
      List<Integer> repeatCols = new ArrayList<Integer>();

      // Add all elements from the lhs to the new projection
      for (int i = 0; i < left.getRowType().getFieldCount(); i++) {
        projectTerms.add(new RexInputRef(i, left.getRowType().getFieldList().get(i).getType()));
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
      RelNode newProject = d.relBuilder.push(left).project(projectTerms).build();

      // Construct the new call to the flatten operation
      RexCall newFlattenCall = oldFlattenCall.clone(oldFlattenCall.type, flattenOperands);

      // Build the new output row type of the flatten node
      List<RelDataTypeField> rowType = new ArrayList<RelDataTypeField>();
      rowType.addAll(flatten.getCallType().getFieldList());
      rowType.addAll(left.getRowType().getFieldList());

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
      List<RexNode> newOrder = new ArrayList<RexNode>();
      int numUsed = flatten.getUsedColOutputs().cardinality();
      for (int i : newFlatten.getRepeatColumns()) {
        newOrder.add(
            new RexInputRef(
                i + numUsed, newFlatten.getRowType().getFieldList().get(i + numUsed).getType()));
      }
      // Shift the used col outputs since there are now
      // repeated columns before them in the inputs.
      for (int i : flatten.getUsedColOutputs()) {
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
}
