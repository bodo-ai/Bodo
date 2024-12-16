package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import kotlin.Pair;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.fun.SqlCastFunction;
import org.apache.calcite.tools.RelBuilder;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.immutables.value.Value;

/**
 * Rule that looks at two projections separated by a filter and determines if any computation in the
 * uppermost filter can be replaced by an RexInputRef. For example imagine we have the following
 * Plan:
 *
 * <p>Project(A=$0, B=$1+1)
 *
 * <p>Filter(cond=[$2=1])
 *
 * <p>Project(col0=$1, col1=$0, col3=$0+1
 *
 * <p>Then this rule determines that we can replace B with col3. The final plan after just this rule
 * would look like
 *
 * <p>Project(A=$0, B=$2)
 *
 * <p>Filter(cond=[$2=1])
 *
 * <p>Project(col0=$1, col1=$0, col3=$0+1
 *
 * <p>To do this we first determine that it would be safe to reorder these instructions and then
 * compare upper most projection after inlining any inputRefs with the values of the column in the
 * lower projection
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class ProjectFilterProjectColumnEliminationRule
    extends RelRule<ProjectFilterProjectColumnEliminationRule.Config>
    implements TransformationRule {

  /** Creates a ProjectFilterProjectColumnEliminationRule. */
  protected ProjectFilterProjectColumnEliminationRule(
      ProjectFilterProjectColumnEliminationRule.Config config) {
    super(config);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Project upperProject = call.rel(0);
    // Note we don't look at the filter once we see a match.
    final Project lowerProject = call.rel(2);
    RelNode x = apply(call, upperProject, lowerProject);
    if (x != null) {
      call.transformTo(x);
    }
  }

  /**
   * Takes a node and replaces any inputRefs with the value in lowerProjectColumns at that index.
   *
   * @param node The node to update
   * @param lowerProjectColumns The columns used to replace inputRefs
   * @return A new node that can be used for comparison but NOT inserted into the plan.
   */
  private static RexNode getInlinedColumn(
      RexNode node, List<RexNode> lowerProjectColumns, RelBuilder builder) {
    // XXX: In the future we can replace this with RexVisitorImpl
    if (node instanceof RexInputRef) {
      RexInputRef inputRef = (RexInputRef) node;
      int idx = inputRef.getIndex();
      return lowerProjectColumns.get(idx);
    } else if (node instanceof RexCall) {
      RexCall callNode = ((RexCall) node);
      // Call expressions need to have their children traversed. No other RexNodes
      // should have children.
      List<RexNode> oldOperands = callNode.getOperands();
      List<RexNode> newOperands = new ArrayList<>();
      boolean replaceNode = false;
      for (RexNode oldOperand : oldOperands) {
        RexNode newOperand = getInlinedColumn(oldOperand, lowerProjectColumns, builder);
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

  /**
   * Creates a map for a projection for the columns that may be used to replace compute inside the
   * upper projection. This map only consists of node with count > 1.
   *
   * <p>The actual Map returned maps the nodeCount -> List[Indices] for efficiently searching for
   * matches.
   *
   * @param lowerProject The project with which to populate the map.
   * @return The mapping from node count to list of indices with that node count that can be used in
   *     replacements.
   */
  private static Map<Integer, List<Integer>> createCandidateReplacementsMap(Project lowerProject) {
    HashMap<Integer, List<Integer>> map = new HashMap<>();
    List<RexNode> lowerProjectColumns = lowerProject.getProjects();
    for (int i = 0; i < lowerProjectColumns.size(); i++) {
      RexNode node = lowerProjectColumns.get(i);
      int count = node.nodeCount();
      if (count > 1) {
        if (!map.containsKey(count)) {
          map.put(count, new ArrayList<>());
        }
        List<Integer> idxList = map.get(count);
        idxList.add(i);
      }
    }
    return map;
  }

  /**
   * Finds any columns in the upper projection that can be replaced with an existing column from the
   * input. This returns a list of indices mapping the upperProjection to the input column number
   * that is equivalent. If the value is -1 it cannot or should not be replaced.
   *
   * <p>To accomplish this we "inline" the value of any inputRefs in the upper projection with their
   * value in the lower projection. Then we check if the two expressions are equal (and therefore
   * equivalent).
   *
   * <p>To simplify the check of if we need to generate a new projection we also return a boolean
   * indicating if any column was replaced.
   *
   * @param upperProject The project whose columns should be checked from replacement.
   * @param lowerProject The lower projection whose columns are used for inlining and comparison.
   * @param countMap A Map used to search for possibly equivalent columns. This maps the node count
   *     to a list of column number in the lower projection that contain that number of nodes.
   * @param builder RelBuilder used to generate the code for the inlining comparison.
   * @return A list of indices used to map the upper projection to the lower projection and a
   *     boolean indicating if any column was replaced.
   */
  private static Pair<List<Integer>, Boolean> findColumnReplacements(
      Project upperProject,
      Project lowerProject,
      Map<Integer, List<Integer>> countMap,
      RelBuilder builder) {
    List<RexNode> upperProjectColumns = upperProject.getProjects();
    List<RexNode> lowerProjectColumns = lowerProject.getProjects();
    List<Integer> replacements = new ArrayList<>(upperProjectColumns.size());
    // Store a flag for any possible replacement.
    boolean canReplace = false;
    for (int i = 0; i < upperProjectColumns.size(); i++) {
      // Initialize to -1
      replacements.add(-1);
      RexNode node = upperProjectColumns.get(i);
      int count = node.nodeCount();
      if (count > 1) {
        // Generate the inlined column
        RexNode inlinedColumn = getInlinedColumn(node, lowerProjectColumns, builder);
        int newCount = inlinedColumn.nodeCount();
        if (countMap.containsKey(newCount)) {
          List<Integer> candidates = countMap.get(newCount);
          for (int colIdx : candidates) {
            RexNode lowerColumn = lowerProjectColumns.get(colIdx);
            if (lowerColumn.equals(inlinedColumn)) {
              // We have found a match.
              canReplace = true;
              replacements.set(i, colIdx);
              break;
            }
          }
        }
      }
    }
    return new Pair<>(replacements, canReplace);
  }

  /**
   * Generate the final projection that results after successfully replacing 1 or more columns.
   *
   * @param upperProject The original upper projection. This is used to extract fields that are
   *     unchanged.
   * @param replacements Mapping from column number to the field in the input that has an equivalent
   *     column. -1 means a column won't be replaced.
   * @param builder The RelBuilder used to generate the final projection.
   * @return The final projection with columns replaced.
   */
  private static RelNode generateFinalProject(
      Project upperProject, List<Integer> replacements, RelBuilder builder) {
    List<RexNode> upperProjectColumns = upperProject.getProjects();
    List<String> fieldNames = upperProject.getRowType().getFieldNames();
    builder.push(upperProject.getInput());
    List<RexNode> finalColumns = new ArrayList<>(upperProjectColumns.size());
    for (int i = 0; i < upperProjectColumns.size(); i++) {
      int idx = replacements.get(i);
      if (idx == -1) {
        finalColumns.add(upperProjectColumns.get(i));
      } else {
        // Generate an inputRef instead.
        finalColumns.add(builder.field(idx));
      }
    }
    builder.project(finalColumns, fieldNames);
    return builder.build();
  }

  private static @Nullable RelNode apply(
      RelOptRuleCall call, Project upperProject, Project lowerProject) {
    RelBuilder builder = call.builder();
    // Step 1: Create a map from each lower project column's node count to the contents. This will
    // limit the search.
    // For simplicity, we also skip any column with size 1 as it is never beneficial to replace
    // those. This map is nodeCount -> List[ColumnNumbers]
    Map<Integer, List<Integer>> countMap = createCandidateReplacementsMap(lowerProject);
    if (countMap.isEmpty()) {
      // If this node is empty then just return
      return null;
    }
    // Step 2: Iterate through the columns in the upper projection.
    // For any columns we may want to replace we inline the body of any inputRefs
    // with the value in the lowerProjection and check for matches.
    Pair<List<Integer>, Boolean> replacementInfo =
        findColumnReplacements(upperProject, lowerProject, countMap, builder);
    List<Integer> replacements = replacementInfo.getFirst();
    boolean canReplace = replacementInfo.getSecond();
    // Store a list of indices that should be used to replace each column.
    // If a column is not replaced we will store -1.
    if (!canReplace) {
      return null;
    }
    // Step 3: Generate the new projection.
    return generateFinalProject(upperProject, replacements, builder);
  }

  /**
   * Determine if a projection is not an identity (all input Refs). If a project is just the
   * identity there is not point in running this rule.
   *
   * @param project The projection to check.
   * @return Is any column ot an inputRef
   */
  public static boolean isNotIdentity(Project project) {
    return !RexUtil.isIdentity(project.getProjects(), project.getInput().getRowType());
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    ProjectFilterProjectColumnEliminationRule.Config DEFAULT =
        com.bodosql
            .calcite
            .application
            .logicalRules
            .ImmutableProjectFilterProjectColumnEliminationRule
            .Config
            .of()
            .withOperandFor(Project.class, Filter.class);

    @Override
    default ProjectFilterProjectColumnEliminationRule toRule() {
      return new ProjectFilterProjectColumnEliminationRule(this);
    }

    /** Defines an operand tree for the given classes. */
    default ProjectFilterProjectColumnEliminationRule.Config withOperandFor(
        Class<? extends Project> projectClass, Class<? extends Filter> filterClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(projectClass)
                      .predicate(p -> !p.containsOver())
                      .predicate(ProjectFilterProjectColumnEliminationRule::isNotIdentity)
                      .oneInput(
                          b1 ->
                              b1.operand(filterClass)
                                  .oneInput(
                                      b2 ->
                                          b2.operand(projectClass)
                                              .predicate(
                                                  ProjectFilterProjectColumnEliminationRule
                                                      ::isNotIdentity)
                                              .anyInputs())))
          .as(ProjectFilterProjectColumnEliminationRule.Config.class);
    }
  }
}
