package com.bodosql.calcite.application.logicalRules;

import static com.bodosql.calcite.application.logicalRules.FilterRulesCommon.filterContainsOr;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.calcite.plan.RelOptPredicateList;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.rules.SubstitutionRule;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexExecutor;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexOver;
import org.apache.calcite.rex.RexSimplify;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Util;
import org.immutables.value.Value;

/**
 * Planner rule that recognizes a {@link Join} whose condition contains a disjunction referencing
 * both inputs and ANDs onto the condition the strongest single-sided predicates the disjunction
 * implies. The original disjunction is kept, so the rewrite is an equivalence on the condition.
 *
 * <p>For example, with A, C over the left input and B, D over the right input:
 *
 * <pre>
 *   OR(AND(A, B), AND(C, D))
 *     -&gt; AND(OR(AND(A, B), AND(C, D)), OR(A, C), OR(B, D))
 * </pre>
 *
 * <p>Neither {@code OR(A, C)} nor {@code OR(B, D)} is a common factor, so {@link
 * JoinReorderConditionRule} cannot extract them. Once they are conjuncts of the join condition,
 * FilterJoinRule pushes them beneath the join and RexSimplify collapses them into SEARCH/IN, which
 * lets them reach the table scans. TPC-H Q7's nation condition is the canonical case.
 *
 * <p>Termination: a derived predicate is only added if it is not already (in canonical form) a
 * conjunct of the join condition or a pulled-up predicate of the input it applies to. After
 * FilterJoinRule moves the derived conjuncts into Filters on the inputs, they show up in {@link
 * RelMetadataQuery#getPulledUpPredicates} and the rule becomes a no-op, exactly as {@code
 * JoinPushTransitivePredicatesRule} relies on. Any node that absorbs a filter must therefore keep
 * reporting it as a pulled-up predicate.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class JoinDeriveOrPredicatesRule extends RelRule<JoinDeriveOrPredicatesRule.Config>
    implements SubstitutionRule {

  /** Creates a JoinDeriveOrPredicatesRule. */
  protected JoinDeriveOrPredicatesRule(JoinDeriveOrPredicatesRule.Config config) {
    super(config);
  }

  /**
   * Determines if this rule could apply to the given join: the condition contains an OR and does
   * not contain sub-queries or window functions, which we do not attempt to duplicate.
   *
   * @param join The join node that may be rewritten.
   * @return If the join is a candidate.
   */
  public static boolean containsOrCondition(Join join) {
    RexNode cond = join.getCondition();
    return filterContainsOr(cond)
        && !RexUtil.SubQueryFinder.containsSubQuery(join)
        && !RexOver.containsOver(cond);
  }

  /** This method is called when the rule finds a RelNode that matches config requirements. */
  @Override
  public void onMatch(RelOptRuleCall call) {
    final Join join = call.rel(0);
    final RexBuilder rexBuilder = call.builder().getRexBuilder();
    final RelMetadataQuery mq = call.getMetadataQuery();
    final RexExecutor executor = Util.first(call.getPlanner().getExecutor(), RexUtil.EXECUTOR);
    final RexSimplify simplify = new RexSimplify(rexBuilder, RelOptPredicateList.EMPTY, executor);

    final int nLeft = join.getLeft().getRowType().getFieldCount();
    final int nRight = join.getRight().getRowType().getFieldCount();
    final ImmutableBitSet leftBits = ImmutableBitSet.range(0, nLeft);
    final ImmutableBitSet rightBits = ImmutableBitSet.range(nLeft, nLeft + nRight);

    final List<RexNode> conjuncts = RelOptUtil.conjunctions(join.getCondition());
    System.out.println("conjuncts = " + conjuncts);

    // Canonical forms of every predicate that already holds at this join: the existing conjuncts
    // of the condition and whatever the inputs already filter on (expressed in join field
    // indices). Anything we derive that is in this set is redundant and must not be re-added,
    // otherwise the rule never reaches a fixed point.
    final Set<RexNode> known = new HashSet<>();
    for (RexNode conjunct : conjuncts) {
      known.add(canonicalize(simplify, conjunct));
    }
    addPulledUpPredicates(known, mq, join.getLeft(), 0, simplify);
    addPulledUpPredicates(known, mq, join.getRight(), nLeft, simplify);

    final List<RexNode> derived = new ArrayList<>();
    for (RexNode conjunct : conjuncts) {
      ImmutableBitSet used = RelOptUtil.InputFinder.bits(conjunct);
      if (!filterContainsOr(conjunct) || leftBits.contains(used) || rightBits.contains(used)) {
        // No OR to look inside, or already single-sided (FilterJoinRule handles those).
        continue;
      }
      for (ImmutableBitSet side : new ImmutableBitSet[] {leftBits, rightBits}) {
        RexNode implied = deriveSidePredicate(rexBuilder, conjunct, side);
        // The result may itself be a conjunction; track each piece separately.
        for (RexNode piece : RelOptUtil.conjunctions(implied)) {
          RexNode canonical = canonicalize(simplify, piece);
          if (canonical.isAlwaysTrue() || !known.add(canonical)) {
            continue;
          }
          derived.add(canonical);
        }
      }
    }

    if (derived.isEmpty()) {
      return;
    }

    final List<RexNode> newConjuncts = new ArrayList<>(conjuncts);
    newConjuncts.addAll(derived);
    final RexNode newCondition = RexUtil.composeConjunction(rexBuilder, newConjuncts);
    final Join newJoin =
        join.copy(
            join.getTraitSet(),
            newCondition,
            join.getLeft(),
            join.getRight(),
            join.getJoinType(),
            join.isSemiJoinDone());
    call.transformTo(newJoin);
  }

  /**
   * Returns the strongest predicate over only the fields in {@code side} that is implied by {@code
   * cond}. Returns TRUE when nothing is implied.
   *
   * <ul>
   *   <li>A leaf is kept if it references only {@code side} (and is deterministic), else TRUE.
   *   <li>AND(x, y) implies AND(derive(x), derive(y)); TRUE parts are dropped.
   *   <li>OR(x, y) implies OR(derive(x), derive(y)); if any part is TRUE the whole thing is TRUE.
   * </ul>
   *
   * <p>NOT and other operators are treated as leaves: weakening the argument of a NOT would
   * strengthen the result, which is unsound.
   */
  private static RexNode deriveSidePredicate(
      RexBuilder rexBuilder, RexNode cond, ImmutableBitSet side) {
    switch (cond.getKind()) {
      case AND:
        {
          List<RexNode> parts = new ArrayList<>();
          for (RexNode operand : ((RexCall) cond).operands) {
            RexNode d = deriveSidePredicate(rexBuilder, operand, side);
            if (!d.isAlwaysTrue()) {
              parts.add(d);
            }
          }
          // composeConjunction returns TRUE for an empty list.
          return RexUtil.composeConjunction(rexBuilder, parts);
        }
      case OR:
        {
          List<RexNode> parts = new ArrayList<>();
          for (RexNode operand : ((RexCall) cond).operands) {
            RexNode d = deriveSidePredicate(rexBuilder, operand, side);
            if (d.isAlwaysTrue()) {
              return rexBuilder.makeLiteral(true);
            }
            parts.add(d);
          }
          return RexUtil.composeDisjunction(rexBuilder, parts);
        }
      default:
        if (side.contains(RelOptUtil.InputFinder.bits(cond)) && RexUtil.isDeterministic(cond)) {
          return cond;
        }
        return rexBuilder.makeLiteral(true);
    }
  }

  /**
   * Adds the canonical form of every pulled-up predicate of {@code input} to {@code known},
   * shifting field references by {@code offset} so they are in the join's field space.
   */
  private static void addPulledUpPredicates(
      Set<RexNode> known, RelMetadataQuery mq, RelNode input, int offset, RexSimplify simplify) {
    RelOptPredicateList preds = mq.getPulledUpPredicates(input);
    if (preds == null) {
      return;
    }
    for (RexNode pred : preds.pulledUpPredicates) {
      known.add(canonicalize(simplify, RexUtil.shift(pred, offset)));
    }
  }

  /**
   * Canonical form used for "already present" comparisons. Both freshly derived predicates and
   * existing predicates go through the same simplifier so that e.g. OR(x = 'A', x = 'B') and
   * SEARCH(x, Sarg['A', 'B']) compare equal. Join/filter conditions treat UNKNOWN as FALSE.
   */
  private static RexNode canonicalize(RexSimplify simplify, RexNode node) {
    return simplify.simplifyUnknownAsFalse(node);
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    // Matches any Join whose condition contains an OR.
    JoinDeriveOrPredicatesRule.Config DEFAULT =
        ImmutableJoinDeriveOrPredicatesRule.Config.of()
            .withOperandSupplier(
                b ->
                    b.operand(Join.class)
                        .predicate(JoinDeriveOrPredicatesRule::containsOrCondition)
                        .anyInputs())
            .as(JoinDeriveOrPredicatesRule.Config.class);

    @Override
    default JoinDeriveOrPredicatesRule toRule() {
      return new JoinDeriveOrPredicatesRule(this);
    }
  }
}
