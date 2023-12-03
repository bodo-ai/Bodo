package com.bodosql.calcite.rel.metadata;

import static java.util.Objects.requireNonNull;

import com.bodosql.calcite.rel.core.RowSample;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptPredicateList;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.BodoRexSimplify;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexExecutor;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexPermuteInputsShuttle;
import org.apache.calcite.rex.RexSimplify;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.rex.RexVisitorImpl;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.util.BitSets;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Util;
import org.apache.calcite.util.mapping.Mapping;
import org.apache.calcite.util.mapping.MappingType;
import org.apache.calcite.util.mapping.Mappings;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * Utility to infer Predicates that are applicable above a RelNode.
 *
 * <p>The only change that Bodo makes is to Join predicate inference. This change can be undone
 * if/when the upstream calcite PR merges: https://github.com/apache/calcite/pull/3452
 */
public class BodoRelMdPredicates implements MetadataHandler<BuiltInMetadata.Predicates> {
  public static final RelMetadataProvider SOURCE =
      ReflectiveRelMetadataProvider.reflectiveSource(
          new BodoRelMdPredicates(), BuiltInMetadata.Predicates.Handler.class);

  private static final List<RexNode> EMPTY_LIST = ImmutableList.of();

  @Override
  public MetadataDef<BuiltInMetadata.Predicates> getDef() {
    return BuiltInMetadata.Predicates.DEF;
  }

  /** Infers predicates for a RowSample. */
  public RelOptPredicateList getPredicates(RowSample sample, RelMetadataQuery mq) {
    RelNode input = sample.getInput();
    return mq.getPulledUpPredicates(input);
  }

  /**
   * Infers predicates for a {@link org.apache.calcite.rel.core.Join} (including {@code SemiJoin}).
   */
  public RelOptPredicateList getPredicates(Join join, RelMetadataQuery mq) {
    RelOptCluster cluster = join.getCluster();
    RexBuilder rexBuilder = cluster.getRexBuilder();
    final RexExecutor executor = Util.first(cluster.getPlanner().getExecutor(), RexUtil.EXECUTOR);
    final RelNode left = join.getInput(0);
    final RelNode right = join.getInput(1);

    final RelOptPredicateList leftInfo = mq.getPulledUpPredicates(left);
    final RelOptPredicateList rightInfo = mq.getPulledUpPredicates(right);

    BodoJoinConditionBasedPredicateInference joinInference =
        new BodoJoinConditionBasedPredicateInference(
            join,
            RexUtil.composeConjunction(rexBuilder, leftInfo.pulledUpPredicates),
            RexUtil.composeConjunction(rexBuilder, rightInfo.pulledUpPredicates),
            new BodoRexSimplify(rexBuilder, RelOptPredicateList.EMPTY, executor));

    return joinInference.inferPredicates(false);
  }

  /**
   * Utility to infer predicates from one side of the join that apply on the other side.
   *
   * <p>Almost an exact copy of
   * org.apache.calcite.rel.metadata.RelMdPredicates.JoinConditionBasedPredicateInference
   */
  static class BodoJoinConditionBasedPredicateInference {
    final Join joinRel;
    final int nSysFields;
    final int nFieldsLeft;
    final int nFieldsRight;
    final ImmutableBitSet leftFieldsBitSet;
    final ImmutableBitSet rightFieldsBitSet;
    final ImmutableBitSet allFieldsBitSet;

    @SuppressWarnings("JdkObsolete")
    SortedMap<Integer, BitSet> equivalence;

    final Map<RexNode, ImmutableBitSet> exprFields;
    final Set<RexNode> allExprs;
    final Set<RexNode> equalityPredicates;
    final @Nullable RexNode leftChildPredicates;
    final @Nullable RexNode rightChildPredicates;
    final RexSimplify simplify;

    @SuppressWarnings("JdkObsolete")
    BodoJoinConditionBasedPredicateInference(
        Join joinRel,
        @Nullable RexNode leftPredicates,
        @Nullable RexNode rightPredicates,
        RexSimplify simplify) {
      super();
      this.joinRel = joinRel;
      this.simplify = simplify;
      nFieldsLeft = joinRel.getLeft().getRowType().getFieldList().size();
      nFieldsRight = joinRel.getRight().getRowType().getFieldList().size();
      nSysFields = joinRel.getSystemFieldList().size();
      leftFieldsBitSet = ImmutableBitSet.range(nSysFields, nSysFields + nFieldsLeft);
      rightFieldsBitSet =
          ImmutableBitSet.range(nSysFields + nFieldsLeft, nSysFields + nFieldsLeft + nFieldsRight);
      allFieldsBitSet = ImmutableBitSet.range(0, nSysFields + nFieldsLeft + nFieldsRight);

      exprFields = new HashMap<>();
      allExprs = new HashSet<>();

      if (leftPredicates == null) {
        leftChildPredicates = null;
      } else {
        Mappings.TargetMapping leftMapping =
            Mappings.createShiftMapping(nSysFields + nFieldsLeft, nSysFields, 0, nFieldsLeft);
        leftChildPredicates =
            leftPredicates.accept(new RexPermuteInputsShuttle(leftMapping, joinRel.getInput(0)));

        allExprs.add(leftChildPredicates);
        for (RexNode r : RelOptUtil.conjunctions(leftChildPredicates)) {
          exprFields.put(r, RelOptUtil.InputFinder.bits(r));
          allExprs.add(r);
        }
      }
      if (rightPredicates == null) {
        rightChildPredicates = null;
      } else {
        Mappings.TargetMapping rightMapping =
            Mappings.createShiftMapping(
                nSysFields + nFieldsLeft + nFieldsRight, nSysFields + nFieldsLeft, 0, nFieldsRight);
        rightChildPredicates =
            rightPredicates.accept(new RexPermuteInputsShuttle(rightMapping, joinRel.getInput(1)));

        allExprs.add(rightChildPredicates);
        for (RexNode r : RelOptUtil.conjunctions(rightChildPredicates)) {
          exprFields.put(r, RelOptUtil.InputFinder.bits(r));
          allExprs.add(r);
        }
      }

      equivalence = new TreeMap<>();
      equalityPredicates = new HashSet<>();
      for (int i = 0; i < nSysFields + nFieldsLeft + nFieldsRight; i++) {
        equivalence.put(i, BitSets.of(i));
      }

      // Only process equivalences found in the join conditions. Processing
      // Equivalences from the left or right side infer predicates that are
      // already present in the Tree below the join.
      List<RexNode> exprs = RelOptUtil.conjunctions(joinRel.getCondition());

      final BodoJoinConditionBasedPredicateInference.EquivalenceFinder eF =
          new BodoJoinConditionBasedPredicateInference.EquivalenceFinder();
      exprs.forEach(input -> input.accept(eF));

      equivalence = BitSets.closure(equivalence);
    }

    // BODO CHANGE:
    /**
     * As RexPermuteInputsShuttle, with one exception. When visiting an inputRef, it will replace
     * the type of the InputRef with the type found in the input fields, instead of keeping the
     * original type. This is used within when generating the Left/RightInferredPredicates, to avoid
     * nullability mismatches between the types of the join and the types of the inputs.
     */
    private class TypeChangingRexPermuteInputsShuttle
        extends org.apache.calcite.rex.RexPermuteInputsShuttle {

      private final Mappings.TargetMapping mapping;
      private final ImmutableList<RelDataTypeField> fields;

      public TypeChangingRexPermuteInputsShuttle(
          Mappings.TargetMapping mapping, RelNode... inputs) {
        super(mapping, inputs);
        this.mapping = mapping;
        this.fields = fields(inputs);
      }

      private ImmutableList<RelDataTypeField> fields(RelNode[] inputs) {
        final ImmutableList.Builder<RelDataTypeField> fields = ImmutableList.builder();
        for (RelNode input : inputs) {
          fields.addAll(input.getRowType().getFieldList());
        }
        return fields.build();
      }

      @Override
      public RexNode visitInputRef(RexInputRef local) {
        final int index = local.getIndex();
        int target = mapping.getTarget(index);
        RelDataType oldTyp = local.getType();
        RelDataType newTyp = fields.get(target).getType();
        return new RexInputRef(target, newTyp);
      }
    }

    /**
     * The PullUp Strategy is sound but not complete.
     *
     * <ol>
     *   <li>We only pullUp inferred predicates for now. Pulling up existing predicates causes an
     *       explosion of duplicates. The existing predicates are pushed back down as new
     *       predicates. Once we have rules to eliminate duplicate Filter conditions, we should
     *       pullUp all predicates.
     *   <li>For Left Outer: we infer new predicates from the left and set them as applicable on the
     *       Right side. No predicates are pulledUp.
     *   <li>Right Outer Joins are handled in an analogous manner.
     *   <li>For Full Outer Joins no predicates are pulledUp or inferred.
     * </ol>
     */
    public RelOptPredicateList inferPredicates(boolean includeEqualityInference) {
      final List<RexNode> inferredPredicates = new ArrayList<>();
      final Set<RexNode> allExprs = new HashSet<>(this.allExprs);
      final JoinRelType joinType = joinRel.getJoinType();
      switch (joinType) {
        case SEMI:
        case INNER:
        case LEFT:
        case ANTI:
          infer(
              leftChildPredicates,
              allExprs,
              inferredPredicates,
              includeEqualityInference,
              joinType == JoinRelType.LEFT ? rightFieldsBitSet : allFieldsBitSet);
          break;
        default:
          break;
      }
      switch (joinType) {
        case SEMI:
        case INNER:
        case RIGHT:
          infer(
              rightChildPredicates,
              allExprs,
              inferredPredicates,
              includeEqualityInference,
              joinType == JoinRelType.RIGHT ? leftFieldsBitSet : allFieldsBitSet);
          break;
        default:
          break;
      }

      Mappings.TargetMapping rightMapping =
          Mappings.createShiftMapping(
              nSysFields + nFieldsLeft + nFieldsRight, 0, nSysFields + nFieldsLeft, nFieldsRight);
      // BODO CHANGE: uses TypeChangingRexPermuteInputsShuttle instead of the default
      // RexPermuteInputsShuttle.
      final RexPermuteInputsShuttle rightPermute =
          new TypeChangingRexPermuteInputsShuttle(rightMapping, joinRel.getRight());
      Mappings.TargetMapping leftMapping =
          Mappings.createShiftMapping(nSysFields + nFieldsLeft, 0, nSysFields, nFieldsLeft);
      final RexPermuteInputsShuttle leftPermute =
          new TypeChangingRexPermuteInputsShuttle(leftMapping, joinRel.getLeft());
      final List<RexNode> leftInferredPredicates = new ArrayList<>();
      final List<RexNode> rightInferredPredicates = new ArrayList<>();

      for (RexNode iP : inferredPredicates) {
        ImmutableBitSet iPBitSet = RelOptUtil.InputFinder.bits(iP);
        if (leftFieldsBitSet.contains(iPBitSet)) {
          leftInferredPredicates.add(iP.accept(leftPermute));
        } else if (rightFieldsBitSet.contains(iPBitSet)) {
          rightInferredPredicates.add(iP.accept(rightPermute));
        }
      }

      final RexBuilder rexBuilder = joinRel.getCluster().getRexBuilder();
      switch (joinType) {
        case SEMI:
          Iterable<RexNode> pulledUpPredicates;
          pulledUpPredicates =
              Iterables.concat(
                  RelOptUtil.conjunctions(leftChildPredicates), leftInferredPredicates);
          return RelOptPredicateList.of(
              rexBuilder, pulledUpPredicates, leftInferredPredicates, rightInferredPredicates);
        case INNER:
          pulledUpPredicates =
              Iterables.concat(
                  RelOptUtil.conjunctions(leftChildPredicates),
                  RelOptUtil.conjunctions(rightChildPredicates),
                  RexUtil.retainDeterministic(RelOptUtil.conjunctions(joinRel.getCondition())),
                  inferredPredicates);
          return RelOptPredicateList.of(
              rexBuilder, pulledUpPredicates, leftInferredPredicates, rightInferredPredicates);
        case LEFT:
        case ANTI:
          return RelOptPredicateList.of(
              rexBuilder,
              RelOptUtil.conjunctions(leftChildPredicates),
              leftInferredPredicates,
              rightInferredPredicates);
        case RIGHT:
          return RelOptPredicateList.of(
              rexBuilder,
              RelOptUtil.conjunctions(rightChildPredicates),
              inferredPredicates,
              EMPTY_LIST);
        default:
          assert inferredPredicates.size() == 0;
          return RelOptPredicateList.EMPTY;
      }
    }

    public @Nullable RexNode left() {
      return leftChildPredicates;
    }

    public @Nullable RexNode right() {
      return rightChildPredicates;
    }

    private void infer(
        @Nullable RexNode predicates,
        Set<RexNode> allExprs,
        List<RexNode> inferredPredicates,
        boolean includeEqualityInference,
        ImmutableBitSet inferringFields) {
      List<RexNode> localInferredPredicates = new ArrayList<>();
      for (RexNode r : RelOptUtil.conjunctions(predicates)) {
        if (!includeEqualityInference && equalityPredicates.contains(r)) {
          continue;
        }
        for (Mapping m : mappings(r)) {
          RexNode tr =
              r.accept(new RexPermuteInputsShuttle(m, joinRel.getInput(0), joinRel.getInput(1)));
          // Filter predicates can be already simplified, so we should work with
          // simplified RexNode versions as well. It also allows prevent of having
          // some duplicates in in result pulledUpPredicates
          RexNode simplifiedTarget = simplify.simplifyFilterPredicates(RelOptUtil.conjunctions(tr));
          if (simplifiedTarget == null) {
            simplifiedTarget = joinRel.getCluster().getRexBuilder().makeLiteral(false);
          }
          if (checkTarget(inferringFields, allExprs, tr)
              && checkTarget(inferringFields, allExprs, simplifiedTarget)) {
            // Bodo Change: Write to an integer mediate target instead.
            localInferredPredicates.add(simplifiedTarget);
            allExprs.add(simplifiedTarget);
          }
        }
      }
      // Bodo Change: Ensure the predicates aren't added if all simplified together. If any
      // predicate matches
      // then we can update the inferred predicates. This won't catch every combination, as its
      // possible that
      // only some filters are new, but this should prevent repeatedly adding the same simplified
      // filters.
      if (localInferredPredicates.size() == 1
          || checkTarget(
              inferringFields,
              allExprs,
              simplify.simplifyFilterPredicates(localInferredPredicates))) {
        inferredPredicates.addAll(localInferredPredicates);
      }
    }

    Iterable<Mapping> mappings(final RexNode predicate) {
      final ImmutableBitSet fields =
          requireNonNull(
              exprFields.get(predicate),
              () -> "exprFields.get(predicate) is null for " + predicate);
      if (fields.cardinality() == 0) {
        return Collections.emptyList();
      }
      return () -> new BodoJoinConditionBasedPredicateInference.ExprsItr(fields);
    }

    private static boolean checkTarget(
        ImmutableBitSet inferringFields, Set<RexNode> allExprs, RexNode tr) {
      return inferringFields.contains(RelOptUtil.InputFinder.bits(tr))
          && !allExprs.contains(tr)
          && !isAlwaysTrue(tr);
    }

    @SuppressWarnings("JdkObsolete")
    private void markAsEquivalent(int p1, int p2) {
      BitSet b = requireNonNull(equivalence.get(p1), () -> "equivalence.get(p1) for " + p1);
      b.set(p2);

      b = requireNonNull(equivalence.get(p2), () -> "equivalence.get(p2) for " + p2);
      b.set(p1);
    }

    /** Find expressions of the form 'col_x = col_y'. */
    class EquivalenceFinder extends RexVisitorImpl<Void> {
      protected EquivalenceFinder() {
        super(true);
      }

      @Override
      public Void visitCall(RexCall call) {
        if (call.getOperator().getKind() == SqlKind.EQUALS) {
          int lPos = pos(call.getOperands().get(0));
          int rPos = pos(call.getOperands().get(1));
          if (lPos != -1 && rPos != -1) {
            markAsEquivalent(lPos, rPos);
            equalityPredicates.add(call);
          }
        }
        return null;
      }
    }

    /**
     * Given an expression returns all the possible substitutions.
     *
     * <p>For example, for an expression 'a + b + c' and the following equivalences:
     *
     * <pre>
     * a : {a, b}
     * b : {a, b}
     * c : {c, e}
     * </pre>
     *
     * <p>The following Mappings will be returned:
     *
     * <pre>
     * {a &rarr; a, b &rarr; a, c &rarr; c}
     * {a &rarr; a, b &rarr; a, c &rarr; e}
     * {a &rarr; a, b &rarr; b, c &rarr; c}
     * {a &rarr; a, b &rarr; b, c &rarr; e}
     * {a &rarr; b, b &rarr; a, c &rarr; c}
     * {a &rarr; b, b &rarr; a, c &rarr; e}
     * {a &rarr; b, b &rarr; b, c &rarr; c}
     * {a &rarr; b, b &rarr; b, c &rarr; e}
     * </pre>
     *
     * <p>which imply the following inferences:
     *
     * <pre>
     * a + a + c
     * a + a + e
     * a + b + c
     * a + b + e
     * b + a + c
     * b + a + e
     * b + b + c
     * b + b + e
     * </pre>
     */
    class ExprsItr implements Iterator<Mapping> {
      final int[] columns;
      final BitSet[] columnSets;
      final int[] iterationIdx;
      @Nullable Mapping nextMapping;
      boolean firstCall;

      @SuppressWarnings("JdkObsolete")
      ExprsItr(ImmutableBitSet fields) {
        nextMapping = null;
        columns = new int[fields.cardinality()];
        columnSets = new BitSet[fields.cardinality()];
        iterationIdx = new int[fields.cardinality()];
        for (int j = 0, i = fields.nextSetBit(0); i >= 0; i = fields.nextSetBit(i + 1), j++) {
          columns[j] = i;
          int fieldIndex = i;
          columnSets[j] =
              requireNonNull(
                  equivalence.get(i),
                  () -> "equivalence.get(i) is null for " + fieldIndex + ", " + equivalence);
          iterationIdx[j] = 0;
        }
        firstCall = true;
      }

      @Override
      public boolean hasNext() {
        if (firstCall) {
          initializeMapping();
          firstCall = false;
        } else {
          computeNextMapping(iterationIdx.length - 1);
        }
        return nextMapping != null;
      }

      @Override
      public Mapping next() {
        if (nextMapping == null) {
          throw new NoSuchElementException();
        }
        return nextMapping;
      }

      @Override
      public void remove() {
        throw new UnsupportedOperationException();
      }

      private void computeNextMapping(int level) {
        int t = columnSets[level].nextSetBit(iterationIdx[level]);
        if (t < 0) {
          if (level == 0) {
            nextMapping = null;
          } else {
            int tmp = columnSets[level].nextSetBit(0);
            requireNonNull(nextMapping, "nextMapping").set(columns[level], tmp);
            iterationIdx[level] = tmp + 1;
            computeNextMapping(level - 1);
          }
        } else {
          requireNonNull(nextMapping, "nextMapping").set(columns[level], t);
          iterationIdx[level] = t + 1;
        }
      }

      private void initializeMapping() {
        nextMapping =
            Mappings.create(
                MappingType.PARTIAL_FUNCTION,
                nSysFields + nFieldsLeft + nFieldsRight,
                nSysFields + nFieldsLeft + nFieldsRight);
        for (int i = 0; i < columnSets.length; i++) {
          BitSet c = columnSets[i];
          int t = c.nextSetBit(iterationIdx[i]);
          if (t < 0) {
            nextMapping = null;
            return;
          }
          nextMapping.set(columns[i], t);
          iterationIdx[i] = t + 1;
        }
      }
    }

    private static int pos(RexNode expr) {
      if (expr instanceof RexInputRef) {
        return ((RexInputRef) expr).getIndex();
      }
      return -1;
    }

    private static boolean isAlwaysTrue(RexNode predicate) {
      if (predicate instanceof RexCall) {
        RexCall c = (RexCall) predicate;
        if (c.getOperator().getKind() == SqlKind.EQUALS) {
          int lPos = pos(c.getOperands().get(0));
          int rPos = pos(c.getOperands().get(1));
          return lPos != -1 && lPos == rPos;
        }
      }
      return predicate.isAlwaysTrue();
    }
  }
}
