package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.operatorTables.AggOperatorTable;
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.rules.TransformationRule;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Pair;
import org.immutables.value.Value;

/**
 * Rule that transforms an Aggregate with a filter into a case statement that converts the false
 * entries into null, provided the function ignores nulls.
 *
 * <p>The rule doesn't contain any logic to "fuse" the filter calculation with a prior projection
 * and that would need to be handled by a separate project merge rule.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class AggregateFilterToCaseRule extends RelRule<AggregateFilterToCaseRule.Config>
    implements TransformationRule {

  /** Creates an AggregateFilterToCaseRule. */
  protected AggregateFilterToCaseRule(AggregateFilterToCaseRule.Config config) {
    super(config);
  }

  static final Set<SqlAggFunction> supportedAggregateFunctions =
      Set.of(
          SqlStdOperatorTable.COUNT,
          SqlStdOperatorTable.SUM,
          SqlStdOperatorTable.SUM0,
          SqlStdOperatorTable.MIN,
          SqlStdOperatorTable.MAX,
          SqlStdOperatorTable.AVG,
          SqlStdOperatorTable.STDDEV_POP,
          SqlStdOperatorTable.STDDEV_SAMP,
          SqlStdOperatorTable.STDDEV,
          SqlStdOperatorTable.VAR_POP,
          SqlStdOperatorTable.VAR_SAMP,
          SqlStdOperatorTable.VARIANCE,
          AggOperatorTable.VARIANCE_POP,
          AggOperatorTable.VARIANCE_SAMP,
          AggOperatorTable.KURTOSIS,
          AggOperatorTable.SKEW,
          AggOperatorTable.CORR,
          AggOperatorTable.APPROX_PERCENTILE,
          AggOperatorTable.MEDIAN,
          AggOperatorTable.COUNT_IF,
          SqlStdOperatorTable.PERCENTILE_CONT,
          SqlStdOperatorTable.PERCENTILE_DISC,
          AggOperatorTable.ARRAY_AGG,
          AggOperatorTable.ARRAY_UNIQUE_AGG,
          SqlStdOperatorTable.LISTAGG,
          AggOperatorTable.OBJECT_AGG,
          AggOperatorTable.BITAND_AGG,
          AggOperatorTable.BITOR_AGG,
          AggOperatorTable.BITXOR_AGG,
          AggOperatorTable.BOOLOR_AGG,
          AggOperatorTable.BOOLAND_AGG,
          AggOperatorTable.BOOLXOR_AGG);

  boolean isSupportedFilterAggregateCall(AggregateCall aggCall) {
    // Note: Arg list must not be empty because we cannot support
    // COUNT(*) with a filter by this method. This would require a separate
    // rewrite that replaces COUNT(*) filter COL with COUNT(FILTER).
    return supportedAggregateFunctions.contains(aggCall.getAggregation())
        && aggCall.hasFilter()
        && !aggCall.getArgList().isEmpty();
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    final Aggregate aggregate = call.rel(0);
    final RelBuilder builder = call.builder();
    builder.push(aggregate.getInput());
    // Cache each input argument to its new location in the projection. An argument
    // used with multiple filters needs to be duplicated, so the second part of the
    // key is the original filter index, or -1 if there is no filter.
    HashMap<Pair<Integer, Integer>, Integer> argumentCache = new HashMap();
    List<RexNode> fields = new ArrayList();
    int columnIndex = 0;
    // Add the grouping sets.
    for (int i : aggregate.getGroupSet()) {
      fields.add(builder.field(i));
      argumentCache.put(Pair.of(i, -1), columnIndex++);
    }
    // Add the aggregate functions.
    boolean changed = false;
    for (AggregateCall agg : aggregate.getAggCallList()) {
      if (isSupportedFilterAggregateCall(agg)) {
        changed = true;
        // Each argument can be converted to a case statement.
        int filterArg = agg.filterArg;
        for (int arg : agg.getArgList()) {
          Pair<Integer, Integer> key = Pair.of(arg, filterArg);
          if (!argumentCache.containsKey(key)) {
            // Build the case statement.
            RexNode cond = builder.field(filterArg);
            RexNode trueNode = builder.field(arg);
            RexNode falseNode = builder.literal(null);
            RexNode caseNode =
                builder
                    .getRexBuilder()
                    .makeCall(SqlStdOperatorTable.CASE, cond, trueNode, falseNode);
            fields.add(caseNode);
            argumentCache.put(key, columnIndex++);
          }
        }
      } else {
        // Push each argument without filtering.
        for (int arg : agg.getArgList()) {
          Pair<Integer, Integer> key = Pair.of(arg, -1);
          if (!argumentCache.containsKey(key)) {
            fields.add(builder.field(arg));
            argumentCache.put(key, columnIndex++);
          }
        }
        // Push the filter.
        if (agg.hasFilter()) {
          fields.add(builder.field(agg.filterArg));
          argumentCache.put(Pair.of(agg.filterArg, -1), columnIndex++);
        }
      }
    }
    if (!changed) {
      return;
    }
    builder.project(fields);
    // Build the new aggregate.
    ImmutableBitSet.Builder groupBuilder = ImmutableBitSet.builder();
    for (int i = 0; i < aggregate.getGroupCount(); i++) {
      groupBuilder.set(i);
    }
    ImmutableBitSet groupKeys = groupBuilder.build();
    List<RelBuilder.AggCall> aggCalls = new ArrayList();
    for (AggregateCall agg : aggregate.getAggCallList()) {
      final List<RexNode> newArgs;
      final boolean keepFilter;
      if (isSupportedFilterAggregateCall(agg)) {
        keepFilter = false;
        int filterArg = agg.filterArg;
        newArgs =
            agg.getArgList().stream()
                .map(i -> builder.field(argumentCache.get(Pair.of(i, filterArg))))
                .collect(Collectors.toList());
      } else {
        keepFilter = agg.hasFilter();
        newArgs =
            agg.getArgList().stream()
                .map(i -> builder.field(argumentCache.get(Pair.of(i, -1))))
                .collect(Collectors.toList());
      }
      RelBuilder.AggCall newAggCall =
          builder
              .aggregateCall(agg.getAggregation(), newArgs)
              .distinct(agg.isDistinct())
              .approximate(agg.isApproximate())
              .ignoreNulls(agg.ignoreNulls())
              .sort(agg.collation);
      if (keepFilter) {
        newAggCall =
            newAggCall.filter(builder.field(argumentCache.get(Pair.of(agg.filterArg, -1))));
      }
      aggCalls.add(newAggCall);
    }
    builder.aggregate(builder.groupKey(groupKeys), aggCalls);
    RelNode newAggregate = builder.build();
    call.transformTo(newAggregate);
  }

  public static boolean isSupportedFilterAggregate(Aggregate aggregate) {
    return Aggregate.isSimple(aggregate)
        && aggregate.getAggCallList().stream().anyMatch(aggCall -> aggCall.hasFilter());
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface Config extends RelRule.Config {
    AggregateFilterToCaseRule.Config DEFAULT =
        ImmutableAggregateFilterToCaseRule.Config.of().withOperandFor(Aggregate.class);

    @Override
    default AggregateFilterToCaseRule toRule() {
      return new AggregateFilterToCaseRule(this);
    }

    /** Defines an operand tree for the given classes. */
    default AggregateFilterToCaseRule.Config withOperandFor(
        Class<? extends Aggregate> aggregateClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(aggregateClass)
                      .predicate(AggregateFilterToCaseRule::isSupportedFilterAggregate)
                      .anyInputs())
          .as(AggregateFilterToCaseRule.Config.class);
    }
  }
}
