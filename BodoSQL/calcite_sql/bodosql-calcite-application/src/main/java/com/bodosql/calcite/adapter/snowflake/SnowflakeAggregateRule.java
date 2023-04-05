package com.bodosql.calcite.adapter.snowflake;

import com.bodosql.calcite.application.Utils.BodoSQLStyleImmutable;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class SnowflakeAggregateRule extends AbstractSnowflakeAggregateRule {
  protected SnowflakeAggregateRule(@NotNull SnowflakeAggregateRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractSnowflakeAggregateRule.Config {
    Config DEFAULT =
        ImmutableSnowflakeAggregateRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(LogicalAggregate.class)
                        .predicate(SnowflakeAggregateRule::isPushableAggregate)
                        .oneInput(b1 -> b1.operand(SnowflakeTableScan.class).noInputs()));

    Config WITH_FILTER =
        ImmutableSnowflakeAggregateRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(LogicalAggregate.class)
                        .predicate(SnowflakeAggregateRule::isPushableAggregate)
                        .oneInput(
                            b1 ->
                                b1.operand(LogicalFilter.class)
                                    .predicate(SnowflakeAggregateRule::isPushableFilter)
                                    .oneInput(
                                        b2 -> b2.operand(SnowflakeTableScan.class).noInputs())))
            .withDescription("SnowflakeAggregateRule::WithFilter");

    @Override
    default @NotNull SnowflakeAggregateRule toRule() {
      return new SnowflakeAggregateRule(this);
    }
  }
}
