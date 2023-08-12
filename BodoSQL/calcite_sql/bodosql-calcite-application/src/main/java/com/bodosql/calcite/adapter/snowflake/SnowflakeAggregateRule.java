package com.bodosql.calcite.adapter.snowflake;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.traits.CombineStreamsExchange;
import org.apache.calcite.rel.core.Aggregate;
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
    Config DEFAULT_CONFIG =
        ImmutableSnowflakeAggregateRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Aggregate.class)
                        .predicate(SnowflakeAggregateRule::isPushableAggregate)
                        .oneInput(b1 -> b1.operand(SnowflakeRel.class).anyInputs()))
            .as(Config.class);

    Config NESTED_CONFIG =
        ImmutableSnowflakeAggregateRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Aggregate.class)
                        .predicate(SnowflakeAggregateRule::isPushableAggregate)
                        .oneInput(
                            b1 ->
                                b1.operand(SnowflakeToPandasConverter.class)
                                    .oneInput(b2 -> b2.operand(SnowflakeRel.class).anyInputs())))
            .withDescription("SnowflakeAggregateRule::WithSnowflakeToPandasConverter")
            .as(Config.class);

    Config STREAMING_CONFIG =
        ImmutableSnowflakeAggregateRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Aggregate.class)
                        .predicate(SnowflakeAggregateRule::isPushableAggregate)
                        .oneInput(
                            b1 ->
                                b1.operand(CombineStreamsExchange.class)
                                    .oneInput(
                                        b2 ->
                                            b2.operand(SnowflakeToPandasConverter.class)
                                                .oneInput(
                                                    b3 ->
                                                        b3.operand(SnowflakeRel.class)
                                                            .anyInputs()))))
            .withDescription("SnowflakeAggregateRule::WithCombineStreamsExchange")
            .as(Config.class);

    @Override
    default @NotNull SnowflakeAggregateRule toRule() {
      return new SnowflakeAggregateRule(this);
    }
  }
}
