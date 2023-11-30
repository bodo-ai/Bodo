package com.bodosql.calcite.adapter.snowflake;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.BodoLogicalRelFactories;
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
                        .oneInput(
                            b1 ->
                                b1.operand(SnowflakeToPandasConverter.class)
                                    .oneInput(b2 -> b2.operand(SnowflakeRel.class).anyInputs())))
            .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
            .as(Config.class);

    @Override
    default @NotNull SnowflakeAggregateRule toRule() {
      return new SnowflakeAggregateRule(this);
    }
  }
}
