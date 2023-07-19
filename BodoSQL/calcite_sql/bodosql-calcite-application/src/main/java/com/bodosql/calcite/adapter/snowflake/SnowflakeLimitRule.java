package com.bodosql.calcite.adapter.snowflake;

import com.bodosql.calcite.application.Utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.logical.BodoLogicalFilter;

import org.apache.calcite.rel.logical.LogicalSort;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class SnowflakeLimitRule extends AbstractSnowflakeLimitRule {
  protected SnowflakeLimitRule(@NotNull SnowflakeLimitRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractSnowflakeLimitRule.Config {
    Config DEFAULT =
        ImmutableSnowflakeLimitRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(LogicalSort.class)
                        .predicate(SnowflakeLimitRule::isOnlyLimit)
                        .oneInput(b1 -> b1.operand(SnowflakeTableScan.class).noInputs()))
            .as(Config.class);

    Config WITH_FILTER =
        ImmutableSnowflakeLimitRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(LogicalSort.class)
                        .predicate(SnowflakeLimitRule::isOnlyLimit)
                        .oneInput(
                            b1 ->
                                b1.operand(BodoLogicalFilter.class)
                                    .predicate(SnowflakeAggregateRule::isPushableFilter)
                                    .oneInput(
                                        b2 -> b2.operand(SnowflakeTableScan.class).noInputs())))
            .withDescription("SnowflakeLimitRule::WithFilter")
            .as(Config.class);

    @Override
    default @NotNull SnowflakeLimitRule toRule() {
      return new SnowflakeLimitRule(this);
    }
  }
}
