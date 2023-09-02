package com.bodosql.calcite.adapter.snowflake;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.traits.CombineStreamsExchange;
import org.apache.calcite.rel.core.Sort;
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
    Config DEFAULT_CONFIG =
        ImmutableSnowflakeLimitRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Sort.class)
                        .predicate(SnowflakeLimitRule::isOnlyLimit)
                        .oneInput(
                            b1 ->
                                b1.operand(SnowflakeToPandasConverter.class)
                                    .oneInput(b2 -> b2.operand(SnowflakeRel.class).anyInputs())))
            .as(Config.class);

    Config STREAMING_CONFIG =
        ImmutableSnowflakeLimitRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Sort.class)
                        .predicate(SnowflakeLimitRule::isOnlyLimit)
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
            .withDescription("SnowflakeLimitRule::WithCombineStreamsExchange")
            .as(Config.class);

    @Override
    default @NotNull SnowflakeLimitRule toRule() {
      return new SnowflakeLimitRule(this);
    }
  }
}
