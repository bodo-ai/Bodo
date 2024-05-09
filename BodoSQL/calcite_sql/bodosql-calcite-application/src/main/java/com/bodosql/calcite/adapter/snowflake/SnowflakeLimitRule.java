package com.bodosql.calcite.adapter.snowflake;

import com.bodosql.calcite.adapter.common.LimitUtils;
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
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
                        .predicate(LimitUtils::isOnlyLimit)
                        .oneInput(
                            b1 ->
                                b1.operand(SnowflakeToBodoPhysicalConverter.class)
                                    .oneInput(b2 -> b2.operand(SnowflakeRel.class).anyInputs())))
            .as(Config.class);

    @Override
    default @NotNull SnowflakeLimitRule toRule() {
      return new SnowflakeLimitRule(this);
    }
  }
}
