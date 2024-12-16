package com.bodosql.calcite.adapter.snowflake;

import com.bodosql.calcite.adapter.common.LimitUtils;
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.BodoLogicalRelFactories;
import com.bodosql.calcite.rel.logical.BodoLogicalSort;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class SnowflakeLimitLockRule extends AbstractSnowflakeLimitRule {
  protected SnowflakeLimitLockRule(@NotNull SnowflakeLimitLockRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractSnowflakeLimitRule.Config {
    Config DEFAULT_CONFIG =
        ImmutableSnowflakeLimitLockRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(BodoLogicalSort.class)
                        .predicate(LimitUtils::isOnlyLimit)
                        .oneInput(b1 -> b1.operand(SnowflakeRel.class).anyInputs()))
            .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
            .as(Config.class);

    @Override
    default @NotNull SnowflakeLimitLockRule toRule() {
      return new SnowflakeLimitLockRule(this);
    }
  }
}
