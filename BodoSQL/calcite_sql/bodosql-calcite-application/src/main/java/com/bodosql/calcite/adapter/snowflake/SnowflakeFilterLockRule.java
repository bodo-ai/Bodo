package com.bodosql.calcite.adapter.snowflake;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.BodoLogicalRelFactories;
import com.bodosql.calcite.rel.logical.BodoLogicalFilter;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class SnowflakeFilterLockRule extends AbstractSnowflakeFilterRule {
  protected SnowflakeFilterLockRule(@NotNull SnowflakeFilterLockRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractSnowflakeFilterRule.Config {
    Config DEFAULT_CONFIG =
        ImmutableSnowflakeFilterLockRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(BodoLogicalFilter.class)
                        .predicate(AbstractSnowflakeFilterRule::isPartiallyPushableFilter)
                        .oneInput(b1 -> b1.operand(SnowflakeRel.class).anyInputs()))
            .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
            .as(Config.class);

    @Override
    default @NotNull SnowflakeFilterLockRule toRule() {
      return new SnowflakeFilterLockRule(this);
    }
  }
}
