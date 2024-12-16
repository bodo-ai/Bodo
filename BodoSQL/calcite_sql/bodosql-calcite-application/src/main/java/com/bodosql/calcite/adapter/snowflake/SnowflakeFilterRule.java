package com.bodosql.calcite.adapter.snowflake;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.BodoLogicalRelFactories;
import org.apache.calcite.rel.core.Filter;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class SnowflakeFilterRule extends AbstractSnowflakeFilterRule {
  protected SnowflakeFilterRule(@NotNull SnowflakeFilterRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractSnowflakeFilterRule.Config {
    Config DEFAULT_CONFIG =
        ImmutableSnowflakeFilterRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Filter.class)
                        .predicate(AbstractSnowflakeFilterRule::isPartiallyPushableFilter)
                        .oneInput(
                            b1 ->
                                b1.operand(SnowflakeToBodoPhysicalConverter.class)
                                    .oneInput(b2 -> b2.operand(SnowflakeRel.class).anyInputs())))
            .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
            .as(Config.class);

    @Override
    default @NotNull SnowflakeFilterRule toRule() {
      return new SnowflakeFilterRule(this);
    }
  }
}
