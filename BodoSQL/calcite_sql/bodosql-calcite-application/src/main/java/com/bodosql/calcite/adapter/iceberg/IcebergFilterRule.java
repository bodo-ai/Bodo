package com.bodosql.calcite.adapter.iceberg;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.BodoLogicalRelFactories;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelRule;
import org.apache.calcite.rel.core.Filter;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class IcebergFilterRule extends RelRule<IcebergFilterRule.Config> {
  protected IcebergFilterRule(@NotNull IcebergFilterRule.Config config) {
    super(config);
  }

  @Override
  public void onMatch(RelOptRuleCall call) {
    AbstractIcebergFilterRuleHelpers.onMatch(call, config.isPartialPushdown());
  }

  @Value.Immutable
  public interface Config extends RelRule.Config {
    IcebergFilterRule.Config DEFAULT_CONFIG =
        ImmutableIcebergFilterRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Filter.class)
                        .oneInput(
                            b1 ->
                                b1.operand(IcebergToBodoPhysicalConverter.class)
                                    .oneInput(b2 -> b2.operand(IcebergRel.class).anyInputs())))
            .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
            .as(IcebergFilterRule.Config.class);

    @Value.Default
    default boolean isPartialPushdown() {
      return false;
    }

    IcebergFilterRule.Config withPartialPushdown(boolean partialPushdown);

    @Override
    default @NotNull IcebergFilterRule toRule() {
      return new IcebergFilterRule(this);
    }
  }
}
