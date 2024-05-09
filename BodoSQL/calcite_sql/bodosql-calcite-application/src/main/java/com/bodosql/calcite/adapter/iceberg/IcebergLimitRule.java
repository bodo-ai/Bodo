package com.bodosql.calcite.adapter.iceberg;

import com.bodosql.calcite.adapter.common.LimitUtils;
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import org.apache.calcite.rel.core.Sort;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class IcebergLimitRule extends AbstractIcebergLimitRule {
  protected IcebergLimitRule(@NotNull IcebergLimitRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractIcebergLimitRule.Config {
    IcebergLimitRule.Config DEFAULT_CONFIG =
        ImmutableIcebergLimitRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Sort.class)
                        .predicate(LimitUtils::isOnlyLimit)
                        .oneInput(
                            b1 ->
                                b1.operand(IcebergToBodoPhysicalConverter.class)
                                    .oneInput(b2 -> b2.operand(IcebergRel.class).anyInputs())))
            .as(IcebergLimitRule.Config.class);

    @Override
    default @NotNull IcebergLimitRule toRule() {
      return new IcebergLimitRule(this);
    }
  }
}
