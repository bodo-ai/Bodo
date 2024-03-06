package com.bodosql.calcite.adapter.iceberg;

import com.bodosql.calcite.adapter.common.LimitUtils;
import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.BodoLogicalRelFactories;
import com.bodosql.calcite.rel.logical.BodoLogicalSort;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class IcebergLimitLockRule extends AbstractIcebergLimitRule {
  protected IcebergLimitLockRule(@NotNull IcebergLimitLockRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractIcebergLimitRule.Config {
    Config DEFAULT_CONFIG =
        ImmutableIcebergLimitLockRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(BodoLogicalSort.class)
                        .predicate(LimitUtils::isOnlyLimit)
                        .oneInput(
                            b1 ->
                                b1.operand(IcebergRel.class)
                                    .predicate(x -> !x.containsIcebergSort())
                                    .anyInputs()))
            .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
            .as(IcebergLimitLockRule.Config.class);

    @Override
    default @NotNull IcebergLimitLockRule toRule() {
      return new IcebergLimitLockRule(this);
    }
  }
}
