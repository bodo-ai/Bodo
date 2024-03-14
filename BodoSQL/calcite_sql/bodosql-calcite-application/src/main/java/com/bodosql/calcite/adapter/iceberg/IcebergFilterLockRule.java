package com.bodosql.calcite.adapter.iceberg;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.BodoLogicalRelFactories;
import com.bodosql.calcite.rel.logical.BodoLogicalFilter;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class IcebergFilterLockRule extends AbstractIcebergFilterRule {
  protected IcebergFilterLockRule(@NotNull IcebergFilterLockRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractIcebergFilterRule.Config {
    IcebergFilterLockRule.Config DEFAULT_CONFIG =
        ImmutableIcebergFilterLockRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(BodoLogicalFilter.class)
                        .oneInput(
                            b1 ->
                                b1.operand(IcebergRel.class)
                                    .predicate(x -> !x.containsIcebergSort())
                                    .anyInputs()))
            .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
            .as(IcebergFilterLockRule.Config.class);

    @Override
    default @NotNull IcebergFilterLockRule toRule() {
      return new IcebergFilterLockRule(this);
    }
  }
}
