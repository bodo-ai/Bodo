package com.bodosql.calcite.adapter.iceberg;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.BodoLogicalRelFactories;
import org.apache.calcite.rel.core.Filter;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class IcebergFilterRule extends AbstractIcebergFilterRule {
  protected IcebergFilterRule(@NotNull IcebergFilterRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractIcebergFilterRule.Config {
    IcebergFilterRule.Config DEFAULT_CONFIG =
        ImmutableIcebergFilterRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Filter.class)
                        .predicate(AbstractIcebergFilterRule::isPartiallyPushableFilter)
                        .oneInput(
                            b1 ->
                                b1.operand(IcebergToPandasConverter.class)
                                    .oneInput(
                                        b2 ->
                                            b2.operand(IcebergRel.class)
                                                .predicate(x -> !x.containsIcebergSort())
                                                .anyInputs())))
            .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
            .as(IcebergFilterRule.Config.class);

    @Override
    default @NotNull IcebergFilterRule toRule() {
      return new IcebergFilterRule(this);
    }
  }
}
