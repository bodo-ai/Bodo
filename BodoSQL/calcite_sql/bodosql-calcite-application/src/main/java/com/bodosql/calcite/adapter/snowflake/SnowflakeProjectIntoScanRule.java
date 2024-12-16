package com.bodosql.calcite.adapter.snowflake;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.BodoLogicalRelFactories;
import com.bodosql.calcite.rel.logical.BodoLogicalProject;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class SnowflakeProjectIntoScanRule extends AbstractSnowflakeProjectIntoScanRule {
  protected SnowflakeProjectIntoScanRule(@NotNull Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractSnowflakeProjectIntoScanRule.Config {
    // Rule for pushing a logical projection into Snowflake. This runs before
    // the physical stage.
    Config BODO_LOGICAL_CONFIG =
        ImmutableSnowflakeProjectIntoScanRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(BodoLogicalProject.class)
                        .predicate(AbstractSnowflakeProjectIntoScanRule::isApplicable)
                        .oneInput(b1 -> b1.operand(SnowflakeTableScan.class).noInputs()))
            .withRelBuilderFactory(BodoLogicalRelFactories.BODO_LOGICAL_BUILDER)
            .as(Config.class);

    @Override
    default @NotNull SnowflakeProjectIntoScanRule toRule() {
      return new SnowflakeProjectIntoScanRule(this);
    }
  }
}
