package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import org.apache.calcite.rel.core.Project;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class BodoCommonSubexpressionRule extends AbstractBodoCommonSubexpressionRule {
  protected BodoCommonSubexpressionRule(
      @NotNull AbstractBodoCommonSubexpressionRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractBodoCommonSubexpressionRule.Config {
    BodoCommonSubexpressionRule.Config DEFAULT =
        ImmutableBodoCommonSubexpressionRule.Config.of()
            .withOperandSupplier(b -> b.operand(Project.class).anyInputs())
            .as(BodoCommonSubexpressionRule.Config.class);

    @Override
    default BodoCommonSubexpressionRule toRule() {
      return new BodoCommonSubexpressionRule(this);
    }
  }
}
