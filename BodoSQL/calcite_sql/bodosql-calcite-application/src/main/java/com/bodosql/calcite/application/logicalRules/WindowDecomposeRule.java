package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import org.apache.calcite.rel.core.Project;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class WindowDecomposeRule extends AbstractWindowDecomposeRule {
  protected WindowDecomposeRule(@NotNull AbstractWindowDecomposeRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractWindowDecomposeRule.Config {
    WindowDecomposeRule.Config DEFAULT =
        ImmutableWindowDecomposeRule.Config.of()
            .withOperandSupplier(
                b ->
                    b.operand(Project.class)
                        .predicate(project -> project.containsOver())
                        .anyInputs())
            .as(WindowDecomposeRule.Config.class);

    @Override
    default WindowDecomposeRule toRule() {
      return new WindowDecomposeRule(this);
    }
  }
}
