package com.bodosql.calcite.adapter.snowflake;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import org.apache.calcite.rel.core.Project;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

/**
 * Instantiation of an AbstractSnowflakeProjectRule. See AbstractSnowflakeProjectRule for
 * information on what this rule does.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class SnowflakeProjectLockRule extends AbstractSnowflakeProjectRule {
  protected SnowflakeProjectLockRule(@NotNull SnowflakeProjectLockRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractSnowflakeProjectRule.Config {

    // Inputs are:
    // project ->
    //      SnowflakeRel
    Config DEFAULT_CONFIG =
        ImmutableSnowflakeProjectLockRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Project.class)
                        .predicate(
                            AbstractSnowflakeProjectRule
                                ::canPushNonTrivialSnowflakeProjectAndProjectNotAlreadySnowflake)
                        .oneInput(b1 -> b1.operand(SnowflakeRel.class).anyInputs()))
            .as(Config.class);

    @Override
    default @NotNull SnowflakeProjectLockRule toRule() {
      return new SnowflakeProjectLockRule(this);
    }
  }
}
