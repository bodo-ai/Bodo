package com.bodosql.calcite.adapter.snowflake;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.traits.CombineStreamsExchange;
import org.apache.calcite.rel.core.Project;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

/**
 * Instantiation of an AbstractSnowflakeProjectRule. See AbstractSnowflakeProjectRule for
 * information on what this rule does.
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class SnowflakeProjectRule extends AbstractSnowflakeProjectRule {
  protected SnowflakeProjectRule(@NotNull SnowflakeProjectRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractSnowflakeProjectRule.Config {

    // Inputs are:
    // project ->
    //   SnowflakeToPandasConverter ->
    //      SnowflakeRel
    Config DEFAULT_CONFIG =
        ImmutableSnowflakeProjectRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Project.class)
                        .predicate(AbstractSnowflakeProjectRule::isPushableProject)
                        .oneInput(
                            b1 ->
                                b1.operand(SnowflakeToPandasConverter.class)
                                    .oneInput(b2 -> b2.operand(SnowflakeRel.class).anyInputs())))
            .as(Config.class);

    // This configuration is done to handle projections with window functions
    // that can look past CombineStreamsExchange.
    Config STREAMING_CONFIG =
        ImmutableSnowflakeProjectRule.Config.of()
            .withOperandSupplier(
                b0 ->
                    b0.operand(Project.class)
                        .predicate(AbstractSnowflakeProjectRule::isPushableProject)
                        .oneInput(
                            b1 ->
                                b1.operand(CombineStreamsExchange.class)
                                    .oneInput(
                                        b2 ->
                                            b2.operand(SnowflakeToPandasConverter.class)
                                                .oneInput(
                                                    b3 ->
                                                        b3.operand(SnowflakeRel.class)
                                                            .anyInputs()))))
            .withDescription("SnowflakeProjectRule::WithCombineStreamsExchange")
            .as(Config.class);

    @Override
    default @NotNull SnowflakeProjectRule toRule() {
      return new SnowflakeProjectRule(this);
    }
  }
}
