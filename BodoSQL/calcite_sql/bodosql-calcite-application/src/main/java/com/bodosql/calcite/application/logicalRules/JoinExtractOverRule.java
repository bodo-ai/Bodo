package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rex.RexOver;
import org.immutables.value.Value;
import org.jetbrains.annotations.NotNull;

@BodoSQLStyleImmutable
@Value.Enclosing
public class JoinExtractOverRule extends AbstractJoinExtractOverRule {
  protected JoinExtractOverRule(@NotNull AbstractJoinExtractOverRule.Config config) {
    super(config);
  }

  @Value.Immutable
  public interface Config extends AbstractJoinExtractOverRule.Config {
    JoinExtractOverRule.Config DEFAULT =
        ImmutableJoinExtractOverRule.Config.of()
            .withOperandSupplier(
                b ->
                    b.operand(LogicalJoin.class)
                        .predicate(join -> RexOver.containsOver(join.getCondition()))
                        .anyInputs())
            .as(JoinExtractOverRule.Config.class);

    @Override
    default JoinExtractOverRule toRule() {
      return new JoinExtractOverRule(this);
    }
  }
}
