package com.bodosql.calcite.application.bodo_sql_rules;

import com.bodosql.calcite.application.Utils.BodoSQLStyleImmutable;
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
    Config DEFAULT =
        ImmutableJoinExtractOverRule.Config.of()
            .withOperandSupplier(
                b ->
                    b.operand(LogicalJoin.class)
                        .predicate(join -> RexOver.containsOver(join.getCondition()))
                        .anyInputs());

    @Override
    default JoinExtractOverRule toRule() {
      return new JoinExtractOverRule(this);
    }
  }
}
