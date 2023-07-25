package com.bodosql.calcite.rex;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexDynamicParam;
import org.apache.calcite.rex.RexNode;

import com.bodosql.calcite.sql.fun.SqlNamedParameterOperator;
import com.google.common.collect.ImmutableList;

import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.List;
import java.util.Objects;

public class RexNamedParam extends RexCall {

  private final String name;

  public RexNamedParam(RelDataType type, String name) {
    // Not sure if it's better to store the name with the value itself or to keep
    // it as part of the operand list.
    // It's a bit easier to handle this without the operand list and the operand
    // list is really only helpful during validation so just going with this.
    super(type, SqlNamedParameterOperator.INSTANCE, ImmutableList.of());
    this.name = name;
  }

  public String getName() {
    return name;
  }

  @Override
  public RexCall clone(RelDataType type, List<RexNode> operands) {
    return new RexNamedParam(type, name);
  }

  @Override
  protected String computeDigest(boolean withType) {
    return "@" + name;
  }

  @Override public boolean equals(@Nullable Object obj) {
    return this == obj
        || obj instanceof RexNamedParam
        && name.equals(((RexNamedParam) obj).name)
        && super.equals(obj);
  }

  @Override public int hashCode() {
    return Objects.hash(type, "@", name);
  }
}
