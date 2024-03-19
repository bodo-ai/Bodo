package com.bodo.iceberg.filters;

import org.apache.iceberg.expressions.Literal;

public class Const extends Filter {
  public final Literal<Object> value;

  public Const(Literal value) {
    this.value = (Literal<Object>) value;
  }
}
