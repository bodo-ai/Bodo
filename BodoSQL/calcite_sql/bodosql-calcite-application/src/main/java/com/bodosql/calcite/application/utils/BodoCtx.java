package com.bodosql.calcite.application.utils;

import java.util.HashSet;

public class BodoCtx {

  // columns used inside CASE statement
  private HashSet<String> dynamicParams;

  public BodoCtx() {
    this.dynamicParams = new HashSet<>();
  }

  public HashSet<String> getDynamicParams() {
    return this.dynamicParams;
  }
}
