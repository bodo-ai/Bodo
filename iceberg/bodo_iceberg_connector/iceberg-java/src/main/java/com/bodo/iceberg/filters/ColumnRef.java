package com.bodo.iceberg.filters;

public class ColumnRef extends Filter {
  public final String name;

  public ColumnRef(String name) {
    this.name = name;
  }
}
