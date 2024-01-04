package com.bodosql.calcite.application.utils;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class BodoCtx {
  private List<String> colsToAddList;

  // columns used inside CASE statement
  private HashSet<Integer> usedColumns;
  private HashSet<String> namedParams;

  public BodoCtx(
      List<String> colsToAddList, HashSet<Integer> usedColumns, HashSet<String> namedParams) {
    this.colsToAddList = colsToAddList;
    this.usedColumns = usedColumns;
    this.namedParams = namedParams;
  }

  public BodoCtx() {
    this.colsToAddList = new ArrayList<>();
    this.usedColumns = new HashSet<>();
    this.namedParams = new HashSet<>();
  }

  public HashSet<String> getNamedParams() {
    return this.namedParams;
  }
}
