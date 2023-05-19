package com.bodosql.calcite.application.Utils;

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

  public List<String> getColsToAddList() {
    return this.colsToAddList;
  }

  public HashSet<Integer> getUsedColumns() {
    return usedColumns;
  }

  public HashSet<String> getNamedParams() {
    return this.namedParams;
  }

  public void unionContext(BodoCtx ctx) {
    this.colsToAddList.addAll(ctx.getColsToAddList());
    this.usedColumns.addAll(ctx.getUsedColumns());
    this.namedParams.addAll(ctx.getNamedParams());
  }
}
