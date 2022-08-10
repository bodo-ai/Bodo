package com.bodosql.calcite.application.Utils;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class BodoCtx {
  private List<String> colsToAddList;
  // column references inside CASE that need explicit NULL handling.
  private HashSet<String> needNullCheckColumns;

  // columns used inside CASE statement
  private HashSet<Integer> usedColumns;
  private HashSet<String> namedParams;

  public BodoCtx(
      List<String> colsToAddList,
      HashSet<String> needNullCheckColumns,
      HashSet<Integer> usedColumns,
      HashSet<String> namedParams) {
    this.colsToAddList = colsToAddList;
    this.needNullCheckColumns = needNullCheckColumns;
    this.usedColumns = usedColumns;
    this.namedParams = namedParams;
  }

  public BodoCtx() {
    this.colsToAddList = new ArrayList<>();
    this.needNullCheckColumns = new HashSet<>();
    this.usedColumns = new HashSet<>();
    this.namedParams = new HashSet<>();
  }

  public List<String> getColsToAddList() {
    return this.colsToAddList;
  }

  public HashSet<String> getNeedNullCheckColumns() {
    return this.needNullCheckColumns;
  }

  public HashSet<String> getNamedParams() {
    return this.namedParams;
  }

  public void setNeedNullCheckColumns(HashSet<String> needNullCheckColumns) {
    this.needNullCheckColumns = needNullCheckColumns;
  }

  public HashSet<Integer> getUsedColumns() {
    return usedColumns;
  }
}
