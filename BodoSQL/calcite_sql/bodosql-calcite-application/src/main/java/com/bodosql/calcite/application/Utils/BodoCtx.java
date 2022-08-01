package com.bodosql.calcite.application.Utils;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class BodoCtx {
  private List<String> colsToAddList;
  private HashSet<String> nullSet;
  private HashSet<String> namedParams;

  public BodoCtx(List<String> colsToAddList, HashSet<String> nullSet, HashSet<String> namedParams) {
    this.colsToAddList = colsToAddList;
    this.nullSet = nullSet;
    this.namedParams = namedParams;
  }

  public BodoCtx() {
    this.colsToAddList = new ArrayList<>();
    this.nullSet = new HashSet<>();
    this.namedParams = new HashSet<>();
  }

  public List<String> getColsToAddList() {
    return this.colsToAddList;
  }

  public HashSet<String> getNullSet() {
    return this.nullSet;
  }

  public HashSet<String> getNamedParams() {
    return this.namedParams;
  }

  public void setColsToAddList(List<String> colsToAddList) {
    this.colsToAddList = colsToAddList;
  }

  public void setNullSet(HashSet<String> nullSet) {
    this.nullSet = nullSet;
  }

  public void setNamedParams(HashSet<String> namedParams) {
    this.namedParams = namedParams;
  }
}
