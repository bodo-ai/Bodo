package com.bodo.iceberg.filters;

import java.util.ArrayList;

public class ArrayConst extends Filter {
  public final ArrayList<Object> value;

  public ArrayConst(ArrayList<Object> value) {
    this.value = value;
  }
}
