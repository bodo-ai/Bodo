package com.bodosql.calcite.application;

public class BodoSQLCodegenException extends RuntimeException {
  /** Exception Class for Runtime Exception produced during Code Generation. */
  public BodoSQLCodegenException(String errorMessage) {
    super(errorMessage);
  }
}
