package com.bodosql.calcite.application;

public class BodoSQLExprTypeDeterminationException extends RuntimeException {
  /** Exception Class for Runtime Exception produced during expression type checking. */
  public BodoSQLExprTypeDeterminationException(String errorMessage) {
    super(errorMessage);
  }
}
