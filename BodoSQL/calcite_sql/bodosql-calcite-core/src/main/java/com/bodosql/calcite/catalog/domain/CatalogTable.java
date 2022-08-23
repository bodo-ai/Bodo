package com.bodosql.calcite.catalog.domain;

import java.util.Set;

public interface CatalogTable {
  public String getTableName();

  public Set<BodoSQLColumn> getColumns();

  public CatalogDatabase getDatabase();
}
