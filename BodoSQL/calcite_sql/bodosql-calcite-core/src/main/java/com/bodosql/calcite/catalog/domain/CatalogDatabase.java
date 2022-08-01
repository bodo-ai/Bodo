package com.bodosql.calcite.catalog.domain;

import java.util.Set;

public interface CatalogDatabase {
	public String
	getDatabaseName();

	public CatalogTable
	getTable(String tableName);

	public Set<String>
	getTableNames();
}
