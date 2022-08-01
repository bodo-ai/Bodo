package com.bodosql.calcite.catalog.domain;

import java.util.Set;

public interface CatalogSchema {
	public String
	getSchemaName();

	public Set<CatalogDatabase>
	getDatabases();

	public CatalogDatabase
	getDatabaseByName(String databaseName);
}
