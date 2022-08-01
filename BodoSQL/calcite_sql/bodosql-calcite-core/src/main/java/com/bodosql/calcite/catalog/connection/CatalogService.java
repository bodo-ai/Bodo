package com.bodosql.calcite.catalog.connection;

import com.bodosql.calcite.catalog.domain.CatalogDatabase;
import com.bodosql.calcite.catalog.domain.CatalogSchema;
import com.bodosql.calcite.catalog.domain.CatalogTable;

import java.util.Collection;

public interface CatalogService {
	public CatalogSchema
	getSchema(String schemaName);

	// for calcite schema get subschemas
	public CatalogDatabase
	getDatabase(String schemaName, String databaseName);

	public Collection<CatalogTable>
	getTables(String databaseName);

	public CatalogTable
	getTable(String schemaName, String tableName);

	// TODO we may not need this api
	public CatalogTable
	getTable(String tableName);
}
