package com.bodosql.calcite.catalog.domain;

public interface CatalogColumn {
	public String
	getColumnName();

	public CatalogColumnDataType
	getColumnDataType();

	public CatalogTable
	getTable();
}
