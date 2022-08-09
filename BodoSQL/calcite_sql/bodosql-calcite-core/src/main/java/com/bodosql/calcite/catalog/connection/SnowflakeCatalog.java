package com.bodosql.calcite.catalog.connection;

import com.bodosql.calcite.catalog.domain.CatalogDatabase;
import java.sql.*;
import java.util.*;
import org.apache.calcite.schema.Table;

public interface SnowflakeCatalog {

  public CatalogDatabase getDb();

  public Set<String> getTableNames(String schemaName) throws SQLException;

  public Table getTable(String schemaName, String tableName) throws SQLException;
}
