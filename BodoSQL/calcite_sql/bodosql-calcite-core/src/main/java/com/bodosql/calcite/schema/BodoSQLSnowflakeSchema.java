package com.bodosql.calcite.schema;

import com.bodosql.calcite.catalog.connection.SnowflakeCatalog;
import java.sql.SQLException;
import java.util.Set;
import org.apache.calcite.schema.Table;

public class BodoSQLSnowflakeSchema extends BodoSqlSchema {
  private final SnowflakeCatalog catalog;
  private final String name;

  public BodoSQLSnowflakeSchema(String name, SnowflakeCatalog catalog) {
    // TODO: Determine what to do for CatalogDatabase
    super(catalog.getDb());
    this.catalog = catalog;
    this.name = name;
  }

  @Override
  public String getName() {
    return this.name;
  }

  @Override
  public Set<String> getTableNames() {
    LOGGER.debug("getting table names");
    try {
      return this.catalog.getTableNames(this.getName());
    } catch (SQLException e) {
      throw new RuntimeException(
          String.format(
              "Unable to get table name for Schema '%s' from Snowflake account. Error message: %s",
              this.getName(), e));
    }
  }

  @Override
  public Table getTable(String name) {
    try {
      return this.catalog.getTable(this.getName(), name);
    } catch (SQLException e) {
      throw new RuntimeException(
          String.format(
              "Unable to get table '%s' for Schema '%s' from Snowflake account. Error message: %s",
              name, this.getName(), e));
    }
  }
}
