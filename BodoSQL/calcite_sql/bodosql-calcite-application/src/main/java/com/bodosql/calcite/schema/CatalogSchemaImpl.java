package com.bodosql.calcite.schema;

import com.bodosql.calcite.catalog.BodoSQLCatalog;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Table;

public class CatalogSchemaImpl extends BodoSqlSchema {
  /**
   * Definition of a schema that is used with a corresponding remote catalog. All APIs are defined
   * to load the necessary information directly from the catalog, in some cases caching the
   * information.
   *
   * <p>See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Schema
   */
  private final BodoSQLCatalog catalog;

  // Set used to cache the table names fetched from catalog
  private Set<String> tableNames = null;

  // Hashmap used to cache tables fetched from the catalog
  private final HashMap<String, Table> tableMap;

  /**
   * Primary constructor for a CatalogSchemaImpl. This stores the relevant attributes and
   * initializes the table cache.
   *
   * @param name Name of the schema.
   * @param catalog Catalog used for loading information.
   */
  public CatalogSchemaImpl(String name, BodoSQLCatalog catalog) {
    super(name);
    this.catalog = catalog;
    this.tableMap = new HashMap<>();
  }

  /**
   * Get the name of all tables stored in this schema. If the names have already been fetched then
   * they are cached, otherwise they are cached.
   *
   * @return Set of all available table names.
   */
  @Override
  public Set<String> getTableNames() {
    if (tableNames == null) {
      tableNames = this.catalog.getTableNames(this.getName());
    }
    return tableNames;
  }

  /**
   * Returns a table object for the given schema. If the table name has already been fetched it is
   * loaded from a cache. Otherwise it is fetched using the catalog.
   *
   * @param name Name of the table to load.
   * @return Table object.
   */
  @Override
  public Table getTable(String name) {
    if (this.tableMap.containsKey(name)) {
      return this.tableMap.get(name);
    }
    Table table = this.catalog.getTable(this, name);
    if (table == null) {
      throw new RuntimeException(
          String.format("Table %s not found in Schema %s.", name, this.getName()));
    } else {
      this.tableMap.put(name, table);
    }
    return table;
  }

  /**
   * Returns the subschema from the catalog corresponding to the given schema name. This is
   * unimplemented as this time.
   *
   * @param schemaName Name of the subschema.
   * @return A Subschema if it exists.
   */
  @Override
  public Schema getSubSchema(String schemaName) {
    return null;
  }

  /**
   * Returns a set of the names of all subschemas in the catalog. This is unimplemented as this
   * time.
   *
   * @return A list of any SubSchema names if they exist.
   */
  @Override
  public Set<String> getSubSchemaNames() {
    return new HashSet<>();
  }

  /**
   * API specific to CatalogSchemaImpl and not all schemas. Since schemas with the same catalog
   * often share the same code generation process, the write code for a given table with a catalog
   * is controlled by that catalog.
   *
   * @param varName The name of the variable being written.
   * @param tableName The name of the table as the write destination.
   * @return The generated code to compute the write in Python.
   */
  public String generateWriteCode(String varName, String tableName) {
    return this.catalog.generateWriteCode(varName, this.getName(), tableName);
  }

  /**
   * API specific to CatalogSchemaImpl and not all schemas. Since schemas with the same catalog
   * often share the same code generation process, the read code for a given table with a catalog is
   * controlled by that catalog.
   *
   * @param tableName The name of the table as the read source.
   * @return The generated code to compute the read in Python.
   */
  public String generateReadCode(String tableName) {
    return this.catalog.generateReadCode(this.getName(), tableName);
  }
}
