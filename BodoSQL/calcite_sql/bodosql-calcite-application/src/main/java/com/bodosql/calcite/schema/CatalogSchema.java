package com.bodosql.calcite.schema;

import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.table.CatalogTable;
import com.google.common.collect.ImmutableList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;
import org.apache.calcite.sql.ddl.SqlCreateTable;

public class CatalogSchema extends BodoSqlSchema {
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
  // Set used to cache the subSchema names fetched from catalog
  private Set<String> subSchemaNames = null;

  // Hashmap used to cache tables fetched from the catalog
  private final HashMap<String, CatalogTable> tableMap;
  // Hashmap used to cache schemas fetched from the catalog
  private final HashMap<String, CatalogSchema> subSchemaMap;

  /**
   * Primary constructor for a CatalogSchema. This stores the relevant attributes and initializes
   * the table cache.
   *
   * @param name Name of the schema.
   * @param catalog Catalog used for loading information.
   */
  public CatalogSchema(
      String name, int depth, ImmutableList<String> schemaPath, BodoSQLCatalog catalog) {
    super(name, depth, schemaPath);
    this.catalog = catalog;
    this.tableMap = new HashMap<>();
    this.subSchemaMap = new HashMap<>();
  }

  /**
   * Get the name of all tables stored in this schema. If the names have already been fetched then
   * they are cached, otherwise they are cached. This returns an empty set if this schema depth
   * doesn't contain any tables.
   *
   * @return Set of all available table names.
   */
  @Override
  public Set<String> getTableNames() {
    if (!catalog.schemaDepthMayContainsTables(getSchemaDepth())) {
      return new HashSet<>();
    }
    if (tableNames == null) {
      tableNames = this.catalog.getTableNames(getFullPath());
    }
    return tableNames;
  }

  /**
   * Returns a table object for the given schema. If the table name has already been fetched it is
   * loaded from a cache. Otherwise, it is fetched using the catalog. This returns null if this
   * schema depth doesn't contain any tables.
   *
   * @param name Name of the table to load.
   * @return Table object.
   */
  @Override
  public CatalogTable getTable(String name) {
    if (!catalog.schemaDepthMayContainsTables(getSchemaDepth())) {
      return null;
    }
    if (this.tableMap.containsKey(name)) {
      return this.tableMap.get(name);
    }
    CatalogTable table = this.catalog.getTable(getFullPath(), name);
    if (table == null) {
      throw new RuntimeException(
          String.format("Table %s not found in Schema %s.", name, this.getName()));
    } else {
      this.tableMap.put(name, table);
    }
    return table;
  }

  /**
   * Returns the names of all possible subSchemas, caching the results. If the results are not found
   * in the schema, then it will traverse the catalog. If the given schema Depth doesn't support
   * subSchemas then it returns an empty Set.
   *
   * @return The Set of subSchema names.
   */
  @Override
  public Set<String> getSubSchemaNames() {
    if (!catalog.schemaDepthMayContainSubSchemas(getSchemaDepth())) {
      return new HashSet<>();
    }
    if (subSchemaNames == null) {
      subSchemaNames = this.catalog.getSchemaNames(this.getFullPath());
    }
    return subSchemaNames;
  }

  /**
   * Returns a subSchema with the given name, caching the results. If the results are not found in
   * the schema, then it will traverse the catalog. If the given schema Depth doesn't support
   * subSchemas then it returns an empty Set.
   *
   * @param schemaName Name of the subSchema.
   * @return The subSchema object.
   */
  @Override
  public CatalogSchema getSubSchema(String schemaName) {
    if (!catalog.schemaDepthMayContainSubSchemas(getSchemaDepth())) {
      return null;
    }
    if (this.subSchemaMap.containsKey(schemaName)) {
      return this.subSchemaMap.get(schemaName);
    }
    CatalogSchema schema = this.catalog.getSchema(getFullPath(), schemaName);
    if (schema == null) {
      throw new RuntimeException(
          String.format("Schema %s not found in Schema %s.", schemaName, this.getName()));
    } else {
      this.subSchemaMap.put(schemaName, schema);
    }
    return schema;
  }

  /**
   * API specific to CatalogSchema and not all schemas. Since schemas with the same catalog often
   * share the same code generation process, the write code for a given table with a catalog is
   * controlled by that catalog.
   *
   * @param varName The name of the variable being written.
   * @param tableName The name of the table as the write destination.
   * @param ifExists Behavior of the write if the table already exists
   * @param createTableType Behavior of the write if we're creating a new table. Defaults to DEFAULT
   * @return The generated code to compute the write in Python.
   */
  public Expr generateWriteCode(
      Variable varName,
      String tableName,
      BodoSQLCatalog.ifExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType) {
    return this.catalog.generateWriteCode(
        varName, createTablePath(tableName), ifExists, createTableType);
  }

  public Expr generateStreamingWriteInitCode(
      Expr.IntegerLiteral operatorID,
      String tableName,
      BodoSQLCatalog.ifExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType) {
    return this.catalog.generateStreamingWriteInitCode(
        operatorID, createTablePath(tableName), ifExists, createTableType);
  }

  public Expr generateStreamingWriteAppendCode(
      Variable stateVarName,
      Variable tableVarName,
      Variable colNamesGlobal,
      Variable isLastVarName,
      Variable iterVarName,
      Expr columnPrecision) {
    return this.catalog.generateStreamingWriteAppendCode(
        stateVarName, tableVarName, colNamesGlobal, isLastVarName, iterVarName, columnPrecision);
  }

  /** @return The catalog for the schema. */
  public BodoSQLCatalog getCatalog() {
    return catalog;
  }

  /**
   * Return the db location to which this schema refers.
   *
   * @return The source DB location.
   */
  public String getDBType() {
    return catalog.getDBType();
  }
}
