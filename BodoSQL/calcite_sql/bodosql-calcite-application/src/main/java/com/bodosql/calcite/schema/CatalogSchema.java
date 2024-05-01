package com.bodosql.calcite.schema;

import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.application.write.WriteTarget;
import com.bodosql.calcite.application.write.WriteTarget.IfExistsBehavior;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.catalog.IcebergCatalog;
import com.bodosql.calcite.catalog.SnowflakeCatalog;
import com.bodosql.calcite.ddl.DDLExecutor;
import com.bodosql.calcite.ddl.IcebergDDLExecutor;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.sql.ddl.SnowflakeCreateTableMetadata;
import com.bodosql.calcite.table.CatalogTable;
import com.google.common.collect.ImmutableList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import org.apache.calcite.schema.Function;
import org.apache.calcite.sql.ddl.SqlCreateTable;

public class CatalogSchema extends BodoSqlSchema {
  /**
   * Definition of a schema that is used with a corresponding remote catalog. All APIs are defined
   * to load the necessary information directly from the catalog, in some cases caching the
   * information.
   *
   * <p>See the design described on Confluence: <a
   * href="https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Schema">...</a>
   */
  private final BodoSQLCatalog catalog;

  // Set used to cache the table names fetched from the catalog
  private Set<String> tableNames = null;
  // Set used to cache the subSchema names fetched from the catalog
  private Set<String> subSchemaNames = null;
  // Set used to cache the function names fetched from the catalog
  private Set<String> functionNames = null;

  // Hashmap used to cache tables fetched from the catalog
  private final HashMap<String, CatalogTable> tableMap;
  // Hashmap used to cache schemas fetched from the catalog
  private final HashMap<String, CatalogSchema> subSchemaMap;
  // Hashmap used to cache functions fetched from the catalog
  private final HashMap<String, Collection<Function>> functionMap;

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
    this.tableMap = new HashMap();
    this.subSchemaMap = new HashMap();
    this.functionMap = new HashMap();
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
    if (!catalog.schemaDepthMayContainTables(getSchemaDepth())) {
      return Set.of();
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
    if (!catalog.schemaDepthMayContainTables(getSchemaDepth())) {
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
      return Set.of();
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
   * Returns all functions defined in this schema with a given name.
   *
   * @param funcName Name of functions with a given name.
   * @return Collection of all functions with that name.
   */
  @Override
  public Collection<Function> getFunctions(String funcName) {
    if (!catalog.schemaDepthMayContainFunctions(getSchemaDepth())) {
      return List.of();
    }
    if (this.functionMap.containsKey(funcName)) {
      return this.functionMap.get(funcName);
    }
    Collection<Function> functions = this.catalog.getFunctions(getFullPath(), funcName);
    if (functions == null) {
      throw new RuntimeException(
          String.format("Function %s not found in Schema %s.", funcName, this.getName()));
    } else {
      this.functionMap.put(funcName, functions);
    }
    return functions;
  }

  /**
   * Returns the name of all functions defined in this schema. This is likely used for a stored
   * procedure syntax but is not implemented for BodoSQL.
   *
   * @return Set of all function names in this schema.
   */
  @Override
  public Set<String> getFunctionNames() {
    if (!catalog.schemaDepthMayContainFunctions(getSchemaDepth())) {
      return Set.of();
    }
    if (functionNames == null) {
      functionNames = this.catalog.getFunctionNames(this.getFullPath());
    }
    return functionNames;
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
      PandasCodeGenVisitor visitor,
      Variable varName,
      String tableName,
      IfExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType,
      SnowflakeCreateTableMetadata meta) {
    return this.catalog.generateWriteCode(
        visitor, varName, createTablePath(tableName), ifExists, createTableType, meta);
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

  /**
   * Return the desired WriteTarget for a create table operation based on the provided catalog. Each
   * catalog may opt of a different write target based on the creation details and the desired
   * table. For example, if we are replacing a table a catalog could opt to keep the table type the
   * same or a catalog could have restrictions based on the type of create attempted.
   *
   * @param tableName The name of the type that will be created.
   * @param createTableType The createTable type.
   * @param ifExistsBehavior The createTable behavior for if there is already a table defined.
   * @param columnNamesGlobal Global Variable holding the output column names.
   * @return The selected write target.
   */
  public WriteTarget getCreateTableWriteTarget(
      String tableName,
      SqlCreateTable.CreateTableType createTableType,
      IfExistsBehavior ifExistsBehavior,
      Variable columnNamesGlobal) {
    return catalog.getCreateTableWriteTarget(
        getFullPath(), tableName, createTableType, ifExistsBehavior, columnNamesGlobal);
  }

  /** Get the DDL Executor for the schema. This is used to execute DDL commands on the table. */
  public DDLExecutor getDDLExecutor() {
    if (catalog instanceof SnowflakeCatalog) {
      return ((SnowflakeCatalog) catalog).getDdlExecutor();
    } else if (catalog instanceof IcebergCatalog) {
      return new IcebergDDLExecutor(((IcebergCatalog) catalog).getIcebergConnection());
    } else {
      throw new UnsupportedOperationException("DDL operations are not supported for this schema");
    }
  }
}
