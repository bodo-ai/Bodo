package com.bodosql.calcite.table;

import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.CatalogSchemaImpl;
import java.util.List;

/**
 *
 *
 * <h1>Stores a table with its corresponding columns</h1>
 *
 * @author bodo
 */
public class CatalogTableImpl extends BodoSqlTable {
  /**
   * See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Table
   */

  /**
   * This constructor is used to fill in all the values for the table.
   *
   * @param name the name of the table that is being created
   * @param schema the BodoSQL schema to which this table belongs
   * @param columns list of columns to be added to the table.
   */
  public CatalogTableImpl(String name, BodoSqlSchema schema, List<BodoSQLColumn> columns) {
    super(name, schema, columns);
    if (!(schema instanceof CatalogSchemaImpl)) {
      throw new RuntimeException("Catalog table must be initialized with a Catalog Schema.");
    }
  }

  /**
   * Returns the schema for the table but cast to the required CatalogSchemaImpl. This is done when
   * actions are catalog specific, such as the read/write code.
   *
   * @return The stored schema cast to a CatalogSchemaImpl.
   */
  private CatalogSchemaImpl getCatalogSchema() {
    // This cast is safe because it's enforced on the constructor.
    return (CatalogSchemaImpl) this.getSchema();
  }

  /**
   * Can BodoSQL write to this table. By default this is true but in the future this may be extended
   * to look at the permissions given in the catalog.
   *
   * @return Can BodoSQL write to this table.
   */
  @Override
  public boolean isWriteable() {
    // TODO: Update with the ability to check permissions from the schema/catalog
    return true;
  }

  /**
   * Generate the code needed to write the given variable to storage. This table type generates code
   * common to all tables in the catalog.
   *
   * @param varName Name of the variable to write.
   * @return The generated code to write the table.
   */
  @Override
  public String generateWriteCode(String varName) {
    return this.getCatalogSchema().generateWriteCode(varName, this.getName());
  }

  /**
   * Generate the code needed to read the table. This table type generates code common to all tables
   * in the catalog.
   *
   * @return The generated code to read the table.
   */
  @Override
  public String generateReadCode() {
    return this.getCatalogSchema().generateReadCode(this.getName());
  }
}
