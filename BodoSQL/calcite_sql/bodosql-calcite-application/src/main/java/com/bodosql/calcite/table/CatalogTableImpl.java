package com.bodosql.calcite.table;

import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.CatalogSchemaImpl;
import java.util.*;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.type.*;

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
   * Generate the code needed to write the given variable to storage.
   *
   * @param varName Name of the variable to write.
   * @param extraArgs Extra arguments to pass to the Python API. They are assume to be escaped by
   *     the calling function and are of the form "key1=value1, ..., keyN=valueN".
   * @return The generated code to write the table.
   */
  public String generateWriteCode(String varName, String extraArgs) {
    throw new UnsupportedOperationException("Catalog APIs do not support additional arguments");
  }

  /**
   * Return the location from which the table is generated. The return value is always entirely
   * capitalized.
   *
   * @return The source DB location.
   */
  @Override
  public String getDBType() {
    return this.getCatalogSchema().getDBType().toUpperCase();
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

  /**
   * Generate the code needed to read the table. This function is called by specialized IO
   * implementations that require passing 1 or more additional arguments.
   *
   * @param extraArgs: Extra arguments to pass to the Python API. They are assume to be escaped by
   *     the calling function and are of the form "key1=value1, ..., keyN=valueN".
   * @return The generated code to read the table.
   */
  @Override
  public String generateReadCode(String extraArgs) {
    throw new UnsupportedOperationException("Catalog APIs do not support additional arguments");
  }

  @Override
  public String generateReadCastCode(String varName) {
    // Snowflake catalog uses _bodo_read_date_as_dt64=True to convert date columns to datetime64
    // without astype() calls in the IR which cause issues for limit pushdown.
    // see BE-4238
    return "";
  }

  /**
   * Generates the code necessary to submit the remote query to the catalog DB. This is not
   * supported for local tables.
   *
   * @param query Query to submit.
   * @return The generated code.
   */
  @Override
  public String generateRemoteQuery(String query) {
    return this.getCatalogSchema().generateRemoteQuery(query);
  }

  @Override
  public Table extend(List<RelDataTypeField> extensionFields) {
    String name = this.getName();
    BodoSqlSchema schema = this.getSchema();
    List<BodoSQLColumn> extendedColumns = new ArrayList<>();
    extendedColumns.addAll(this.columns);
    for (int i = 0; i < extensionFields.size(); i++) {
      RelDataTypeField curField = extensionFields.get(0);
      String fieldName = curField.getName();
      RelDataType colType = curField.getType();
      BodoSQLColumn.BodoSQLColumnDataType newColType =
          BodoSQLColumn.BodoSQLColumnDataType.fromSqlType(colType);
      // getTZInfo() returns null if the type is not TZAware Timestamp
      BodoTZInfo tzInfo = colType.getTZInfo();
      BodoSQLColumn newCol = new BodoSQLColumnImpl(fieldName, newColType, false, tzInfo);
      extendedColumns.add(newCol);
    }
    return new CatalogTableImpl(name, schema, extendedColumns);
  }

  /**
   * Returns if calling `generateReadCode()` for a table will result in an IO operation in the Bodo
   * generated code.
   *
   * @return Does the table require IO?
   */
  @Override
  public boolean readRequiresIO() {
    return true;
  }
}
