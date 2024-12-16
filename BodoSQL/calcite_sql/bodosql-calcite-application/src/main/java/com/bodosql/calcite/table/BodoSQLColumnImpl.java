package com.bodosql.calcite.table;

import com.bodosql.calcite.ir.Variable;

/**
 *
 *
 * <h1>Representaion of a column in a table</h1>
 *
 * A {@link CatalogTable} contains several columns. The point of this class is to be able to store
 * names and types.
 *
 * <p>For more information, see the design described on Confluence:
 * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Column
 *
 * @author bodo
 */
public class BodoSQLColumnImpl implements BodoSQLColumn {

  /** The name of the column. */
  private final String readName;

  /** The name of the column to use for writing. */
  private final String writeName;

  /** Type information for the column. */
  private final ColumnDataTypeInfo dataTypeInfo;

  /**
   * Create a new column from a name and data type information.
   *
   * @param name the name that we will give the column
   * @param dataTypeInfo The data type information.
   */
  public BodoSQLColumnImpl(String name, ColumnDataTypeInfo dataTypeInfo) {
    this.readName = name;
    this.writeName = name;
    this.dataTypeInfo = dataTypeInfo;
  }

  /**
   * Create a new column from a read name, write name and data type information.
   *
   * @param readName the name that we will give the column when reading
   * @param writeName the name that we will give the column when writing
   * @param dataTypeInfo The data type information.
   */
  public BodoSQLColumnImpl(String readName, String writeName, ColumnDataTypeInfo dataTypeInfo) {
    this.readName = readName;
    this.writeName = writeName;
    this.dataTypeInfo = dataTypeInfo;
  }

  @Override
  public String getColumnName() {
    return this.readName;
  }

  @Override
  public String getWriteColumnName() {
    return this.writeName;
  }

  @Override
  public ColumnDataTypeInfo getDataTypeInfo() {
    return dataTypeInfo;
  }

  @Override
  public boolean requiresReadCast() {
    // TODO: Fix for recursion. This is an API we want to remove anyways.
    return dataTypeInfo.getDataType().requiresReadCast();
  }

  /**
   * Generate the expression to cast this column to its BodoSQL type with a read.
   *
   * @param varName Name of the table to use.
   * @return The string passed to __bodosql_replace_columns_dummy to cast this column to its
   *     original data type with a read.
   */
  @Override
  public String getReadCastExpr(Variable varName) {
    String dtype = dataTypeInfo.getDataType().getTypeString();
    // Categorical data should be cast to the elem type. This cannot
    // be described in a single BodoSQLColumnDataType yet.
    return getCommonCastExpr(varName, String.format("'%s'", dtype));
  }

  private String getCommonCastExpr(Variable varName, String castValue) {
    return String.format(
        "%s['%s'].astype(%s, copy=False)", varName.emit(), this.readName, castValue);
  }
}
