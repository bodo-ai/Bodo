package com.bodosql.calcite.table;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;

/**
 *
 *
 * <h1>Representaion of a column in a table</h1>
 *
 * A {@link CatalogTableImpl} contains several columns. The point of this class is to be able to
 * store names and types.
 *
 * <p>For more information, see the design described on Confluence:
 * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Column
 *
 * @author bodo
 */
public class BodoSQLColumnImpl implements BodoSQLColumn {

  /** An enum type which maps to a Bodo Type */
  private final BodoSQLColumnDataType dataType;

  /** The name of the column. */
  private final String name;

  /**
   * Create a new column from a name and a type.
   *
   * @param name the name that we will give the column
   * @param type the {@link BodoSQLColumnDataType} which maps to a Bodo type in Python
   */
  public BodoSQLColumnImpl(String name, BodoSQLColumnDataType type) {
    this.dataType = type;
    this.name = name;
  }

  @Override
  public String getColumnName() {
    return this.name;
  }

  @Override
  public BodoSQLColumnDataType getColumnDataType() {
    return this.dataType;
  }

  @Override
  public RelDataType convertToSqlType(RelDataTypeFactory typeFactory) {
    return this.dataType.convertToSqlType(typeFactory);
  }

  @Override
  public boolean requiresCast() {
    return this.dataType.requiresCast();
  }
}
