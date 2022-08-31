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

  /** Type used for the child with parameterizable types. For example, Categorical types. */
  private final BodoSQLColumnDataType elemType;

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
    this.elemType = BodoSQLColumnDataType.EMPTY;
  }

  public BodoSQLColumnImpl(
      String name, BodoSQLColumnDataType type, BodoSQLColumnDataType elemType) {
    this.dataType = type;
    this.name = name;
    this.elemType = elemType;
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
    BodoSQLColumnDataType dtype = this.dataType;
    if (this.dataType == BodoSQLColumnDataType.CATEGORICAL) {
      // Categorical code should be treated as its underlying elemType
      dtype = this.elemType;
    }
    return dtype.convertToSqlType(typeFactory);
  }

  @Override
  public boolean requiresCast() {
    return this.dataType.requiresCast();
  }

  /**
   * @return The string passed to __bodosql_replace_columns_dummy to cast this column to its BodoSQL
   *     supported type.
   */
  @Override
  public String getCastString(String varName) {
    String dtype;
    if (this.dataType == BodoSQLColumnDataType.CATEGORICAL) {
      // Categorical data should be cast to the elem type. This cannot
      // be described in a single BodoSQLColumnDataType yet.
      dtype = this.elemType.getCastType().getTypeString();
    } else {
      dtype = this.dataType.getCastType().getTypeString();
    }
    return String.format("%s['%s'].astype('%s', copy=False)", varName, this.name, dtype);
  }
}
