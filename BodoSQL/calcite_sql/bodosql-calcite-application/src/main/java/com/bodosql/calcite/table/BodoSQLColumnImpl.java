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
  private final String readName;

  /** The name of the column to use for writing. */
  private final String writeName;

  /** Is this column type nullable? */
  private final boolean nullable;

  /**
   * Create a new column from a name and a type.
   *
   * @param name the name that we will give the column
   * @param type the {@link BodoSQLColumnDataType} which maps to a Bodo type in Python
   * @param nullable Is the column type nullable?
   */
  public BodoSQLColumnImpl(String name, BodoSQLColumnDataType type, boolean nullable) {
    this.dataType = type;
    this.readName = name;
    this.writeName = name;
    this.elemType = BodoSQLColumnDataType.EMPTY;
    this.nullable = nullable;
  }

  /**
   * Create a new column from a name and a type.
   *
   * @param readName the name that we will give the column when reading
   * @param writeName the name that we will give the column when writing
   * @param type the {@link BodoSQLColumnDataType} which maps to a Bodo type in Python
   * @param nullable Is the column type nullable?
   */
  public BodoSQLColumnImpl(
      String readName, String writeName, BodoSQLColumnDataType type, boolean nullable) {
    this.dataType = type;
    this.readName = readName;
    this.writeName = writeName;
    this.elemType = BodoSQLColumnDataType.EMPTY;
    this.nullable = nullable;
  }

  /**
   * Create a new column from a name and a type.
   *
   * @param name the name that we will give the column
   * @param type the {@link BodoSQLColumnDataType} which maps to a Bodo type in Python
   * @param elemType the {@link BodoSQLColumnDataType} for the element in categorical types
   * @param nullable Is the column type nullable?
   */
  public BodoSQLColumnImpl(
      String name, BodoSQLColumnDataType type, BodoSQLColumnDataType elemType, boolean nullable) {
    this.dataType = type;
    this.readName = name;
    this.writeName = name;
    this.elemType = elemType;
    this.nullable = nullable;
  }

  /**
   * Create a new column from a name and a type.
   *
   * @param readName the name that we will give the column when reading
   * @param writeName the name that we will give the column when writing
   * @param type the {@link BodoSQLColumnDataType} which maps to a Bodo type in Python
   * @param elemType the {@link BodoSQLColumnDataType} for the element in categorical types
   * @param nullable Is the column type nullable?
   */
  public BodoSQLColumnImpl(
      String readName,
      String writeName,
      BodoSQLColumnDataType type,
      BodoSQLColumnDataType elemType,
      boolean nullable) {
    this.dataType = type;
    this.readName = readName;
    this.writeName = writeName;
    this.elemType = elemType;
    this.nullable = nullable;
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
  public BodoSQLColumnDataType getColumnDataType() {
    return this.dataType;
  }

  @Override
  public RelDataType convertToSqlType(RelDataTypeFactory typeFactory, boolean nullable) {
    BodoSQLColumnDataType dtype = this.dataType;
    if (this.dataType == BodoSQLColumnDataType.CATEGORICAL) {
      // Categorical code should be treated as its underlying elemType
      dtype = this.elemType;
    }
    return dtype.convertToSqlType(typeFactory, nullable);
  }

  @Override
  public boolean requiresReadCast() {
    return this.dataType.requiresReadCast();
  }

  @Override
  public boolean requiresWriteCast() {
    return this.dataType.requiresWriteCast();
  }

  @Override
  public boolean isNullable() {
    return nullable;
  }

  /**
   * Generate the expression to cast this column to its BodoSQL type with a read.
   *
   * @param varName Name of the table to use.
   * @return The string passed to __bodosql_replace_columns_dummy to cast this column to its
   *     original data type with a read.
   */
  @Override
  public String getReadCastExpr(String varName) {
    String dtype;
    if (this.dataType == BodoSQLColumnDataType.CATEGORICAL) {
      // Categorical data should be cast to the elem type. This cannot
      // be described in a single BodoSQLColumnDataType yet.
      dtype = this.elemType.getCastType().getTypeString();
    } else {
      dtype = this.dataType.getCastType().getTypeString();
    }
    return getCommonCastExpr(varName, String.format("'%s'", dtype));
  }

  /**
   * Generate the expression to cast this column to its BodoSQL type with a write.
   *
   * @param varName Name of the table to use.
   * @return The string passed to __bodosql_replace_columns_dummy to cast this column to its
   *     original data type with a write.
   */
  @Override
  public String getWriteCastExpr(String varName) {
    String dtype;
    if (this.dataType == BodoSQLColumnDataType.DATE) {
      dtype = "bodo.datetime_date_type";
    } else {
      dtype = String.format("'%s'", this.elemType.getTypeString());
    }
    return getCommonCastExpr(varName, dtype);
  }

  private String getCommonCastExpr(String varName, String castValue) {
    return String.format("%s['%s'].astype(%s, copy=False)", varName, this.readName, castValue);
  }
}
