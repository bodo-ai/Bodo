package com.bodosql.calcite.table;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Variable;
import javax.annotation.Nullable;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.type.BodoTZInfo;

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

  /** What is the precision for this type? Currently only used by Time. */
  private final int precision;

  /** What is the timezone info for this column if it is a tz-aware timestamp. */
  private final @Nullable BodoTZInfo tzInfo;

  /**
   * Create a new column from a name, type, and nullability.
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
    this.tzInfo = null;
    this.precision = RelDataType.PRECISION_NOT_SPECIFIED;
  }

  /**
   * Create a new column from a name, type, nullability and tzInfo.
   *
   * @param name the name that we will give the column
   * @param type the {@link BodoSQLColumnDataType} which maps to a Bodo type in Python
   * @param nullable Is the column type nullable?
   * @param tzInfo The timezone to use for this column if its timezone aware. This may be null.
   */
  public BodoSQLColumnImpl(
      String name, BodoSQLColumnDataType type, boolean nullable, BodoTZInfo tzInfo) {
    this.dataType = type;
    this.readName = name;
    this.writeName = name;
    this.elemType = BodoSQLColumnDataType.EMPTY;
    this.nullable = nullable;
    this.tzInfo = tzInfo;
    this.precision = RelDataType.PRECISION_NOT_SPECIFIED;
  }

  /**
   * Create a new column from a name, type, nullability and tzInfo.
   *
   * @param name the name that we will give the column
   * @param type the {@link BodoSQLColumnDataType} which maps to a Bodo type in Python
   * @param nullable Is the column type nullable?
   * @param precision The precision to use when creating the type. Currently only used for Time
   *     types.
   */
  public BodoSQLColumnImpl(
      String name, BodoSQLColumnDataType type, boolean nullable, int precision) {
    this.dataType = type;
    this.readName = name;
    this.writeName = name;
    this.elemType = BodoSQLColumnDataType.EMPTY;
    this.nullable = nullable;
    this.tzInfo = null;
    this.precision = precision;
  }

  /**
   * Create a new column from a read name, write name, type, and nullability.
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
    this.tzInfo = null;
    this.precision = RelDataType.PRECISION_NOT_SPECIFIED;
  }

  /**
   * Create a new column from a read name, write name, type, nullability and tzInfo.
   *
   * @param readName the name that we will give the column when reading
   * @param writeName the name that we will give the column when writing
   * @param type the {@link BodoSQLColumnDataType} which maps to a Bodo type in Python
   * @param nullable Is the column type nullable?
   * @param tzInfo The timezone to use for this column if its timezone aware. This may be null.
   * @param precision The precision to use when creating the type. Currently only used for Time
   *     types.
   */
  public BodoSQLColumnImpl(
      String readName,
      String writeName,
      BodoSQLColumnDataType type,
      boolean nullable,
      BodoTZInfo tzInfo,
      int precision) {
    this.dataType = type;
    this.readName = readName;
    this.writeName = writeName;
    this.elemType = BodoSQLColumnDataType.EMPTY;
    this.nullable = nullable;
    this.tzInfo = tzInfo;
    this.precision = precision;
  }

  /**
   * Create a new column from a name, type, elemType, and nullability. This is used for categorical
   * data.
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
    this.tzInfo = null;
    this.precision = RelDataType.PRECISION_NOT_SPECIFIED;
  }

  /**
   * Create a new column from a name, type, elemType, nullability and tzInfo. This is used for
   * categorical data.
   *
   * @param name the name that we will give the column
   * @param type the {@link BodoSQLColumnDataType} which maps to a Bodo type in Python
   * @param elemType the {@link BodoSQLColumnDataType} for the element in categorical types
   * @param nullable Is the column type nullable?
   * @param tzInfo The timezone to use for this column if its timezone aware. This may be null.
   */
  public BodoSQLColumnImpl(
      String name,
      BodoSQLColumnDataType type,
      BodoSQLColumnDataType elemType,
      boolean nullable,
      BodoTZInfo tzInfo) {
    this.dataType = type;
    this.readName = name;
    this.writeName = name;
    this.elemType = elemType;
    this.nullable = nullable;
    this.tzInfo = tzInfo;
    this.precision = RelDataType.PRECISION_NOT_SPECIFIED;
  }

  /**
   * Create a new column from a name, type, elemType, nullability and tzInfo. This is used for
   * categorical data.
   *
   * @param name the name that we will give the column
   * @param type the {@link BodoSQLColumnDataType} which maps to a Bodo type in Python
   * @param elemType the {@link BodoSQLColumnDataType} for the element in categorical types
   * @param nullable Is the column type nullable?
   * @param tzInfo The timezone to use for this column if its timezone aware. This may be null.
   * @param precision The precision to use when creating the type. Currently only used for Time
   *     types.
   */
  public BodoSQLColumnImpl(
      String name,
      BodoSQLColumnDataType type,
      BodoSQLColumnDataType elemType,
      boolean nullable,
      BodoTZInfo tzInfo,
      int precision) {
    this.dataType = type;
    this.readName = name;
    this.writeName = name;
    this.elemType = elemType;
    this.nullable = nullable;
    this.tzInfo = tzInfo;
    this.precision = precision;
  }

  /**
   * Create a new column from a readName, writeName type, elemType, and nullability. This is used
   * for categorical data.
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
    this.tzInfo = null;
    this.precision = RelDataType.PRECISION_NOT_SPECIFIED;
  }

  /**
   * Create a new column from a readName, writeName type, elemType, nullability and tzInfo. This is
   * used for categorical data.
   *
   * @param readName the name that we will give the column when reading
   * @param writeName the name that we will give the column when writing
   * @param type the {@link BodoSQLColumnDataType} which maps to a Bodo type in Python
   * @param elemType the {@link BodoSQLColumnDataType} for the element in categorical types
   * @param nullable Is the column type nullable?
   * @param tzInfo The timezone to use for this column if its timezone aware. This may be null.
   */
  public BodoSQLColumnImpl(
      String readName,
      String writeName,
      BodoSQLColumnDataType type,
      BodoSQLColumnDataType elemType,
      boolean nullable,
      BodoTZInfo tzInfo) {
    this.dataType = type;
    this.readName = readName;
    this.writeName = writeName;
    this.elemType = elemType;
    this.nullable = nullable;
    this.tzInfo = tzInfo;
    this.precision = RelDataType.PRECISION_NOT_SPECIFIED;
  }

  /**
   * Create a new column from a readName, writeName type, elemType, nullability, tzInfo and
   * precision.
   *
   * @param readName the name that we will give the column when reading
   * @param writeName the name that we will give the column when writing
   * @param type the {@link BodoSQLColumnDataType} which maps to a Bodo type in Python
   * @param elemType the {@link BodoSQLColumnDataType} for the element in categorical types
   * @param nullable Is the column type nullable?
   * @param tzInfo The timezone to use for this column if its timezone aware. This may be null.
   */
  public BodoSQLColumnImpl(
      String readName,
      String writeName,
      BodoSQLColumnDataType type,
      BodoSQLColumnDataType elemType,
      boolean nullable,
      BodoTZInfo tzInfo,
      int precision) {
    this.dataType = type;
    this.readName = readName;
    this.writeName = writeName;
    this.elemType = elemType;
    this.nullable = nullable;
    this.tzInfo = tzInfo;
    this.precision = precision;
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
  public RelDataType convertToSqlType(
      RelDataTypeFactory typeFactory, boolean nullable, BodoTZInfo tzInfo, int precision) {
    BodoSQLColumnDataType dtype = this.dataType;
    if (this.dataType == BodoSQLColumnDataType.CATEGORICAL) {
      // Categorical code should be treated as its underlying elemType
      dtype = this.elemType;
    }
    if (this.dataType == BodoSQLColumnDataType.ARRAY
        && !(this.elemType == BodoSQLColumnDataType.VARIANT)) {
      if (this.elemType == BodoSQLColumnDataType.EMPTY) {
        throw new BodoSQLCodegenException("Cannot have ARRAY type with dtype EMPTY");
      }
      RelDataType elemRelType =
          this.elemType.convertToSqlType(typeFactory, nullable, tzInfo, precision);
      return typeFactory.createArrayType(elemRelType, -1);
    }
    return dtype.convertToSqlType(typeFactory, nullable, tzInfo, precision);
  }

  @Override
  public boolean requiresReadCast() {
    return this.dataType.requiresReadCast();
  }

  @Override
  public boolean isNullable() {
    return nullable;
  }

  @Override
  public BodoTZInfo getTZInfo() {
    return tzInfo;
  }

  @Override
  public int getPrecision() {
    return precision;
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
    String dtype = this.elemType.getTypeString();
    // Categorical data should be cast to the elem type. This cannot
    // be described in a single BodoSQLColumnDataType yet.
    return getCommonCastExpr(varName, String.format("'%s'", dtype));
  }

  private String getCommonCastExpr(Variable varName, String castValue) {
    return String.format(
        "%s['%s'].astype(%s, copy=False)", varName.emit(), this.readName, castValue);
  }
}
