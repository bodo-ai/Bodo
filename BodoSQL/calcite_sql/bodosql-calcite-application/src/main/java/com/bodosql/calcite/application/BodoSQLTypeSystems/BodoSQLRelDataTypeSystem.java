package com.bodosql.calcite.application.BodoSQLTypeSystems;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeSystemImpl;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.sql.type.SqlTypeName;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Class for the RelDataTypeSystem used by BodoSQL. */
public class BodoSQLRelDataTypeSystem extends RelDataTypeSystemImpl {
  private final BodoTZInfo defaultTZInfo;

  // This is found in SqlTypeName.
  // Added it here for simplicity.
  // TODO (HA): update with other default tickets.
  public static final int MAX_DATETIME_PRECISION = 9;

  /*
  WEEK_START parameter that determines which weekday a week starts with.
  We follow Snowflake behavior, mapping 0 and 1 to Monday (default), and
  2-7 for the rest of the days up to and including Sunday.

  Possible values: 0-7
   */
  private @Nullable Integer weekStart;

  /*
  WEEK_OF_YEAR_POLICY parameter that determines whether to follow ISO semantics
  or determine the first week of the year by looking at if the week of interest
  contains January 1st of that year.

  0 -> ISO semantics
  1 -> 1st week of year = week that contains January 1st

  Possible values: 0-1
   */
  private @Nullable Integer weekOfYearPolicy;

  /*
  CURRENT_DATABSE: Name of the database in use for the current session.

  */
  private String currentDatabase;

  public BodoSQLRelDataTypeSystem() {
    this(BodoTZInfo.UTC, 0, 0, null);
  }

  public BodoSQLRelDataTypeSystem(
      BodoTZInfo tzInfo, Integer weekStart, Integer weekOfYearPolicy, String currentDatabase) {
    defaultTZInfo = tzInfo;
    this.weekStart = weekStart;
    this.weekOfYearPolicy = weekOfYearPolicy;
    this.currentDatabase = currentDatabase;
  }

  /**
   * Convert finding a common type on multiple Char to VarChar. This ensures we won't append extra
   * spaces.
   */
  @Override
  public boolean shouldConvertRaggedUnionTypesToVarying() {
    return true;
  }

  // TODO: Go over these methods and update default to match Snowflake.
  // All other methods just call the method for defaultTypeSystem

  @Override
  public int getMaxScale(SqlTypeName typeName) {
    switch (typeName) {
      case DECIMAL:
        return getMaxNumericScale();
      default:
        return super.getMaxScale(typeName);
    }
  }

  @Override
  public int getDefaultPrecision(SqlTypeName typeName) {
    switch (typeName) {
      case INTERVAL_YEAR:
      case INTERVAL_YEAR_MONTH:
      case INTERVAL_MONTH:
      case INTERVAL_DAY:
      case INTERVAL_DAY_HOUR:
      case INTERVAL_DAY_MINUTE:
      case INTERVAL_DAY_SECOND:
      case INTERVAL_HOUR:
      case INTERVAL_HOUR_MINUTE:
      case INTERVAL_HOUR_SECOND:
      case INTERVAL_MINUTE:
      case INTERVAL_MINUTE_SECOND:
      case INTERVAL_SECOND:
        return SqlTypeName.MAX_INTERVAL_START_PRECISION;
      case BOOLEAN:
      case CHAR:
        return 1;
      case BINARY:
      case VARCHAR:
        return RelDataType.PRECISION_NOT_SPECIFIED;
        // Snowflake:
        // INT , INTEGER , BIGINT , SMALLINT , TINYINT , BYTEINT
        // Synonymous with NUMBER, except that precision and scale cannot be specified
      case TINYINT:
      case SMALLINT:
        // NOTE Disabling integer as it impacted bitwise and other time epoch tests.
        // case INTEGER:
      case BIGINT:
      case DECIMAL:
        return getMaxNumericPrecision();
        // Snowflake: Time precision default is 9.
      case TIME:
      case TIMESTAMP:
        return MAX_DATETIME_PRECISION;
      default:
        return super.getDefaultPrecision(typeName);
    }
  }

  @Override
  public int getMaxPrecision(SqlTypeName typeName) {
    // TODO(HA anothr PR): port other defaults.
    // These are needed as getCastSpec call getMaxPrecision
    switch (typeName) {
      case TINYINT:
      case SMALLINT:
      case BIGINT:
      case DECIMAL:
        return getMaxNumericPrecision();
      case TIME:
      case TIMESTAMP:
        return MAX_DATETIME_PRECISION;
      case VARBINARY:
      case BINARY:
        return 8388608;
      case CHAR:
      case VARCHAR:
        return 16777216;
      default:
        return super.getMaxPrecision(typeName);
    }
  }

  @Override
  public int getMaxNumericScale() {
    return 37;
  }

  @Override
  public int getMaxNumericPrecision() {
    return 38;
  }

  @Override
  public RelDataType deriveSumType(RelDataTypeFactory typeFactory, RelDataType argumentType) {
    switch (argumentType.getSqlTypeName()) {
      case TINYINT:
      case SMALLINT:
      case INTEGER:
      case BIGINT:
        // Sum always returns an int64 in Bodo for integers
        return typeFactory.createTypeWithNullability(
            typeFactory.createSqlType(SqlTypeName.BIGINT), argumentType.isNullable());
      default:
        // Match the calcite defaults if we haven't explicitly supported it.
        // Other types may need to be updated in the future.
        return super.deriveSumType(typeFactory, argumentType);
    }
  }

  @Override
  // This function, misleadingly is also used to get the types of a number of different
  // aggregations, namely STD and VAR. In Calcite, it returns the type of the input, which
  // is correct by the ANSI SQL standard. However, for our purposes, it makes more sense
  // for this to upcast to double.
  public RelDataType deriveAvgAggType(RelDataTypeFactory typeFactory, RelDataType argumentType) {
    return typeFactory.createTypeWithNullability(
        typeFactory.createSqlType(SqlTypeName.DOUBLE), argumentType.isNullable());
  }

  public BodoTZInfo getDefaultTZInfo() {
    return defaultTZInfo;
  }

  public @Nullable Integer getWeekStart() {
    return weekStart;
  }

  public @Nullable Integer getWeekOfYearPolicy() {
    return weekOfYearPolicy;
  }

  public @Nullable String getCatalogName() {
    return currentDatabase;
  }

  /**
   * Helper function that, given a decimal scale, returns the minimum sized integer that can hold
   * that number. Note that this function returns bigint even if the scale is so large that a bigint
   * could not hold it.
   */
  public static SqlTypeName getMinIntegerSize(int scale) {
    if (scale < 3) {
      // 127
      return SqlTypeName.TINYINT;
    } else if (scale < 5) {
      // 32767
      return SqlTypeName.SMALLINT;
    } else if (scale < 10) {
      // 2 147 483 647
      return SqlTypeName.INTEGER;
    } else {
      return SqlTypeName.BIGINT;
    }
  }
}
