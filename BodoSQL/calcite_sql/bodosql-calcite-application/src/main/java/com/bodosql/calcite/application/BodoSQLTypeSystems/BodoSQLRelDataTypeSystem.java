package com.bodosql.calcite.application.BodoSQLTypeSystems;

import javax.annotation.Nullable;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.sql.type.SqlTypeName;

/**
 * Class for the RelDataTypeSystem used by BodoSQL. This recycles the default type system but
 * specifies that chars should be converted to varchars.
 */
public class BodoSQLRelDataTypeSystem implements RelDataTypeSystem {
  // A copy of the default type system. This will be used whenever
  // we don't need custom behavior.
  private RelDataTypeSystem defaultTypeSystem = RelDataTypeSystem.DEFAULT;

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

  public BodoSQLRelDataTypeSystem() {
    this(BodoTZInfo.UTC, 0, 0);
  }

  public BodoSQLRelDataTypeSystem(BodoTZInfo tzInfo, Integer weekStart, Integer weekOfYearPolicy) {
    defaultTZInfo = tzInfo;
    this.weekStart = weekStart;
    this.weekOfYearPolicy = weekOfYearPolicy;
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
    }
    return defaultTypeSystem.getMaxScale(typeName);
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
        return 1;
      case BINARY:
        return RelDataType.PRECISION_NOT_SPECIFIED;
        // Snowflake:
        // INT , INTEGER , BIGINT , SMALLINT , TINYINT , BYTEINT
        // Synonymous with NUMBER, except that precision and scale cannot be specified
      case TINYINT:
      case SMALLINT:
        // NOTE Disbling integer as it impacted bitwise and other time epoch tests.
        // case INTEGER:
      case BIGINT:
      case DECIMAL:
        return getMaxNumericPrecision();
        // Snowflake: Time precision defaul is 9.
      case TIME:
      case TIMESTAMP:
        return MAX_DATETIME_PRECISION;
    }
    return defaultTypeSystem.getDefaultPrecision(typeName);
  }

  @Override
  public int getMaxPrecision(SqlTypeName typeName) {
    // TODO(HA anothr PR): port other defaults.
    // These're needed as getCastSpec call getMaxPrecision
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
    }
    return defaultTypeSystem.getMaxPrecision(typeName);
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
  public String getLiteral(SqlTypeName typeName, boolean isPrefix) {
    return defaultTypeSystem.getLiteral(typeName, isPrefix);
  }

  @Override
  public boolean isCaseSensitive(SqlTypeName typeName) {
    return defaultTypeSystem.isCaseSensitive(typeName);
  }

  @Override
  public boolean isAutoincrement(SqlTypeName typeName) {
    return defaultTypeSystem.isAutoincrement(typeName);
  }

  @Override
  public int getNumTypeRadix(SqlTypeName typeName) {
    return defaultTypeSystem.getNumTypeRadix(typeName);
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
        // By default match the calcite defaults. Other types may need to be updated in the future.
        return defaultTypeSystem.deriveSumType(typeFactory, argumentType);
    }
  }

  @Override
  // This function, misleadingly is also used to get the types of a number of different
  // aggregations, namely STD and VAR. In Calcite, it returns the type of the input, which
  // is correct by the ANSI SQL standard. However, for our purposes, it makes more sense
  // for this to upcast to double.
  public RelDataType deriveAvgAggType(RelDataTypeFactory typeFactory, RelDataType argumentType) {
    return typeFactory.createSqlType(SqlTypeName.DOUBLE);
  }

  @Override
  public RelDataType deriveCovarType(
      RelDataTypeFactory typeFactory, RelDataType arg0Type, RelDataType arg1Type) {
    return defaultTypeSystem.deriveCovarType(typeFactory, arg0Type, arg1Type);
  }

  @Override
  public RelDataType deriveFractionalRankType(RelDataTypeFactory typeFactory) {
    return defaultTypeSystem.deriveFractionalRankType(typeFactory);
  }

  @Override
  public RelDataType deriveRankType(RelDataTypeFactory typeFactory) {
    return defaultTypeSystem.deriveRankType(typeFactory);
  }

  @Override
  public boolean isSchemaCaseSensitive() {
    return defaultTypeSystem.isSchemaCaseSensitive();
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
}
