package com.bodosql.calcite.application.BodoSQLTypeSystems;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataTypeSystemImpl;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Class for the RelDataTypeSystem used by BodoSQL. */
public class BodoSQLRelDataTypeSystem extends RelDataTypeSystemImpl {
  private final BodoTZInfo defaultTZInfo;

  // This is found in SqlTypeName.
  // Added it here for simplicity.
  // TODO (HA): update with other default tickets.
  public static final int MAX_DATETIME_PRECISION = 9;

  public static final int MAX_STRING_PRECISION = 16777216;

  public static final int MAX_BINARY_PRECISION = 8388608;
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

  // TODO(aneesh) remove enableStreamingSort once streaming sort is fully implemented
  public boolean enableStreamingSort;
  // TODO(ivan) remove enableStreamingSortLimitOffset once streaming sort limit and offset is fully
  // implemented
  public boolean enableStreamingSortLimitOffset;

  public static class CatalogContext {
    public String currentDatabase;
    public String currentAccount;

    public CatalogContext(String database, String account) {
      this.currentDatabase = database;
      this.currentAccount = account;
    }
  }

  /*
  CURRENT_DATABSE: Name of the database in use for the current session.

  */
  private CatalogContext currentDatabase;

  public BodoSQLRelDataTypeSystem() {
    this(BodoTZInfo.UTC, 0, 0, null, true, true);
  }

  public BodoSQLRelDataTypeSystem(
      boolean enableStreamingSort, boolean enableStreamingSortLimitOffset) {
    this(BodoTZInfo.UTC, 0, 0, null, enableStreamingSort, enableStreamingSortLimitOffset);
  }

  public BodoSQLRelDataTypeSystem(
      BodoTZInfo tzInfo,
      Integer weekStart,
      Integer weekOfYearPolicy,
      @Nullable CatalogContext currentDatabase,
      boolean enableStreamingSort,
      boolean enableStreamingSortLimitOffset) {
    defaultTZInfo = tzInfo;
    this.weekStart = weekStart;
    this.weekOfYearPolicy = weekOfYearPolicy;
    this.currentDatabase = currentDatabase;
    this.enableStreamingSort = enableStreamingSort;
    this.enableStreamingSortLimitOffset = enableStreamingSortLimitOffset;
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
      case VARBINARY:
      case VARCHAR:
        return getMaxPrecision(typeName);
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
      case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
      case TIMESTAMP_TZ:
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
      case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
      case TIMESTAMP_TZ:
        return MAX_DATETIME_PRECISION;
      case VARBINARY:
      case BINARY:
        return MAX_BINARY_PRECISION;
      case CHAR:
      case VARCHAR:
        return MAX_STRING_PRECISION;
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

  /** Derives the output type of functions in the AVG, VAR, STD family. */
  public RelDataType deriveAvgVarStdType(
      RelDataTypeFactory typeFactory, SqlOperator operator, RelDataType argumentType) {
    // For Avg/Var, the output will also be decimal
    boolean isAvg = operator.kind == SqlKind.AVG;
    boolean isVar = (operator.kind == SqlKind.VAR_SAMP) || (operator.kind == SqlKind.VAR_POP);
    boolean isSample =
        (operator.kind == SqlKind.STDDEV_SAMP) || (operator.kind == SqlKind.VAR_SAMP);
    boolean outNullable = argumentType.isNullable() || isSample;
    if ((isAvg || isVar) && argumentType.getSqlTypeName() == SqlTypeName.DECIMAL) {
      // The new precision must be the maximum because we don't know how much the magnitude was
      // increased by sums/products across many rows.
      int prec = getMaxNumericPrecision();
      int scale = argumentType.getScale();
      // For Var: double the scale (increasing it to no more than 12) to account for multiplying by
      // itself
      // (follows multiplication protocols).
      if (isVar) {
        scale = Integer.max(scale, Integer.min(scale * 2, 12));
      }
      // Then for Avg or Var: increase the scale by 6, but do not increase beyond 12 (follows
      // division protocols).
      scale = Integer.max(scale, Integer.min(scale + 6, 12));
      return typeFactory.createTypeWithNullability(
          typeFactory.createSqlType(SqlTypeName.DECIMAL, prec, scale), outNullable);
    }
    return typeFactory.createTypeWithNullability(
        typeFactory.createSqlType(SqlTypeName.DOUBLE), outNullable);
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

  public @Nullable CatalogContext getCatalogContext() {
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

  /**
   * Infers the return type of a decimal addition. Decimal addition involves at least one decimal
   * operand and requires both operands to have exact numeric types.
   *
   * <p>This matches the Snowflake type definition.
   * https://docs.snowflake.com/en/sql-reference/operators-arithmetic#addition-and-subtraction
   *
   * @param typeFactory TypeFactory used to create output type
   * @param type1 Type of the first operand
   * @param type2 Type of the second operand
   * @return Result type for a decimal addition
   */
  @Override
  @Nullable
  public RelDataType deriveDecimalPlusType(
      RelDataTypeFactory typeFactory, RelDataType type1, RelDataType type2) {
    if (SqlTypeUtil.isExactNumeric(type1) && SqlTypeUtil.isExactNumeric(type2)) {
      if (SqlTypeUtil.isDecimal(type1) || SqlTypeUtil.isDecimal(type2)) {
        // Java numeric will always have invalid precision/scale,
        // use its default decimal precision/scale instead.
        type1 = RelDataTypeFactoryImpl.isJavaType(type1) ? typeFactory.decimalOf(type1) : type1;
        type2 = RelDataTypeFactoryImpl.isJavaType(type2) ? typeFactory.decimalOf(type2) : type2;
        int p1 = type1.getPrecision();
        int p2 = type2.getPrecision();
        int s1 = type1.getScale();
        int s2 = type2.getScale();
        int l1 = p1 - s1;
        int l2 = p2 - s2;
        int l = Math.max(l1, l2) + 1;
        final int scale = Math.min(Math.max(s1, s2), getMaxNumericScale());
        final int precision = Math.min(l + scale, getMaxNumericPrecision());
        return typeFactory.createSqlType(SqlTypeName.DECIMAL, precision, scale);
      }
    }
    return null;
  }

  /**
   * Infers the return type of a decimal multiplication. Decimal multiplication involves at least
   * one decimal operand and requires both operands to have exact numeric types.
   *
   * <p>This matches the Snowflake type definition.
   * https://docs.snowflake.com/en/sql-reference/operators-arithmetic#multiplication
   *
   * @param typeFactory TypeFactory used to create output type
   * @param type1 Type of the first operand
   * @param type2 Type of the second operand
   * @return Result type for a decimal multiplication, or null if decimal multiplication should not
   *     be applied to the operands
   */
  @Override
  @Nullable
  public RelDataType deriveDecimalMultiplyType(
      RelDataTypeFactory typeFactory, RelDataType type1, RelDataType type2) {
    if (SqlTypeUtil.isExactNumeric(type1) && SqlTypeUtil.isExactNumeric(type2)) {
      if (SqlTypeUtil.isDecimal(type1) || SqlTypeUtil.isDecimal(type2)) {
        // Java numeric will always have invalid precision/scale,
        // use its default decimal precision/scale instead.
        type1 = RelDataTypeFactoryImpl.isJavaType(type1) ? typeFactory.decimalOf(type1) : type1;
        type2 = RelDataTypeFactoryImpl.isJavaType(type2) ? typeFactory.decimalOf(type2) : type2;
        int p1 = type1.getPrecision();
        int p2 = type2.getPrecision();
        int s1 = type1.getScale();
        int s2 = type2.getScale();

        int l1 = p1 - s1;
        int l2 = p2 - s2;
        int l = l1 + l2;
        int s = Math.min(s1 + s2, Math.max(Math.max(s1, s2), 12));
        final int scale = Math.min(s, getMaxNumericScale());
        final int precision = Math.min(l + scale, getMaxNumericPrecision());

        RelDataType ret;
        ret = typeFactory.createSqlType(SqlTypeName.DECIMAL, precision, scale);

        return ret;
      }
    }

    return null;
  }

  /**
   * Infers the return type of a decimal division. Decimal division involves at least one decimal
   * operand and requires both operands to have exact numeric types.
   *
   * <p>This matches the Snowflake type definition.
   * https://docs.snowflake.com/en/sql-reference/operators-arithmetic#division
   *
   * @param typeFactory TypeFactory used to create output type
   * @param type1 Type of the first operand
   * @param type2 Type of the second operand
   * @return Result type for a decimal division, or null if decimal division should not be applied
   *     to the operands
   */
  @Override
  @Nullable
  public RelDataType deriveDecimalDivideType(
      RelDataTypeFactory typeFactory, RelDataType type1, RelDataType type2) {

    if (SqlTypeUtil.isExactNumeric(type1) && SqlTypeUtil.isExactNumeric(type2)) {
      if (SqlTypeUtil.isDecimal(type1) || SqlTypeUtil.isDecimal(type2)) {
        // Java numeric will always have invalid precision/scale,
        // use its default decimal precision/scale instead.
        type1 = RelDataTypeFactoryImpl.isJavaType(type1) ? typeFactory.decimalOf(type1) : type1;
        type2 = RelDataTypeFactoryImpl.isJavaType(type2) ? typeFactory.decimalOf(type2) : type2;
        int p1 = type1.getPrecision();
        int s1 = type1.getScale();
        int s2 = type2.getScale();

        int l1 = p1 - s1;
        int l = l1 + s2;
        int s = Math.max(s1, Math.min(s1 + 6, 12));
        final int scale = Math.min(s, getMaxNumericScale());
        final int precision = Math.min(l + scale, getMaxNumericPrecision());

        RelDataType ret;
        ret = typeFactory.createSqlType(SqlTypeName.DECIMAL, precision, scale);

        return ret;
      }
    }

    return null;
  }
}
