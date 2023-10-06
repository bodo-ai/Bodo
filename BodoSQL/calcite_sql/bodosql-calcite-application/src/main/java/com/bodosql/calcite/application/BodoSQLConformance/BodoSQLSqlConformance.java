package com.bodosql.calcite.application.BodoSQLConformance;

import org.apache.calcite.sql.validate.SqlAbstractConformance;

/**
 * SQL Conformance that matches the SqlAbstractConformance default except where explicitly
 * overridden.
 */
public class BodoSQLSqlConformance extends SqlAbstractConformance {

  /**
   * Allow using an alias in groupby
   *
   * @return true
   */
  @Override
  public boolean isGroupByAlias() {
    return true;
  }

  /**
   * Allow using a column number in groupby
   *
   * @return true
   */
  @Override
  public boolean isGroupByOrdinal() {
    return true;
  }

  /**
   * Allow using an alias in having
   *
   * @return true
   */
  @Override
  public boolean isHavingAlias() {
    return true;
  }

  /**
   * Allow using column numbers in sort by
   *
   * @return true
   */
  @Override
  public boolean isSortByOrdinal() {
    return true;
  }

  /**
   * Allow using an alias in sortBy
   *
   * @return true
   */
  @Override
  public boolean isSortByAlias() {
    return true;
  }

  /**
   * Allow !=
   *
   * @return True
   */
  @Override
  public boolean isBangEqualAllowed() {
    return true;
  }

  /**
   * Allow %
   *
   * @return True
   */
  @Override
  public boolean isPercentRemainderAllowed() {
    return true;
  }

  /**
   * Convert multiple chars to Varying
   *
   * @return True
   */
  @Override
  public boolean shouldConvertRaggedUnionTypesToVarying() {
    return true;
  }

  /**
   * Allow time usnits ending in s
   *
   * @return True
   */
  @Override
  public boolean allowPluralTimeUnits() {
    return true;
  }
}
