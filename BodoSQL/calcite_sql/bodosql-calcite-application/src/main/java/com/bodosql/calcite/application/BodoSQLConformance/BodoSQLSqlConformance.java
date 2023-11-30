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

  /**
   * Whether {@code VALUE} is allowed as an alternative to {@code VALUES} in the parser.
   *
   * <p>Among the built-in conformance levels, true in {@link SqlConformanceEnum#BABEL}, {@link
   * SqlConformanceEnum#LENIENT}, {@link SqlConformanceEnum#MYSQL_5}; false otherwise.
   */
  @Override
  public boolean isValueAllowed() {
    return false;
  }

  /**
   * Whether to allow lenient type coercions.
   *
   * <p>Coercions include:
   *
   * <ul>
   *   <li>Coercion of string literal to array literal. For example, {@code SELECT ARRAY[0,1,2] ==
   *       '{0,1,2}'}
   *   <li>Casting {@code BOOLEAN} values to one of the following numeric types: {@code TINYINT},
   *       {@code SMALLINT}, {@code INTEGER}, {@code BIGINT}.
   * </ul>
   *
   * <p>Among the built-in conformance levels, true in {@link SqlConformanceEnum#BABEL}, false
   * otherwise.
   */
  @Override
  public boolean allowLenientCoercion() {
    return false;
  }
}
