package com.bodosql.calcite.application.BodoSQLConformance;

import static org.apache.calcite.sql.validate.SqlConformanceEnum.*;

import org.apache.calcite.sql.fun.SqlLibrary;
import org.apache.calcite.sql.validate.SqlAbstractConformance;
import org.apache.calcite.sql.validate.SqlConformanceEnum;

/** SQL Conformance that applies the default except where explicitly overridden. */
public class BodoSQLSqlConformance extends SqlAbstractConformance {

  // Default value used if we haven't set the conformance.
  private SqlConformanceEnum defaultConformance = SqlConformanceEnum.DEFAULT;

  @Override
  public boolean isLiberal() {
    return defaultConformance.isLiberal();
  }

  @Override
  public boolean allowCharLiteralAlias() {
    return defaultConformance.allowCharLiteralAlias();
  }

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

  @Override
  public boolean isSortByAliasObscures() {
    return defaultConformance.isSortByAliasObscures();
  }

  @Override
  public boolean isFromRequired() {
    return defaultConformance.isFromRequired();
  }

  @Override
  public boolean splitQuotedTableName() {
    return defaultConformance.splitQuotedTableName();
  }

  @Override
  public boolean allowHyphenInUnquotedTableName() {
    return defaultConformance.allowHyphenInUnquotedTableName();
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

  @Override
  public boolean isMinusAllowed() {
    return defaultConformance.isMinusAllowed();
  }

  @Override
  public boolean isApplyAllowed() {
    return defaultConformance.isApplyAllowed();
  }

  @Override
  public boolean isInsertSubsetColumnsAllowed() {
    return defaultConformance.isInsertSubsetColumnsAllowed();
  }

  @Override
  public boolean allowAliasUnnestItems() {
    return defaultConformance.allowAliasUnnestItems();
  }

  @Override
  public boolean allowNiladicParentheses() {
    return defaultConformance.allowNiladicParentheses();
  }

  @Override
  public boolean allowExplicitRowValueConstructor() {
    return defaultConformance.allowExplicitRowValueConstructor();
  }

  @Override
  public boolean allowExtend() {
    return defaultConformance.allowExtend();
  }

  @Override
  public boolean isLimitStartCountAllowed() {
    return defaultConformance.isLimitStartCountAllowed();
  }

  @Override
  public boolean allowGeometry() {
    return defaultConformance.allowGeometry();
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

  @Override
  public boolean allowExtendedTrim() {
    return defaultConformance.allowExtendedTrim();
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

  @Override
  public boolean allowQualifyingCommonColumn() {
    return defaultConformance.allowQualifyingCommonColumn();
  }

  @Override
  public SqlLibrary semantics() {
    return defaultConformance.semantics();
  }
}
