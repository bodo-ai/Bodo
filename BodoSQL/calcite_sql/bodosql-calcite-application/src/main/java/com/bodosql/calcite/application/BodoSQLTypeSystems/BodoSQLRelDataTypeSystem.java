package com.bodosql.calcite.application.BodoSQLTypeSystems;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.sql.type.SqlTypeName;

/**
 * Class for the RelDataTypeSystem used by BodoSQL. This recycles the default type system but
 * specifies that chars should be converted to varchars.
 */
public class BodoSQLRelDataTypeSystem implements RelDataTypeSystem {
  // A copy of the default type system. This will be used whenever
  // we don't need custom behavior.
  private RelDataTypeSystem defaultTypeSystem = RelDataTypeSystem.DEFAULT;

  /**
   * Convert finding a common type on multiple Char to VarChar. This ensures we won't append extra
   * spaces.
   */
  @Override
  public boolean shouldConvertRaggedUnionTypesToVarying() {
    return true;
  }

  // All other methods just call the method for defaultTypeSystem

  @Override
  public int getMaxScale(SqlTypeName typeName) {
    return defaultTypeSystem.getMaxScale(typeName);
  }

  @Override
  public int getDefaultPrecision(SqlTypeName typeName) {
    return defaultTypeSystem.getDefaultPrecision(typeName);
  }

  @Override
  public int getMaxPrecision(SqlTypeName typeName) {
    return defaultTypeSystem.getMaxPrecision(typeName);
  }

  @Override
  public int getMaxNumericScale() {
    return defaultTypeSystem.getMaxNumericScale();
  }

  @Override
  public int getMaxNumericPrecision() {
    return defaultTypeSystem.getMaxNumericPrecision();
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
    return defaultTypeSystem.deriveSumType(typeFactory, argumentType);
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
}
