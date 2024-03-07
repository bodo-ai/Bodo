package com.bodosql.calcite.rel.type;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.type.VariantSqlType;

public interface BodoRelDataTypeFactory extends RelDataTypeFactory {
  /**
   * Creates a {@link VariantSqlType}.
   *
   * @return The VariantSQLType
   */
  RelDataType createVariantSqlType();

  static RelDataType createVariantSqlType(RelDataTypeFactory typeFactory) {
    if (typeFactory instanceof BodoRelDataTypeFactory) {
      BodoRelDataTypeFactory bodoTypeFactory = (BodoRelDataTypeFactory) typeFactory;
      return bodoTypeFactory.createVariantSqlType();
    }
    throw new UnsupportedOperationException(
        "createVariantSqlType() is only supported with the BodoRelDataTypeFactory");
  }
}
