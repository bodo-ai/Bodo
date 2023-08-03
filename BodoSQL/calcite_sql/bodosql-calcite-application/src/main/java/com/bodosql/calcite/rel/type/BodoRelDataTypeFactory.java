package com.bodosql.calcite.rel.type;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.TZAwareSqlType;

import org.checkerframework.checker.nullness.qual.Nullable;

public interface BodoRelDataTypeFactory extends RelDataTypeFactory {
  /**
   * Creates a {@link TZAwareSqlType} with the given time zone information.
   * @param tzInfo timezone information. if null, uses the type system default.
   * @return TZAwareSqlType
   */
  RelDataType createTZAwareSqlType(@Nullable BodoTZInfo tzInfo);

  static RelDataType createTZAwareSqlType(RelDataTypeFactory typeFactory, @Nullable BodoTZInfo tzInfo) {
    if (typeFactory instanceof BodoRelDataTypeFactory) {
      BodoRelDataTypeFactory bodoTypeFactory = (BodoRelDataTypeFactory) typeFactory;
      return bodoTypeFactory.createTZAwareSqlType(tzInfo);
    }

    // Timezone information is lost if used without an appropriate type factory.
    return typeFactory.createSqlType(SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE);
  }
}
