package com.bodosql.calcite.rel.type;

import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.TZAwareSqlType;

import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem;

import org.checkerframework.checker.nullness.qual.Nullable;

import static java.util.Objects.requireNonNull;

public class BodoTypeFactoryImpl extends JavaTypeFactoryImpl implements BodoRelDataTypeFactory {
  public BodoTypeFactoryImpl(RelDataTypeSystem typeSystem) {
    super(typeSystem);
  }

  @Override
  public RelDataType createSqlType(SqlTypeName typeName) {
    if (typeName == SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
      return createTZAwareSqlType(null);
    }
    return super.createSqlType(typeName);
  }

  @Override
  public RelDataType createSqlType(SqlTypeName typeName, int precision) {
    if (typeName == SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
      return createTZAwareSqlType(null);
    }
    return super.createSqlType(typeName, precision);
  }

  @Override
  public RelDataType createSqlType(SqlTypeName typeName, int precision, int scale) {
    if (typeName == SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
      return createTZAwareSqlType(null);
    }
    return super.createSqlType(typeName, precision, scale);
  }

  @Override
  public RelDataType createTypeWithNullability(
      final RelDataType type,
      final boolean nullable) {
    if (type instanceof TZAwareSqlType) {
      if (type.isNullable() == nullable) {
        return type;
      }
      return copyTZAwareSqlType((TZAwareSqlType) type, nullable);
    }
    return super.createTypeWithNullability(type, nullable);
  }

  @Override
  public RelDataType createTZAwareSqlType(@Nullable BodoTZInfo tzInfo) {
    if (tzInfo == null) {
      if (typeSystem instanceof BodoSQLRelDataTypeSystem) {
        tzInfo = ((BodoSQLRelDataTypeSystem) typeSystem).getDefaultTZInfo();
      } else {
        tzInfo = BodoTZInfo.UTC;
      }
    }
    return canonize(new TZAwareSqlType(tzInfo, false));
  }

  private RelDataType copyTZAwareSqlType(TZAwareSqlType type, boolean nullable) {
    BodoTZInfo tzInfo = requireNonNull(type.getTZInfo());
    return canonize(new TZAwareSqlType(tzInfo, nullable));
  }
}
