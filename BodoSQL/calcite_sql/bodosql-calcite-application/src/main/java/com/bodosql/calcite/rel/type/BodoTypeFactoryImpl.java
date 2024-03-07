package com.bodosql.calcite.rel.type;

import java.util.List;
import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.VariantSqlType;
import org.checkerframework.checker.nullness.qual.Nullable;

public class BodoTypeFactoryImpl extends JavaTypeFactoryImpl implements BodoRelDataTypeFactory {
  public BodoTypeFactoryImpl(RelDataTypeSystem typeSystem) {
    super(typeSystem);
  }

  @Override
  public RelDataType createSqlType(SqlTypeName typeName) {
    if (typeName == SqlTypeName.OTHER) {
      return createVariantSqlType();
    }
    return super.createSqlType(typeName);
  }

  @Override
  public RelDataType createSqlType(SqlTypeName typeName, int precision) {
    if (typeName == SqlTypeName.OTHER) {
      return createVariantSqlType();
    }
    return super.createSqlType(typeName, precision);
  }

  @Override
  public RelDataType createSqlType(SqlTypeName typeName, int precision, int scale) {
    if (typeName == SqlTypeName.OTHER) {
      return createVariantSqlType();
    }
    return super.createSqlType(typeName, precision, scale);
  }

  @Override
  public RelDataType createTypeWithNullability(final RelDataType type, final boolean nullable) {
    if (type instanceof VariantSqlType) {
      if (type.isNullable() == nullable) {
        return type;
      }
      return copyVariantSqlType((VariantSqlType) type, nullable);
    }
    return super.createTypeWithNullability(type, nullable);
  }

  @Override
  public RelDataType createVariantSqlType() {
    return canonize(new VariantSqlType(false));
  }

  public RelDataType createVariantSqlType(boolean nullable) {
    return canonize(new VariantSqlType(nullable));
  }

  private RelDataType copyVariantSqlType(VariantSqlType type, boolean nullable) {
    return canonize(new VariantSqlType(nullable));
  }

  /**
   * Implementation of leastRestrictive that handles Variant Types. Any other types are handled by
   * the parent class. Since ANY has priority over VARIANT this also needs to support variant.
   *
   * @param types The types to coerce to the least restrictive type.
   * @return The least restrictive type according to our rules.
   */
  @Override
  public @Nullable RelDataType leastRestrictive(List<RelDataType> types) {
    // Implementation of leastRestrictive that handles Variant Types.
    // Any other types are handled by the parent class.
    assert types != null;
    assert types.size() >= 1;
    int anyCount = 0;
    int nullCount = 0;
    boolean seenVariant = false;

    for (RelDataType type : types) {
      if (type.getSqlTypeName() == SqlTypeName.ANY) {
        anyCount++;
      }
      if (type.isNullable() || type.getSqlTypeName() == SqlTypeName.NULL) {
        nullCount++;
      }
      if (type instanceof VariantSqlType) {
        seenVariant = true;
      }
    }
    if (anyCount > 0) {
      return createTypeWithNullability(createSqlType(SqlTypeName.ANY), nullCount > 0);
    }
    if (seenVariant) {
      return createTypeWithNullability(createVariantSqlType(), nullCount > 0);
    }
    return super.leastRestrictive(types);
  }
}
