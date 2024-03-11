package com.bodosql.calcite.rel.type;

import java.util.List;
import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactoryImpl;
import org.apache.calcite.rel.type.RelDataTypeFamily;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
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

    // Bodo Change:
    // Update Exact Numeric types to compute the max
    // total digits while maintaining scale
    RelDataType resultType = null;

    for (int i = 0; i < types.size(); ++i) {
      RelDataType type = types.get(i);
      RelDataTypeFamily family = type.getFamily();

      final SqlTypeName typeName = type.getSqlTypeName();
      if (typeName == SqlTypeName.NULL) {
        continue;
      }

      if (resultType == null) {
        resultType = type;
      }

      RelDataTypeFamily resultFamily = resultType.getFamily();
      SqlTypeName resultTypeName = resultType.getSqlTypeName();

      if (resultFamily != family) {
        resultType = null;
        break;
      }
      if (SqlTypeUtil.isExactNumeric(type)) {
        if (!type.equals(resultType)) {
          if (!typeName.allowsScale() && !resultTypeName.allowsScale()) {
            // Use the bigger primitive if we have only integers.
            // Note we don't use the value of the scale yet since something like
            // DECIMAL(21, 0) should perhaps be bigger than our bigint representation.
            if (type.getPrecision() > resultType.getPrecision()) {
              resultType = type;
            }
          } else {
            // Maintain the maximum precision via
            // Snowflake's calculation.
            // https://docs.snowflake.com/en/sql-reference/operators-arithmetic#other-n-ary-operations

            int p1 = resultType.getPrecision();
            int p2 = type.getPrecision();
            int s1 = resultType.getScale();
            int s2 = type.getScale();
            int sMax = Math.max(s1, s2);
            int l1 = p1 - s1;
            int l2 = p2 - s2;
            int lMax = Math.max(l1, l2);
            int pMax = lMax + sMax;
            final int maxPrecision = typeSystem.getMaxNumericPrecision();
            final int maxScale = typeSystem.getMaxNumericScale();
            int precision = Math.min(pMax, maxPrecision);
            int scale = Math.min(sMax, maxScale);

            resultType = createSqlType(SqlTypeName.DECIMAL, precision, scale);
          }
        }
      } else {
        // Type not supported in our conversion
        resultType = null;
        break;
      }
    }
    if (resultType != null) {
      return createTypeWithNullability(resultType, nullCount > 0);
    } else {
      return super.leastRestrictive(types);
    }
  }

  // Copied from RelDataTypeFactoryImpl to fix the implicit precision for types
  @Override
  public RelDataType decimalOf(RelDataType type) {
    // create decimal type and sync nullability
    return createTypeWithNullability(decimalOf2(type), type.isNullable());
  }

  /** Create decimal type equivalent with the given {@code type} while sans nullability. */
  private RelDataType decimalOf2(RelDataType type) {
    assert SqlTypeUtil.isNumeric(type) || SqlTypeUtil.isNull(type);
    SqlTypeName typeName = type.getSqlTypeName();
    assert typeName != null;
    switch (typeName) {
      case DECIMAL:
        // Fix the precision when the type is JavaType.
        return RelDataTypeFactoryImpl.isJavaType(type)
            ? SqlTypeUtil.getMaxPrecisionScaleDecimal(this)
            : type;
      case TINYINT:
        return createSqlType(SqlTypeName.DECIMAL, 3, 0);
      case SMALLINT:
        return createSqlType(SqlTypeName.DECIMAL, 5, 0);
      case INTEGER:
        return createSqlType(SqlTypeName.DECIMAL, 10, 0);
      case BIGINT:
        return createSqlType(SqlTypeName.DECIMAL, 38, 0);
      case REAL:
      case FLOAT:
      case DOUBLE:
        return createSqlType(
            SqlTypeName.DECIMAL, typeSystem.getMaxPrecision(SqlTypeName.DECIMAL), 15);
      default:
        // default precision and scale.
        return createSqlType(SqlTypeName.DECIMAL);
    }
  }
}
