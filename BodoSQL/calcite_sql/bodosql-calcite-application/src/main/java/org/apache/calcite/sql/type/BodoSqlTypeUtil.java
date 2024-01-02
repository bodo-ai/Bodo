package org.apache.calcite.sql.type;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlTypeNameSpec;
import org.apache.calcite.sql.SqlTzAwareTypeNameSpec;
import org.apache.calcite.sql.VariantTypeNameSpec;

import java.util.Objects;

import static org.apache.calcite.sql.type.SqlTypeUtil.inCharFamily;

public class BodoSqlTypeUtil {
  public static SqlDataTypeSpec convertTypeToSpec(RelDataType type) {
    // Note: This is copied from Calcite.
    String charSetName = inCharFamily(type) ? type.getCharset().name() : null;
    return convertTypeToSpec(type, charSetName, -1, -1);
  }

  public static SqlDataTypeSpec convertTypeToSpec(RelDataType type, @Nullable String charSetName, int maxPrecision, int maxScale) {
    if (type instanceof TZAwareSqlType) {
      TZAwareSqlType castType = (TZAwareSqlType) type;
      SqlTypeNameSpec typeNameSpec = new SqlTzAwareTypeNameSpec(castType.getSqlIdentifier(), castType.getPrecision(), castType.getTZInfo());
      return new SqlDataTypeSpec(typeNameSpec, SqlParserPos.ZERO);
    } else if (type instanceof VariantSqlType) {
      SqlTypeNameSpec typeNameSpec = new VariantTypeNameSpec(Objects.requireNonNull(type.getSqlIdentifier()));
      return new SqlDataTypeSpec(typeNameSpec, SqlParserPos.ZERO);
    }
    return SqlTypeUtil.convertTypeToSpec(type, charSetName, maxPrecision, maxScale);
  }

  /**
   * Bodo extension to SqlTypeUtil.equalSansNullability with special behavior
   * when comparing two literals.
   */
  public static boolean literalEqualSansNullability(
          RelDataTypeFactory factory,
          RelDataType type1,
          RelDataType type2) {
    if (SqlTypeFamily.CHARACTER.contains(type1) && SqlTypeFamily.CHARACTER.contains(type2)) {
      return true;
    } else if (SqlTypeFamily.INTEGER.contains(type1) && SqlTypeFamily.INTEGER.contains(type2)) {
      return true;
    } else {
      return SqlTypeUtil.equalSansNullability(factory, type1, type2);
    }
  }
}
