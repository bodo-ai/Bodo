package org.apache.calcite.sql.type;

import com.bodosql.calcite.sql.validate.BodoCoercionUtil;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelRecordType;
import org.apache.calcite.schema.Function;
import org.apache.calcite.sql.SnowflakeUserDefinedBaseFunction;
import org.apache.calcite.sql.SqlBasicTypeNameSpec;
import org.apache.calcite.sql.SqlCollectionTypeNameSpec;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlMapTypeNameSpec;
import org.apache.calcite.sql.SqlRowTypeNameSpec;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.validate.SqlUserDefinedFunction;
import org.apache.calcite.sql.validate.SqlUserDefinedTableFunction;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlTypeNameSpec;
import org.apache.calcite.sql.VariantTypeNameSpec;

import java.math.BigDecimal;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import static java.util.Objects.requireNonNull;
import static org.apache.calcite.sql.type.NonNullableAccessors.getComponentTypeOrThrow;
import static org.apache.calcite.sql.type.SqlTypeUtil.inCharFamily;
import static org.apache.calcite.sql.type.SqlTypeUtil.isAtomic;
import static org.apache.calcite.sql.type.SqlTypeUtil.isCollection;
import static org.apache.calcite.sql.type.SqlTypeUtil.isMap;
import static org.apache.calcite.sql.type.SqlTypeUtil.isNull;
import static org.apache.calcite.sql.type.SqlTypeUtil.isRow;

public class BodoSqlTypeUtil {
  public static SqlDataTypeSpec convertTypeToSpec(RelDataType type) {
    // Note: This is copied from Calcite.
    String charSetName = inCharFamily(type) ? type.getCharset().name() : null;
    return convertTypeToSpec(type, charSetName, -1, -1);
  }

  /**
   * Extends SqlTypeUtil.convertTypeToSpec with support for our custom data types.
   * Note we cannot dispatch to the SqlTypeUtil.convertTypeToSpec because there are
   * recursive calls.
   */
  public static SqlDataTypeSpec convertTypeToSpec(RelDataType type, @Nullable String charSetName, int maxPrecision, int maxScale) {
    if (type instanceof VariantSqlType) {
      SqlTypeNameSpec typeNameSpec = new VariantTypeNameSpec(Objects.requireNonNull(type.getSqlIdentifier()));
      return new SqlDataTypeSpec(typeNameSpec, SqlParserPos.ZERO);
    } else {
      SqlTypeName typeName = type.getSqlTypeName();

      // TODO jvs 28-Dec-2004:  support row types, user-defined types,
      // interval types, multiset types, etc
      assert typeName != null;

      final SqlTypeNameSpec typeNameSpec;
      if (isAtomic(type) || isNull(type)
              || type.getSqlTypeName() == SqlTypeName.UNKNOWN
              || type.getSqlTypeName() == SqlTypeName.GEOMETRY) {
        int precision = typeName.allowsPrec() ? type.getPrecision() : -1;
        // fix up the precision.
        if (maxPrecision > 0 && precision > maxPrecision) {
          precision = maxPrecision;
        }
        int scale = typeName.allowsScale() ? type.getScale() : -1;
        if (maxScale > 0 && scale > maxScale) {
          scale = maxScale;
        }

        typeNameSpec =
                new SqlBasicTypeNameSpec(typeName, precision, scale, charSetName,
                        SqlParserPos.ZERO);
      } else if (isCollection(type)) {
        typeNameSpec =
                new SqlCollectionTypeNameSpec(
                        convertTypeToSpec(getComponentTypeOrThrow(type)).getTypeNameSpec(),
                        typeName, SqlParserPos.ZERO);
      } else if (isRow(type)) {
        RelRecordType recordType = (RelRecordType) type;
        List<RelDataTypeField> fields = recordType.getFieldList();
        List<SqlIdentifier> fieldNames = fields.stream()
                .map(f -> new SqlIdentifier(f.getName(), SqlParserPos.ZERO))
                .collect(Collectors.toList());
        List<SqlDataTypeSpec> fieldTypes = fields.stream()
                .map(f -> convertTypeToSpec(f.getType()))
                .collect(Collectors.toList());
        typeNameSpec = new SqlRowTypeNameSpec(SqlParserPos.ZERO, fieldNames, fieldTypes);
      } else if (isMap(type)) {
        final RelDataType keyType =
                requireNonNull(type.getKeyType(), () -> "keyType of " + type);
        final RelDataType valueType =
                requireNonNull(type.getValueType(), () -> "valueType of " + type);
        final SqlDataTypeSpec keyTypeSpec = convertTypeToSpec(keyType);
        final SqlDataTypeSpec valueTypeSpec = convertTypeToSpec(valueType);
        typeNameSpec = new SqlMapTypeNameSpec(keyTypeSpec, valueTypeSpec, SqlParserPos.ZERO);
      } else {
        throw new UnsupportedOperationException(
                "Unsupported type when convertTypeToSpec: " + typeName);
      }

      // REVIEW jvs 28-Dec-2004:  discriminate between precision/scale
      // zero and unspecified?

      // REVIEW angel 11-Jan-2006:
      // Use neg numbers to indicate unspecified precision/scale

      return new SqlDataTypeSpec(typeNameSpec, SqlParserPos.ZERO);
    }
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

  /**
   * Bodo extension to SqlTypeUtil.equalSansNullability with special behavior
   * when comparing two literals.
   */
  public static boolean literalEqualSansNullability(
          RelDataType type1,
          RelDataType type2) {
    if (SqlTypeFamily.CHARACTER.contains(type1) && SqlTypeFamily.CHARACTER.contains(type2)) {
      return true;
    } else if (SqlTypeFamily.INTEGER.contains(type1) && SqlTypeFamily.INTEGER.contains(type2)) {
      return true;
    } else {
      return SqlTypeUtil.equalSansNullability(type1, type2);
    }
  }

  /**
   * Expansion of SqlTypeUtil.isValidDecimalValue to also consider scale differences.
   */
  public static boolean isValidDecimalValue(@Nullable BigDecimal value, RelDataType toType) {
    if (value == null) {
      return true;
    }
    switch (toType.getSqlTypeName()) {
      case DECIMAL:
        final int intDigits = value.precision() - value.scale();
        final int maxIntDigits = toType.getPrecision() - toType.getScale();
        return intDigits <= maxIntDigits && value.scale() <= toType.getScale();
      default:
        return true;
    }
  }

  /**
   * Wrapper around SqlTypeUtil.canCastFrom
   * with specialized behavior for SnowflakeUDFs.
   */
  public static boolean canCastFromWrapper(
          SqlFunction function,
          RelDataType toType,
          RelDataType fromType,
          boolean coerce) {
    if (function instanceof SqlUserDefinedFunction) {
      Function functionImpl = ((SqlUserDefinedFunction) function).getFunction();
      if (functionImpl instanceof SnowflakeUserDefinedBaseFunction) {
        return BodoCoercionUtil.Companion.canCastFromUDF(fromType, toType, coerce);
      }
    } else if (function instanceof  SqlUserDefinedTableFunction) {
      Function functionImpl = ((SqlUserDefinedTableFunction) function).getFunction();
      if (functionImpl instanceof SnowflakeUserDefinedBaseFunction) {
        return BodoCoercionUtil.Companion.canCastFromUDF(fromType, toType, coerce);
      }
    }
    return SqlTypeUtil.canCastFrom(toType, fromType, coerce);
  }
}
