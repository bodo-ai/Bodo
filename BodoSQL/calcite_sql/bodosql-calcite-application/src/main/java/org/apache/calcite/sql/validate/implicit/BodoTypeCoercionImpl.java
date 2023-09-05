package org.apache.calcite.sql.validate.implicit;

import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeFactoryImpl;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlDynamicParam;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.BodoSqlTypeUtil;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.type.VariantSqlType;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorScope;

import org.checkerframework.checker.nullness.qual.Nullable;

import static java.util.Objects.requireNonNull;

public class BodoTypeCoercionImpl extends TypeCoercionImpl {
  public BodoTypeCoercionImpl(RelDataTypeFactory typeFactory, SqlValidator validator) {
    super(typeFactory, validator);
  }

  public static TypeCoercionFactory FACTORY = new TypeCoercionFactoryImpl();

  @Override
  public @Nullable RelDataType getTightestCommonType(
      @Nullable RelDataType type1, @Nullable RelDataType type2) {
    if (SqlTypeUtil.isNumeric(type1) && SqlTypeUtil.isBoolean(type2)) {
      return type2;
    }
    if (SqlTypeUtil.isNumeric(type2) && SqlTypeUtil.isBoolean(type1)) {
      return type1;
    }
    return super.getTightestCommonType(type1, type2);
  }

  @Override
  public @Nullable RelDataType implicitCast(RelDataType in, SqlTypeFamily expected) {
    // Allow casting from Variant to any string.
    // TODO: Add the other type families.
    if (in instanceof VariantSqlType && expected != SqlTypeFamily.ANY && expected.getTypeNames().contains(SqlTypeName.VARCHAR)) {
      return factory.createTypeWithNullability(
              factory.createSqlType(SqlTypeName.VARCHAR),
              in.isNullable()
      );
    }
    if ((SqlTypeUtil.isNumeric(in) || SqlTypeUtil.isCharacter(in))
        && expected == SqlTypeFamily.BOOLEAN) {
      return factory.createSqlType(SqlTypeName.BOOLEAN);
    }

    if ((SqlTypeUtil.isCharacter(in)) && (expected == SqlTypeFamily.DATETIME)) {
      return factory.createSqlType(SqlTypeName.TIMESTAMP);
    }
    return super.implicitCast(in, expected);
  }

  @Override
  public @Nullable RelDataType commonTypeForBinaryComparison(
      @Nullable RelDataType type1, @Nullable RelDataType type2) {
    if (type1 == null || type2 == null) {
      return null;
    }
    SqlTypeName typeName1 = type1.getSqlTypeName();
    SqlTypeName typeName2 = type2.getSqlTypeName();
    if (typeName1 == null || typeName2 == null) {
      return null;
    }
    // Any variant type should be the common type.
    if (type1 instanceof VariantSqlType) {
      return type1;
    }
    if (type2 instanceof VariantSqlType) {
      return type2;
    }

    // BOOLEAN + NUMERIC -> BOOLEAN
    if (SqlTypeUtil.isBoolean(type1) && SqlTypeUtil.isNumeric(type2)) {
      return type1;
    }
    if (SqlTypeUtil.isNumeric(type1) && SqlTypeUtil.isBoolean(type2)) {
      return type2;
    }
    return super.commonTypeForBinaryComparison(type1, type2);
  }

  @Override
  protected boolean coerceOperandType(
      @Nullable SqlValidatorScope scope,
      SqlCall call,
      int index,
      RelDataType targetType) {
    // Mostly duplicated from calcite. We need to customize this operation
    // because calcite's support for custom datatypes isn't very well supported.
    // It's easier to just copy the code for this method and intercept the
    // coercion logic for tz aware timestamps than it is to try and figure out how
    // to get calcite to recognize the custom datatype.
    //
    // This method is really only required because of tz aware timestamps.
    // Tz aware timestamps are only needed because Bodo enforces that columns
    // must have a single timezone and that operations must be performed on two
    // columns with the same timezone information so we need to explicitly add
    // a cast when they differ. Snowflake doesn't have this requirement and snowflake
    // treats timezones as only the offset and the offsets as being part of the value
    // itself and not the column/data type. If we chose to lighten the requirement
    // for timezone logic, this method and the tz aware timestamp itself wouldn't
    // be needed.
    //
    // Keeping this note here just in case we decide to revisit this decision in
    // the future to make it easier for anyone looking at this to understand
    // why this is here.

    // Original Calcite code follows from here.

    // Transform the JavaType to SQL type because the SqlDataTypeSpec
    // does not support deriving JavaType yet.
    if (RelDataTypeFactoryImpl.isJavaType(targetType)) {
      targetType = ((JavaTypeFactory) factory).toSql(targetType);
    }

    SqlNode operand = call.getOperandList().get(index);
    if (operand instanceof SqlDynamicParam) {
      // Do not support implicit type coercion for dynamic param.
      return false;
    }
    requireNonNull(scope, "scope");
    // Check it early.
    if (!needToCast(scope, operand, targetType)) {
      return false;
    }
    // Fix up nullable attr.
    RelDataType targetType1 = syncAttributes(validator.deriveType(scope, operand), targetType);
    // Customize the castTo operation to use our own implementation.
    SqlNode desired = castTo(operand, targetType1);
    call.setOperand(index, desired);
    updateInferredType(desired, targetType1);
    return true;
  }

  private static SqlNode castTo(SqlNode node, RelDataType type) {
    // Utilize our own version of convertTypeToSpec.
    return SqlStdOperatorTable.CAST.createCall(SqlParserPos.ZERO, node,
        BodoSqlTypeUtil.convertTypeToSpec(type).withNullable(type.isNullable()));
  }

  private static class TypeCoercionFactoryImpl implements TypeCoercionFactory {
    @Override
    public TypeCoercion create(RelDataTypeFactory typeFactory, SqlValidator validator) {
      return new BodoTypeCoercionImpl(typeFactory, validator);
    }
  }
}
