package com.bodosql.calcite.sql.validate.implicit;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.implicit.TypeCoercion;
import org.apache.calcite.sql.validate.implicit.TypeCoercionFactory;
import org.apache.calcite.sql.validate.implicit.TypeCoercionImpl;

import javax.annotation.Nullable;

public class BodoTypeCoercionImpl extends TypeCoercionImpl {
    public BodoTypeCoercionImpl(RelDataTypeFactory typeFactory, SqlValidator validator) {
        super(typeFactory, validator);
    }

    public static TypeCoercionFactory FACTORY = new TypeCoercionFactoryImpl();

    @Override
    public @Nullable RelDataType commonTypeForBinaryComparison(@Nullable RelDataType type1, @Nullable RelDataType type2) {
        if (type1 == null || type2 == null) {
            return null;
        }
        SqlTypeName typeName1 = type1.getSqlTypeName();
        SqlTypeName typeName2 = type2.getSqlTypeName();
        if (typeName1 == null || typeName2 == null) {
            return null;
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

    private static class TypeCoercionFactoryImpl implements TypeCoercionFactory {
        @Override
        public TypeCoercion create(RelDataTypeFactory typeFactory, SqlValidator validator) {
            return new BodoTypeCoercionImpl(typeFactory, validator);
        }
    }
}
