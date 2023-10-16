package org.apache.calcite.sql.type;

import com.google.common.base.Preconditions;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.sql.SqlCollation;
import org.apache.calcite.sql.SqlOperatorBinding;

import com.bodosql.calcite.rel.type.BodoRelDataTypeFactory;

import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.List;

import static java.util.Objects.requireNonNull;
import static org.apache.calcite.sql.type.NonNullableAccessors.getCharset;
import static org.apache.calcite.sql.type.NonNullableAccessors.getCollation;
import static org.apache.calcite.util.Static.RESOURCE;

public class BodoReturnTypes {

    /**
     * Convert a given type to have an unknown precision. This should
     * only be called on STRING_TYPES.
     */
    public static final SqlTypeTransform UNKNOWN_PRECISION =
            (opBinding, typeToTransform) ->
                    opBinding.getTypeFactory().createTypeWithNullability(
                            opBinding.getTypeFactory().createSqlType(typeToTransform.getSqlTypeName(), RelDataType.PRECISION_NOT_SPECIFIED),
                            typeToTransform.isNullable()
                    );

    /**
     * Type-inference strategy whereby the output type is equivalent to
     * ReturnTypes.ARG0_NULLABLE_VARYING but with an UNKNOWN precision.
     */
    public static final SqlReturnTypeInference ARG0_NULLABLE_VARYING_UNKNOWN_PRECISION = ReturnTypes.ARG0_NULLABLE_VARYING.andThen(UNKNOWN_PRECISION);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 1
     */
    public static final SqlReturnTypeInference VARCHAR_1 = ReturnTypes.explicit(SqlTypeName.VARCHAR, 1);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 1
     * and SqlTypeTransforms.TO_NULLABLE.
     */
    public static final SqlReturnTypeInference VARCHAR_1_NULLABLE = VARCHAR_1.andThen(SqlTypeTransforms.TO_NULLABLE);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 3
     */
    public static final SqlReturnTypeInference VARCHAR_3 = ReturnTypes.explicit(SqlTypeName.VARCHAR, 3);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 3
     * and SqlTypeTransforms.TO_NULLABLE.
     */
    public static final SqlReturnTypeInference VARCHAR_3_NULLABLE = VARCHAR_3.andThen(SqlTypeTransforms.TO_NULLABLE);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 32
     */
    public static final SqlReturnTypeInference VARCHAR_32 = ReturnTypes.explicit(SqlTypeName.VARCHAR, 32);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 32
     * and SqlTypeTransforms.TO_NULLABLE.
     */
    public static final SqlReturnTypeInference VARCHAR_32_NULLABLE = VARCHAR_32.andThen(SqlTypeTransforms.TO_NULLABLE);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 128
     */
    public static final SqlReturnTypeInference VARCHAR_128 = ReturnTypes.explicit(SqlTypeName.VARCHAR, 128);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 128
     * and SqlTypeTransforms.TO_NULLABLE.
     */
    public static final SqlReturnTypeInference VARCHAR_128_NULLABLE = VARCHAR_128.andThen(SqlTypeTransforms.TO_NULLABLE);

    /**
     * Type-inference strategy for returning a VARCHAR with an unknown precision
     */
    public static final SqlReturnTypeInference VARCHAR_UNKNOWN_PRECISION = ReturnTypes.explicit(SqlTypeName.VARCHAR, RelDataType.PRECISION_NOT_SPECIFIED);

    /**
     * Type-inference strategy whereby the output type is a VARCHAR with an unknown
     * precision and SqlTypeTransforms.TO_NULLABLE.
     */
    public static final SqlReturnTypeInference VARCHAR_UNKNOWN_PRECISION_NULLABLE = VARCHAR_UNKNOWN_PRECISION.andThen(SqlTypeTransforms.TO_NULLABLE);

    /**
     * Type-inference strategy whereby the result type of a call is an integer
     * if both operands ares.
     */
    public static final SqlReturnTypeInference DATE_SUB = opBinding -> {
        RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
        RelDataType type1 = opBinding.getOperandType(0);
        RelDataType type2 = opBinding.getOperandType(1);
        if (type1.getSqlTypeName() == SqlTypeName.DATE
                && type2.getSqlTypeName() == SqlTypeName.DATE) {
            return typeFactory.createSqlType(SqlTypeName.INTEGER);
        }
        // This rule doesn't apply
        return null;
    };

    /**
     * Same as {@link #DATE_SUB} but returns with nullability if any
     * of the operands is nullable by using
     * {@link SqlTypeTransforms#TO_NULLABLE}.
     */
    public static final SqlReturnTypeInference DATE_SUB_NULLABLE =
            DATE_SUB.andThen(SqlTypeTransforms.TO_NULLABLE);

    /**
     * Type-inference strategy whereby the result type of a call is
     * {@link ReturnTypes#DECIMAL_SUM_NULLABLE} with a fallback to date - date, and
     * finally with a fallback to {@link ReturnTypes#LEAST_RESTRICTIVE}
     * These rules are used for subtraction.
     */
    public static final SqlReturnTypeInference NULLABLE_SUB =
            new SqlReturnTypeInferenceChain(
                    ReturnTypes.DECIMAL_SUM_NULLABLE, DATE_SUB_NULLABLE, ReturnTypes.LEAST_RESTRICTIVE);

    public static final SqlReturnTypeInference TZAWARE_TIMESTAMP =
            new SqlReturnTypeInference() {
                @Override
                public @Nullable RelDataType inferReturnType(final SqlOperatorBinding opBinding) {
                    return BodoRelDataTypeFactory.createTZAwareSqlType(
                            opBinding.getTypeFactory(), null);
                }
            };

    public static final SqlReturnTypeInference TZAWARE_TIMESTAMP_NULLABLE =
            TZAWARE_TIMESTAMP.andThen(SqlTypeTransforms.TO_NULLABLE);

    public static final SqlReturnTypeInference UTC_TIMESTAMP =
            new SqlReturnTypeInference() {
                @Override
                public @Nullable RelDataType inferReturnType(final SqlOperatorBinding opBinding) {
                    return BodoRelDataTypeFactory.createTZAwareSqlType(
                            opBinding.getTypeFactory(), BodoTZInfo.UTC);
                }
            };

    public static final SqlReturnTypeInference VARIANT =
            new SqlReturnTypeInference() {
                @Override
                public @Nullable RelDataType inferReturnType(final SqlOperatorBinding opBinding) {
                    return BodoRelDataTypeFactory.createVariantSqlType(opBinding.getTypeFactory());
                }
            };

    public static final SqlReturnTypeInference VARIANT_NULLABLE =
            VARIANT.andThen(SqlTypeTransforms.FORCE_NULLABLE);

    /**
     * Type inference strategy the give the least restrictive between the start
     * index (inclusive) and end index (exclusive).
     */
    public static SqlReturnTypeInference leastRestrictiveSubset(int fromIndex, int toIndex) {
        return opBinding -> opBinding.getTypeFactory().leastRestrictive(
                opBinding.collectOperandTypes().subList(fromIndex, toIndex));
    }

    /**
     * Parameter type-inference transform strategy where a derived type is
     * transformed into the same type but nullable if any of a calls operands
     * between the start index (inclusive) and end index (exclusive) is
     * nullable.
     */
    public static final SqlTypeTransform toNullableSubset(int fromIndex, int toIndex) {
        return (opBinding, typeToTransform) ->
                SqlTypeUtil.makeNullableIfOperandsAre(opBinding.getTypeFactory(),
                        opBinding.collectOperandTypes().subList(fromIndex, toIndex),
                        requireNonNull(typeToTransform, "typeToTransform"));
    }

    /**
     * Type inference strategy that combines leastRestrictiveSubset and toNullableSubset.
     */
    public static SqlReturnTypeInference leastRestrictiveSubsetNullable(int fromIndex, int toIndex) {
        return leastRestrictiveSubset(fromIndex, toIndex).andThen(toNullableSubset(fromIndex, toIndex));
    }

    /**
     * Type Inference strategy for calculating the precision of INSERT.
     * This is largely influenced by ReturnTypes.DYADIC_STRING_SUM_PRECISION
     * because unfortunately there is no good way with which to interface.
     */
    private static final SqlReturnTypeInference INSERT_RETURN_PRECISION =
            opBinding -> {
                final RelDataType argType0 = opBinding.getOperandType(0);
                final RelDataType argType3 = opBinding.getOperandType(3);

                final boolean containsAnyType =
                        (argType0.getSqlTypeName() == SqlTypeName.ANY)
                                || (argType3.getSqlTypeName() == SqlTypeName.ANY);

                final boolean containsNullType =
                        (argType0.getSqlTypeName() == SqlTypeName.NULL)
                                || (argType3.getSqlTypeName() == SqlTypeName.NULL);
                SqlCollation pickedCollation = null;
                if (!containsAnyType
                        && !containsNullType
                        && SqlTypeUtil.inCharFamily(argType0)) {
                    if (!SqlTypeUtil.isCharTypeComparable(
                            List.of(argType0, argType3))) {
                        throw opBinding.newError(
                                RESOURCE.typeNotComparable(
                                        argType0.getFullTypeString(),
                                        argType3.getFullTypeString()));
                    }

                    pickedCollation = requireNonNull(
                            SqlCollation.getCoercibilityDyadicOperator(
                                    getCollation(argType0), getCollation(argType3)),
                            () -> "getCoercibilityDyadicOperator is null for " + argType0 + " and " + argType3);
                }

                // Determine whether result is variable-length
                SqlTypeName typeName =
                        argType0.getSqlTypeName();
                if (SqlTypeUtil.isBoundedVariableWidth(argType3)) {
                    typeName = argType3.getSqlTypeName();
                }

                RelDataType ret;
                int typePrecision;
                // Note Snowflake has the precision as (2 * arg0) + arg3.
                // Its unclear why this is done and may be a bug in Snowflake.
                final long x =
                        (2 * (long) argType0.getPrecision()) + (long) argType3.getPrecision();
                final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
                final RelDataTypeSystem typeSystem = typeFactory.getTypeSystem();
                if (argType0.getPrecision() == RelDataType.PRECISION_NOT_SPECIFIED
                        || argType3.getPrecision() == RelDataType.PRECISION_NOT_SPECIFIED
                        || x > typeSystem.getMaxPrecision(typeName)) {
                    typePrecision = RelDataType.PRECISION_NOT_SPECIFIED;
                } else {
                    typePrecision = (int) x;
                }

                ret = typeFactory.createSqlType(typeName, typePrecision);
                if (null != pickedCollation) {
                    RelDataType pickedType;
                    if (getCollation(argType0).equals(pickedCollation)) {
                        pickedType = argType0;
                    } else if (getCollation(argType3).equals(pickedCollation)) {
                        pickedType = argType3;
                    } else {
                        throw new AssertionError("should never come here, "
                                + "argType0=" + argType0 + ", argType3=" + argType3);
                    }
                    ret =
                            typeFactory.createTypeWithCharsetAndCollation(ret,
                                    getCharset(pickedType), getCollation(pickedType));
                }
                if (ret.getSqlTypeName() == SqlTypeName.NULL) {
                    ret = typeFactory.createTypeWithNullability(
                            typeFactory.createSqlType(SqlTypeName.VARCHAR), true);
                }
                return ret;
            };


    /**
     * Type Inference strategy for calculating the return type of INSERT.
     * This is largely copied from ReturnTypes.DYADIC_STRING_SUM_PRECISION
     * because unfortunately there is no good way with which to interface.
     */
    public static final SqlReturnTypeInference INSERT_RETURN_TYPE = INSERT_RETURN_PRECISION.andThen(SqlTypeTransforms.TO_NULLABLE).andThen(SqlTypeTransforms.TO_VARYING);

    /**
     * Transformation tha converts a SQL type to have an undefined precision. This is meant
     * for String data.
     */
    public static final SqlTypeTransform TO_UNDEFINED_PRECISION =
            (opBinding, typeToTransform) -> opBinding.getTypeFactory().createTypeWithNullability(opBinding.getTypeFactory().createSqlType(typeToTransform.getSqlTypeName(), RelDataType.PRECISION_NOT_SPECIFIED), typeToTransform.isNullable());

    /**
     * Type Inference strategy that
     */
    public static final SqlReturnTypeInference ARG0_NULLABLE_VARYING_UNDEFINED_PRECISION = ReturnTypes.ARG0_NULLABLE_VARYING.andThen(TO_UNDEFINED_PRECISION);
}
