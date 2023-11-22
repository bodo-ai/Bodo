package org.apache.calcite.sql.type;

import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem;
import com.google.common.base.Preconditions;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.runtime.CalciteContextException;
import org.apache.calcite.sql.SqlCollation;
import org.apache.calcite.sql.SqlOperatorBinding;

import com.bodosql.calcite.rel.type.BodoRelDataTypeFactory;

import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.ArrayList;
import java.util.List;

import static java.util.Objects.requireNonNull;
import static org.apache.calcite.sql.type.NonNullableAccessors.getCharset;
import static org.apache.calcite.sql.type.NonNullableAccessors.getCollation;
import static org.apache.calcite.sql.type.ReturnTypes.ARG0_NULLABLE;
import static org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE;
import static org.apache.calcite.util.Static.RESOURCE;

public class BodoReturnTypes {

    /**
     * Converts the input type to a map type with varchar keys.
     */
    public static final SqlTypeTransform TO_MAP =
            (opBinding, typeToTransform) -> {
                BodoRelDataTypeFactory typeFactory = (BodoRelDataTypeFactory) opBinding.getTypeFactory();
                return typeFactory.createMapType(
                        typeFactory.createSqlType(SqlTypeName.VARCHAR), typeToTransform);
            };



    /**
     * Converts the input type to a nullable type if there's a possibility
     * of an empty group. Based off of Calcite's ARG0_NULLABLE_IF_EMPTY.
     */
    public static final SqlTypeTransform FORCE_NULLABLE_IF_EMPTY_GROUP =
            (opBinding, typeToTransform) -> {
                if (opBinding.getGroupCount() == 0 || opBinding.hasFilter()) {
                    return opBinding.getTypeFactory()
                            .createTypeWithNullability(typeToTransform, true);
                } else {
                    return typeToTransform;
                }
            };

    public static final SqlReturnTypeInference BOOL_AGG_RET_TYPE = ReturnTypes.BOOLEAN_NULLABLE.andThen(FORCE_NULLABLE_IF_EMPTY_GROUP);

    /**
     * Defines the return type for FLATTEN
     */
    public static final SqlReturnTypeInference FLATTEN_RETURN_TYPE =
            (opBinding) -> {
                RelDataTypeFactory factory = opBinding.getTypeFactory();
                RelDataType inputType = opBinding.getOperandType(0);
                // Flatten returns a table with 6 values.
                // SEQ : BIGINT NOT NULL
                // KEY : NULLABLE VARCHAR
                // PATH : NULLABLE VARCHAR
                // INDEX: NULLABLE INTEGER
                // VALUE: INPUT.DTYPE  (NULLABLE)
                // THIS: INPUT_TYPE
                RelDataType type0 = factory.createSqlType(SqlTypeName.BIGINT);
                RelDataType type1 = factory.createTypeWithNullability(factory.createSqlType(SqlTypeName.VARCHAR), true);
                RelDataType type2 = type1;
                RelDataType type3 = factory.createTypeWithNullability(factory.createSqlType(SqlTypeName.INTEGER), true);
                RelDataType type4;
                if (inputType instanceof MapSqlType) {
                    type4 = factory.createTypeWithNullability(inputType.getValueType(), true);
                } else if (inputType instanceof ArraySqlType) {
                    type4 = factory.createTypeWithNullability(inputType.getComponentType(), true);
                } else {
                    throw new CalciteContextException("", BODO_SQL_RESOURCE.requiresArrayOrJson("FLATTEN", "INPUT").ex());
                }
                RelDataType type5 = inputType;
                List<RelDataType> types = List.of(type0, type1, type2, type3, type4, type5);
                List<String> names = List.of("SEQ", "KEY", "PATH", "INDEX", "VALUE", "THIS");
                return factory.createStructType(types, names);
            };

    /**
     * Determine the return type of the TO_ARRAY function
     * @param binding The operand bindings for the function signature.
     * @return the return type of the function
     */
    public static RelDataType toArrayReturnType(SqlOperatorBinding binding) {
        return toArrayTypeIfNotAlready(binding, binding.getOperandType(0), true);
    }

    /**
     * Wraps the input type as an array type, if it isn't already.
     *
     * @param binding The operand bindings for the function signature.
     * @param inputType the type to wrap as an array.
     * @param innerNullabilityToOuter should the nullability of the input type propagate to the outer type
     *  IE if innerNullabilityToOuter is true, NULLABLE INT ---> NULLABLE ARRAY(int)
     *     if innerNullabilityToOuter is false, NULLABLE INT ---> ARRAY(NULLABLE int)
     *
     * @return The return type of the function
     */
    public static RelDataType toArrayTypeIfNotAlready(SqlOperatorBinding binding,
                                                      RelDataType inputType, boolean innerNullabilityToOuter) {
        RelDataTypeFactory typeFactory = binding.getTypeFactory();
        if (inputType.getSqlTypeName().equals(SqlTypeName.NULL))
            // if the input is null, TO_ARRAY will return NULL, not an array of NULL
            return inputType;
        if (inputType instanceof ArraySqlType)
            // if the input is an array, just return it
            return inputType;

        if (innerNullabilityToOuter) {
            RelDataType arrayType =
                    typeFactory.createArrayType(
                            typeFactory.createTypeWithNullability(inputType, false), -1);
            return typeFactory.createTypeWithNullability(arrayType, inputType.isNullable());
        } else {
            return typeFactory.createArrayType(inputType, -1);
        }
    }

    /**
     * Convert type XXX to type XXX ARRAY
     */
    public static final SqlTypeTransform WRAP_TYPE_TO_ARRAY =
            (opBinding, typeToTransform) ->
                    toArrayTypeIfNotAlready(opBinding, typeToTransform, false)
                    ;



    public static final SqlReturnTypeInference DECODE_RETURN_TYPE = opBinding -> decodeReturnType(opBinding);


    // Obtains the least restrictive union of all the argument types
    // corresponding to outputs in the key-value pairs of arguments
    // (plus the optional default value argument)
    public static RelDataType decodeReturnType(SqlOperatorBinding binding) {
        RelDataTypeFactory typeFactory = binding.getTypeFactory();
        RelDataType leastRestrictiveType = typeFactory.leastRestrictive(collectOutputTypes(binding));

        // If there is no default, force the output to be nullable, since
        // the default output is null
        if (binding.getOperandCount() % 2 == 0) {
            leastRestrictiveType = typeFactory.createTypeWithNullability(leastRestrictiveType, true);
        }
        return leastRestrictiveType;
    }

    /**
     * Takes in the arguments to a DECODE call and extracts a subset of the argument types that
     * correspond to outputs. For example:
     *
     * <p>DECODE(A, B, C, D, E) --> [C, E]
     *
     * <p>DECODE(A, B, C, D, E, F) --> [C, E, F]
     *
     * @param binding a container for all the operands of the DECODE function call
     * @return a list of all the output types corresponding to output arguments of DECODE
     */
    public static List<RelDataType> collectOutputTypes(SqlOperatorBinding binding) {
        List<RelDataType> operandTypes = binding.collectOperandTypes();
        List<RelDataType> outputTypes = new ArrayList<RelDataType>();
        int count = binding.getOperandCount();
        for (int i = 2; i < count; i++) {
            if (i % 2 == 0) {
                outputTypes.add(operandTypes.get(i));
            }
        }
        if (count > 3 && count % 2 == 0) {
            outputTypes.add(operandTypes.get(count - 1));
        }
        return outputTypes;
    }

    /** Nulls are dropped by arrayAgg, so return a non-null array of the input type. */
    public static RelDataType ArrayAggReturnType(SqlOperatorBinding binding) {
        RelDataTypeFactory typeFactory = binding.getTypeFactory();
        RelDataType inputType = binding.collectOperandTypes().get(0);
        return typeFactory.createArrayType(typeFactory.createTypeWithNullability(inputType, false), -1);
    }

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
     * Type-inference strategy whereby the output type is a VARCHAR with an unknown
     * precision and SqlTypeTransforms.FORCE_NULLABLE.
     */
    public static final SqlReturnTypeInference VARCHAR_UNKNOWN_PRECISION_FORCE_NULLABLE = VARCHAR_UNKNOWN_PRECISION.andThen(SqlTypeTransforms.FORCE_NULLABLE);

    /**
     * Type-inference strategy whereby the result type of a call is an integer
     * if both operands are date.
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
                            opBinding.getTypeFactory(), null, BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION);
                }
            };

    public static final SqlReturnTypeInference TZAWARE_TIMESTAMP_NULLABLE =
            TZAWARE_TIMESTAMP.andThen(SqlTypeTransforms.TO_NULLABLE);

    public static final SqlReturnTypeInference VARIANT =
            new SqlReturnTypeInference() {
                @Override
                public @Nullable RelDataType inferReturnType(final SqlOperatorBinding opBinding) {
                    return BodoRelDataTypeFactory.createVariantSqlType(opBinding.getTypeFactory());
                }
            };

    public static final SqlReturnTypeInference VARIANT_NULLABLE =
            VARIANT.andThen(SqlTypeTransforms.TO_NULLABLE);

    public static final SqlReturnTypeInference VARIANT_FORCE_NULLABLE =
            VARIANT.andThen(SqlTypeTransforms.FORCE_NULLABLE);

    public static final SqlReturnTypeInference MAP_VARIANT =
            new SqlReturnTypeInference() {
                @Override
                public @Nullable RelDataType inferReturnType(final SqlOperatorBinding opBinding) {
                    BodoRelDataTypeFactory typeFactory = (BodoRelDataTypeFactory) opBinding.getTypeFactory();
                    RelDataType valueType = typeFactory.createVariantSqlType();
                    return typeFactory.createMapType(typeFactory.createSqlType(SqlTypeName.VARCHAR), valueType);
                }
            };

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
     * Helper function to compute the return type from summing together
     * the precision of any two inputs. This is used as a helper function
     * implement functions like INSERT, CONCAT, and CONCAT_WS that contain custom
     * precision logic.
     *
     * opBinding is passed in for utilities like error reporting, but it should not be
     * used to load operands.
     */
    private static RelDataType computeSumPrecisionType(final RelDataType argType0, final RelDataType argType1, final SqlOperatorBinding opBinding) {
        final boolean containsAnyType =
                (argType0.getSqlTypeName() == SqlTypeName.ANY)
                        || (argType1.getSqlTypeName() == SqlTypeName.ANY);

        final boolean containsNullType =
                (argType0.getSqlTypeName() == SqlTypeName.NULL)
                        || (argType1.getSqlTypeName() == SqlTypeName.NULL);

        if (!containsAnyType
                && !containsNullType
                && !(SqlTypeUtil.inCharOrBinaryFamilies(argType0)
                && SqlTypeUtil.inCharOrBinaryFamilies(argType1))) {
            Preconditions.checkArgument(
                    SqlTypeUtil.sameNamedType(argType0, argType1));
        }

        SqlCollation pickedCollation = null;
        if (!containsAnyType
                && !containsNullType
                && SqlTypeUtil.inCharFamily(argType0)) {
            if (!SqlTypeUtil.isCharTypeComparable(
                    List.of(argType0, argType1))) {
                throw opBinding.newError(
                        RESOURCE.typeNotComparable(
                                argType0.getFullTypeString(),
                                argType1.getFullTypeString()));
            }

            pickedCollation = requireNonNull(
                    SqlCollation.getCoercibilityDyadicOperator(
                            getCollation(argType0), getCollation(argType1)),
                    () -> "getCoercibilityDyadicOperator is null for " + argType0 + " and " + argType1);
        }

        // Determine whether result is variable-length
        SqlTypeName typeName =
                argType0.getSqlTypeName();
        if (SqlTypeUtil.isBoundedVariableWidth(argType1)) {
            typeName = argType1.getSqlTypeName();
        }

        RelDataType ret;
        int typePrecision;
        final long x = (long) argType0.getPrecision() + (long) argType1.getPrecision();
        final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
        final RelDataTypeSystem typeSystem = typeFactory.getTypeSystem();
        if (argType0.getPrecision() == RelDataType.PRECISION_NOT_SPECIFIED
                || argType1.getPrecision() == RelDataType.PRECISION_NOT_SPECIFIED
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
            } else if (getCollation(argType1).equals(pickedCollation)) {
                pickedType = argType1;
            } else {
                throw new AssertionError("should never come here, "
                        + "argType0=" + argType0 + ", argType1=" + argType1);
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
    }

    /**
     * Type Inference strategy for calculating the precision of INSERT.
     */
    private static final SqlReturnTypeInference INSERT_RETURN_PRECISION =
            // Note: Snowflake does 2 * arg0 + arg3, but its unclear why.
            opBinding -> computeSumPrecisionType(opBinding.getOperandType(0), computeSumPrecisionType(opBinding.getOperandType(0), opBinding.getOperandType(3), opBinding), opBinding);


    /**
     * Type Inference strategy for calculating the return type of INSERT.
     */
    public static final SqlReturnTypeInference INSERT_RETURN_TYPE = INSERT_RETURN_PRECISION.andThen(SqlTypeTransforms.TO_NULLABLE).andThen(SqlTypeTransforms.TO_VARYING);

    /**
     * Type Inference strategy for calculating the precision of for CONCAT.
     */
    private static final SqlReturnTypeInference CONCAT_RETURN_PRECISION =
            opBinding -> {
                RelDataType returnType = opBinding.getOperandType(0);
                for (int i = 1; i < opBinding.getOperandCount(); i++) {
                    returnType = computeSumPrecisionType(opBinding.getOperandType(i), returnType, opBinding);
                }
                return returnType;
            };

    /**
     * Type Inference strategy for calculating the return type of CONCAT.
     */
    public static final SqlReturnTypeInference CONCAT_RETURN_TYPE = CONCAT_RETURN_PRECISION.andThen(SqlTypeTransforms.TO_NULLABLE);

    /**
     * Type Inference strategy for calculating the precision of for CONCAT.
     */
    private static final SqlReturnTypeInference CONCAT_WS_RETURN_PRECISION =
            opBinding -> {
                // Sum together the inputs being concatenated.
                // Note: Arg 0 is the separator.
                RelDataType returnType = opBinding.getOperandType(1);
                for (int i = 2; i < opBinding.getOperandCount(); i++) {
                    // Compute the result of adding the separator
                    returnType = computeSumPrecisionType(opBinding.getOperandType(0), returnType, opBinding);
                    // Add the second value.
                    returnType = computeSumPrecisionType(opBinding.getOperandType(i), returnType, opBinding);
                }
                return returnType;
            };

    /**
     * Type Inference strategy for calculating the return type of CONCAT.
     */
    public static final SqlReturnTypeInference CONCAT_WS_RETURN_TYPE = CONCAT_WS_RETURN_PRECISION.andThen(SqlTypeTransforms.TO_NULLABLE);

    /**
     * Return type is essentially the same as TO_ARRAY, but we convert the array element type to varying,
     * since split can create variable sized strings on each row.
     */
    public static final SqlReturnTypeInference SPLIT_RETURN_TYPE = ARG0_NULLABLE.andThen(SqlTypeTransforms.TO_VARYING).andThen(
            (opBinding, typeToTransform) -> {
                return toArrayTypeIfNotAlready(opBinding, typeToTransform, true);
            }
    );

    /**
     * Transformation tha converts a SQL type to have an undefined precision. This is meant
     * for String data.
     */
    public static final SqlTypeTransform TO_UNDEFINED_PRECISION =
            (opBinding, typeToTransform) -> opBinding.getTypeFactory().createTypeWithNullability(opBinding.getTypeFactory().createSqlType(typeToTransform.getSqlTypeName(), RelDataType.PRECISION_NOT_SPECIFIED), typeToTransform.isNullable());


    public static final SqlReturnTypeInference ARG0_FORCE_NULLABLE_VARYING = ReturnTypes.ARG0_FORCE_NULLABLE.andThen(SqlTypeTransforms.TO_VARYING);

    /**
     * Type Inference strategy that
     */
    public static final SqlReturnTypeInference ARG0_NULLABLE_VARYING_UNDEFINED_PRECISION = ReturnTypes.ARG0_NULLABLE_VARYING.andThen(TO_UNDEFINED_PRECISION);

    public static final SqlReturnTypeInference SMALLINT = ReturnTypes.explicit(SqlTypeName.SMALLINT);

    public static final SqlReturnTypeInference SMALLINT_NULLABLE = SMALLINT.andThen(SqlTypeTransforms.TO_NULLABLE);

    public static final SqlReturnTypeInference VARBINARY_NULLABLE = ReturnTypes.explicit(SqlTypeName.VARBINARY, RelDataType.PRECISION_NOT_SPECIFIED).andThen(SqlTypeTransforms.TO_NULLABLE);

    public static final SqlReturnTypeInference VARBINARY_FORCE_NULLABLE= ReturnTypes.explicit(SqlTypeName.VARBINARY).andThen(SqlTypeTransforms.FORCE_NULLABLE);

    public static final SqlReturnTypeInference VARBINARY_NULLABLE_UNKNOWN_PRECISION = VARBINARY_NULLABLE.andThen(TO_UNDEFINED_PRECISION);

    public static final SqlReturnTypeInference VARBINARY_FORCE_NULLABLE_UNKNOWN_PRECISION = VARBINARY_FORCE_NULLABLE.andThen(TO_UNDEFINED_PRECISION);


    public static final SqlReturnTypeInference DATE_FORCE_NULLABLE = ReturnTypes.DATE.andThen(SqlTypeTransforms.FORCE_NULLABLE);

    public static final SqlReturnTypeInference TIMESTAMP_FORCE_NULLABLE = ReturnTypes.TIMESTAMP.andThen(SqlTypeTransforms.FORCE_NULLABLE);

    public static final SqlReturnTypeInference DOUBLE_FORCE_NULLABLE = ReturnTypes.DOUBLE.andThen(SqlTypeTransforms.FORCE_NULLABLE);

    public static final SqlReturnTypeInference INTEGER_FORCE_NULLABLE = ReturnTypes.INTEGER.andThen(SqlTypeTransforms.FORCE_NULLABLE);

    public static final SqlReturnTypeInference TIME_FORCE_NULLABLE = ReturnTypes.TIME.andThen(SqlTypeTransforms.FORCE_NULLABLE);

    public static final SqlReturnTypeInference TO_NULLABLE_VARYING_ARRAY = ReturnTypes.ARG0_NULLABLE_VARYING.andThen(SqlTypeTransforms.TO_ARRAY);

    public static final SqlTypeTransform TO_NULLABLE_ARG1 = (opBinding, typeToTransform) -> opBinding.getTypeFactory().createTypeWithNullability(typeToTransform, SqlTypeUtil.containsNullable(opBinding.getOperandType(1)));

    /**
     * Determine the return type for BITOR_AGG, BITAND_AGG, and BITXOR_AGG. The return type is the
     * same as the input if it is an integer, otherwise the return type is always int64 (BIGINT).
     *
     * @param binding The operand bindings for the function signature.
     * @return The return type of the function
     */
    public static RelDataType bitX_ret_type(SqlOperatorBinding binding) {
        RelDataType arg0Type = binding.getOperandType(0);
        SqlTypeFamily arg0TypeFamily = arg0Type.getSqlTypeName().getFamily();
        if (arg0TypeFamily.equals(SqlTypeFamily.INTEGER)) {
            return ReturnTypes.ARG0_NULLABLE_IF_EMPTY.inferReturnType(binding);
        } else {
            return ReturnTypes.BIGINT_NULLABLE.andThen(BodoReturnTypes.FORCE_NULLABLE_IF_EMPTY_GROUP).inferReturnType(binding);
        }
    }
}
