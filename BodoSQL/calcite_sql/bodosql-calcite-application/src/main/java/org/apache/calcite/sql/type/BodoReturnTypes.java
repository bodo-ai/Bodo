package org.apache.calcite.sql.type;

import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem;
import com.bodosql.calcite.application.operatorTables.DatetimeFnUtils;
import com.bodosql.calcite.rel.type.BodoTypeFactoryImpl;
import com.google.common.collect.Sets;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.runtime.CalciteContextException;
import org.apache.calcite.sql.SqlCollation;
import org.apache.calcite.sql.SqlOperatorBinding;

import com.bodosql.calcite.rel.type.BodoRelDataTypeFactory;

import org.checkerframework.checker.nullness.qual.Nullable;
import static com.google.common.base.Preconditions.checkArgument;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import static com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem.getMinIntegerSize;
import static java.util.Objects.requireNonNull;
import static org.apache.calcite.sql.type.NonNullableAccessors.getCharset;
import static org.apache.calcite.sql.type.NonNullableAccessors.getCollation;
import static org.apache.calcite.sql.type.NonNullableAccessors.getComponentTypeOrThrow;
import static org.apache.calcite.sql.type.ReturnTypes.ARG0_NULLABLE;
import static org.apache.calcite.sql.type.ReturnTypes.explicit;
import static org.apache.calcite.sql.validate.SqlNonNullableAccessors.getOperandLiteralValueOrThrow;
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

    public static final SqlReturnTypeInference CONVERT_TIMEZONE_RETURN_TYPE = (opBinding) -> {
        RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
        if (opBinding.getOperandCount() == 2) {
            return typeFactory.createSqlType(SqlTypeName.TIMESTAMP_TZ, opBinding.getOperandType(1).getPrecision());
        } else {
            return typeFactory.createSqlType(SqlTypeName.TIMESTAMP, opBinding.getOperandType(2).getPrecision());
        }
    };

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
                } else if (inputType instanceof VariantSqlType) {
                    if (!(factory instanceof BodoRelDataTypeFactory)) {
                        throw new RuntimeException("Internal Error: Unexpected RelDataTypeFactory");
                    }
                    type4 = ((BodoRelDataTypeFactory) factory).createVariantSqlType();
                } else {
                    throw new CalciteContextException("", BODO_SQL_RESOURCE.requiresArrayOrJson("FLATTEN", "INPUT").ex());
                }
                // TODO: The value should but nullable if OUTER=TRUE and otherwise not nullable
                RelDataType type5 = inputType;
                List<RelDataType> types = List.of(type0, type1, type2, type3, type4, type5);
                List<String> names = List.of("SEQ", "KEY", "PATH", "INDEX", "VALUE", "THIS");
                return factory.createStructType(types, names);
            };

    public static final SqlReturnTypeInference PARSE_URL_RETURN_TYPE =
            (opBinding) -> {


                RelDataTypeFactory factory = opBinding.getTypeFactory();
                RelDataType stringType = factory.createSqlType(SqlTypeName.VARCHAR);
                RelDataType mapType = factory.createMapType(stringType, stringType);
                return mapType;
                // TODO: Replace the above implementation with the below implementation when
                // we have proper struct support in Calcite.
//                return factory.createStructType(
//                        List.of(stringType, stringType, mapType, stringType, stringType, stringType, stringType),
//                        List.of("fragment", "host", "parameters",                                       "path", "port", "query", "scheme"));
            };

    /**
     * Defines the return type for GENERATOR
     */
    public static final SqlReturnTypeInference GENERATOR_RETURN_TYPE =
            (opBinding) -> {
                RelDataTypeFactory factory = opBinding.getTypeFactory();
                return factory.createStructType(List.of(), List.of());
            };


    /**
     * Defines the return type for EXTERNAL_TABLE_FILES
     */
    public static final SqlReturnTypeInference EXTERNAL_TABLE_FILES_RETURN_TYPE =
            (opBinding) -> {
                RelDataTypeFactory factory = opBinding.getTypeFactory();
                // EXTERNAL_TABLE_FILES returns a table with 6 values.
                // FILE_NAME : VARCHAR NOT NULL
                // REGISTERED_ON : TIMESTAMP_LTZ NOT NULL
                // FILE_SIZE : BIGINT NOT NULL
                // LAST_MODIFIED: TIMESTAMP_LTZ NOT NULL
                // ETAG: VARCHAR NOT NULL
                // MD5: VARCHAR NOT NULL
                RelDataType type0 = factory.createTypeWithNullability(factory.createSqlType(SqlTypeName.VARCHAR), false);
                RelDataType type1 = factory.createTypeWithNullability(factory.createSqlType(SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE), false);
                RelDataType type2 = factory.createTypeWithNullability(factory.createSqlType(SqlTypeName.BIGINT), false);
                RelDataType type3 = type1;
                RelDataType type4 = type0;
                RelDataType type5 = type0;
                List<RelDataType> types = List.of(type0, type1, type2, type3, type4, type5);
                List<String> names = List.of("FILE_NAME", "REGISTERED_ON", "FILE_SIZE", "LAST_MODIFIED", "ETAG", "MD5");
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
                            opBinding.getTypeFactory().createSqlType(typeToTransform.getSqlTypeName(), BodoSQLRelDataTypeSystem.MAX_STRING_PRECISION),
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
    public static final SqlReturnTypeInference VARCHAR_1 = explicit(SqlTypeName.VARCHAR, 1);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 1
     * and SqlTypeTransforms.TO_NULLABLE.
     */
    public static final SqlReturnTypeInference VARCHAR_1_NULLABLE = VARCHAR_1.andThen(SqlTypeTransforms.TO_NULLABLE);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 3
     */
    public static final SqlReturnTypeInference VARCHAR_3 = explicit(SqlTypeName.VARCHAR, 3);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 3
     * and SqlTypeTransforms.TO_NULLABLE.
     */
    public static final SqlReturnTypeInference VARCHAR_3_NULLABLE = VARCHAR_3.andThen(SqlTypeTransforms.TO_NULLABLE);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 32
     */
    public static final SqlReturnTypeInference VARCHAR_32 = explicit(SqlTypeName.VARCHAR, 32);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 32
     * and SqlTypeTransforms.TO_NULLABLE.
     */
    public static final SqlReturnTypeInference VARCHAR_32_NULLABLE = VARCHAR_32.andThen(SqlTypeTransforms.TO_NULLABLE);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 128
     */
    public static final SqlReturnTypeInference VARCHAR_128 = explicit(SqlTypeName.VARCHAR, 128);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 36 (e.g. UUIDs)
     */
    public static final SqlReturnTypeInference VARCHAR_36 = explicit(SqlTypeName.VARCHAR, 36);

    /**
     * Type-inference strategy for returning a VARCHAR with precision 128
     * and SqlTypeTransforms.TO_NULLABLE.
     */
    public static final SqlReturnTypeInference VARCHAR_128_NULLABLE = VARCHAR_128.andThen(SqlTypeTransforms.TO_NULLABLE);

    /**
     * Type-inference strategy for returning a VARCHAR with an unknown precision
     */
    public static final SqlReturnTypeInference VARCHAR_UNKNOWN_PRECISION = explicit(SqlTypeName.VARCHAR, BodoSQLRelDataTypeSystem.MAX_STRING_PRECISION);

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
     * Type-inference strategy whereby either result type of call is an INTEGER
     * if both operands are DATE, or if the left argument is a DATE and the right argument is NUMERIC, then the result is a DATE.
     */
    public static final SqlReturnTypeInference DATE_SUB = opBinding -> {
        RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
        RelDataType type1 = opBinding.getOperandType(0);
        RelDataType type2 = opBinding.getOperandType(1);
        if (type1.getSqlTypeName() == SqlTypeName.DATE
                && type2.getSqlTypeName() == SqlTypeName.DATE) {
            return typeFactory.createSqlType(SqlTypeName.INTEGER);
        }

        // Accept DATE - NUMERIC as a DATE
        if (SqlTypeFamily.DATE.contains(type1) && SqlTypeFamily.NUMERIC.contains(type2)) {
            return type1;
        } else if (SqlTypeFamily.DATE.contains(type2) && SqlTypeFamily.NUMERIC.contains(type1)) {
            return type2;
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

    /**
     * Type-inference for sums involving DATEs and NUMERICs
     */
    public static final SqlReturnTypeInference DATE_SUM = opBinding -> {
        RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
        RelDataType type1 = opBinding.getOperandType(0);
        RelDataType type2 = opBinding.getOperandType(1);

        // Accept DATE + NUMERIC as a DATE
        if (SqlTypeFamily.DATE.contains(type1) && SqlTypeFamily.NUMERIC.contains(type2)) {
            return type1;
        } else if (SqlTypeFamily.DATE.contains(type2) && SqlTypeFamily.NUMERIC.contains(type1)) {
            return type2;
        }
        // This rule doesn't apply
        return null;
    };

    public static final SqlReturnTypeInference DATE_SUM_NULLABLE =
            DATE_SUM.andThen(SqlTypeTransforms.TO_NULLABLE);
    public static final SqlReturnTypeInference NULLABLE_SUM =
            new SqlReturnTypeInferenceChain(DATE_SUM_NULLABLE, ReturnTypes.NULLABLE_SUM);


    public static final SqlReturnTypeInference VARIANT =
            new SqlReturnTypeInference() {
                @Override
                public @Nullable RelDataType inferReturnType(final SqlOperatorBinding opBinding) {
                    return BodoRelDataTypeFactory.createVariantSqlType(opBinding.getTypeFactory());
                }
            };


    public static final SqlReturnTypeInference TO_OBJECT_RETURN_TYPE_INFERENCE =
            new SqlReturnTypeInference() {

                /**
                 * From SF docs:

                 For a VARIANT value containing an OBJECT, returns the OBJECT.
                 For NULL input, or for a VARIANT value containing only JSON null, returns NULL.
                 For an OBJECT, returns the OBJECT itself.
                 For all other input values, reports an error.
                 */
                @Override
                public @Nullable RelDataType inferReturnType(SqlOperatorBinding opBinding) {
                    RelDataType inputType = opBinding.getOperandType(0);
                    SqlTypeFamily inputTypeFamily = inputType.getSqlTypeName().getFamily();

                    boolean isMap = (inputTypeFamily != null && inputTypeFamily.equals(SqlTypeFamily.MAP));
                    if (inputType.isStruct() || isMap){
                        return inputType;
                    }
                    else if (inputType instanceof VariantSqlType) {
                        // If we have JSON input, we currently map this to MAP(String, String) in Calcite.
                        // The key should always be string, but we don't know the types here until runtime,
                        // so we return MAP(STRING, VARIANT)
                        RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
                        if (!(typeFactory instanceof BodoTypeFactoryImpl)){
                            throw new RuntimeException("Internal error in TO_OBJECT_RETURN_TYPE_INFERENCE.inferReturnType: Type factory is not a bodo type factory.");
                        }
                        BodoTypeFactoryImpl bodoTypeFactory = (BodoTypeFactoryImpl) typeFactory;
                        return opBinding.getTypeFactory().createMapType(bodoTypeFactory.createSqlType(SqlTypeName.VARCHAR), bodoTypeFactory.createVariantSqlType());
                    }

                    return null;
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
            checkArgument(SqlTypeUtil.sameNamedType(argType0, argType1));
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
            typePrecision = typeSystem.getMaxPrecision(typeName);
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
                    typeFactory.createSqlType(SqlTypeName.VARCHAR, typePrecision), true);
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
            (opBinding, typeToTransform) -> toArrayTypeIfNotAlready(opBinding, typeToTransform, true)
    );

    /**
     * Transformation tha converts a SQL type to have an undefined precision. This is meant
     * for String data.
     */
    public static final SqlTypeTransform TO_UNDEFINED_PRECISION =
            (opBinding, typeToTransform) -> opBinding.getTypeFactory().createTypeWithNullability(opBinding.getTypeFactory().createSqlType(typeToTransform.getSqlTypeName(), BodoSQLRelDataTypeSystem.MAX_STRING_PRECISION), typeToTransform.isNullable());


    public static final SqlReturnTypeInference ARG0_FORCE_NULLABLE_VARYING = ReturnTypes.ARG0_FORCE_NULLABLE.andThen(SqlTypeTransforms.TO_VARYING);

    /**
     * Type Inference strategy that
     */
    public static final SqlReturnTypeInference ARG0_NULLABLE_VARYING_UNDEFINED_PRECISION = ReturnTypes.ARG0_NULLABLE_VARYING.andThen(TO_UNDEFINED_PRECISION);

    public static final SqlReturnTypeInference SMALLINT = explicit(SqlTypeName.SMALLINT);

    public static final SqlReturnTypeInference SMALLINT_NULLABLE = SMALLINT.andThen(SqlTypeTransforms.TO_NULLABLE);

    public static final SqlReturnTypeInference VARBINARY_NULLABLE = explicit(SqlTypeName.VARBINARY, BodoSQLRelDataTypeSystem.MAX_BINARY_PRECISION).andThen(SqlTypeTransforms.TO_NULLABLE);

    public static final SqlReturnTypeInference VARBINARY_FORCE_NULLABLE= explicit(SqlTypeName.VARBINARY).andThen(SqlTypeTransforms.FORCE_NULLABLE);

    public static final SqlReturnTypeInference VARBINARY_NULLABLE_UNKNOWN_PRECISION = VARBINARY_NULLABLE.andThen(TO_UNDEFINED_PRECISION);

    public static final SqlReturnTypeInference VARBINARY_FORCE_NULLABLE_UNKNOWN_PRECISION = VARBINARY_FORCE_NULLABLE.andThen(TO_UNDEFINED_PRECISION);


    public static final SqlReturnTypeInference DATE_FORCE_NULLABLE = ReturnTypes.DATE.andThen(SqlTypeTransforms.FORCE_NULLABLE);

    public static final SqlReturnTypeInference TIMESTAMP_FORCE_NULLABLE = ReturnTypes.TIMESTAMP.andThen(SqlTypeTransforms.FORCE_NULLABLE);

    public static final SqlReturnTypeInference DOUBLE_FORCE_NULLABLE = ReturnTypes.DOUBLE.andThen(SqlTypeTransforms.FORCE_NULLABLE);

    public static final SqlReturnTypeInference INTEGER_FORCE_NULLABLE = ReturnTypes.INTEGER.andThen(SqlTypeTransforms.FORCE_NULLABLE);

    public static final SqlReturnTypeInference TIME_DEFAULT_PRECISION = explicit(SqlTypeName.TIME, BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION);

    public static final SqlReturnTypeInference TIME_DEFAULT_PRECISION_NULLABLE = TIME_DEFAULT_PRECISION.andThen(SqlTypeTransforms.TO_NULLABLE);

    public static final SqlReturnTypeInference TIME_DEFAULT_PRECISION_FORCE_NULLABLE = TIME_DEFAULT_PRECISION.andThen(SqlTypeTransforms.FORCE_NULLABLE);


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

    /**
     * Determine the base return type for all TO_TIMESTAMP functions, including
     * the TZ_AWARE and TZ_NAIVE.
     */
    public static SqlReturnTypeInference toTimestampReturnType(String fnName, boolean isTzAware, boolean forceNullable) {
        SqlReturnTypeInference retType = opBinding -> {
            // Determine the return precision. If we have a timestamp input of any type
            // then we reuse that precision. Otherwise, if we have a two argument input with a string
            // input then the integer literal should specify the precision. Finally, if none of these apply
            // the default precision is 9;
            // TODO: Support inferring scale from Decimal inputs
            RelDataType arg0Type = opBinding.getOperandType(0);
            int precision = BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION;
            if (SqlTypeFamily.TIMESTAMP.contains(arg0Type)) {
                precision = arg0Type.getPrecision();
            } else if (SqlTypeFamily.INTEGER.contains(arg0Type)) {
                if (opBinding.getOperandCount() == 2) {
                    // Try load the scale as a literal.
                    RelDataType arg1Type = opBinding.getOperandType(1);
                    // Add a defensive check
                    if (SqlTypeFamily.INTEGER.contains(arg1Type)) {
                        precision = opBinding.getOperandLiteralValue(1, Integer.class);
                        if (precision > BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION || precision < 0) {
                            throw new RuntimeException(String.format(Locale.ROOT, "%s() requires a scale between 0 and %d if provided. Found %d.", fnName, BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION, precision));
                        }
                    }
                }
            }
            // Create the return type.
            if (isTzAware) {
                return opBinding.getTypeFactory().createSqlType(SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE, precision);
            } else {
                return opBinding.getTypeFactory().createSqlType(SqlTypeName.TIMESTAMP, precision);
            }
        };
        // Force Nullable if necessary. Otherwise, use TO_NULLABLE.
        if (forceNullable) {
            return retType.andThen(SqlTypeTransforms.FORCE_NULLABLE);
        } else {
            return retType.andThen(SqlTypeTransforms.TO_NULLABLE);
        }
    }

    /**
     * Determine the return type of the Snowflake DATEADD function
     * without considering nullability.
     *
     * @param binding The operand bindings for the function signature.
     * @return The return type of the function
     */
    private static RelDataType snowflakeDateaddReturnType(SqlOperatorBinding binding, String fnName) {
        List<RelDataType> operandTypes = binding.collectOperandTypes();
        RelDataType datetimeType = operandTypes.get(2);
        RelDataType typeArg0 = operandTypes.get(0);
        assert typeArg0.getSqlTypeName().equals(SqlTypeName.SYMBOL): "Internal error in snowflakeDateaddReturnType: arg0 is not symbol";
        DatetimeFnUtils.DateTimePart datePartEnum = binding.getOperandLiteralValue(0, DatetimeFnUtils.DateTimePart.class);
        assert (datePartEnum != null): "Internal error in snowflakeDateaddReturnType: Cannot interpret arg0 as the correct enum";

        // TODO: refactor standardizeTimeUnit function to change the third argument to
        // SqlTypeName
        final RelDataType returnType;
        // The output always needs to be cast to precision 9
        final int precision = BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION;
        final RelDataTypeFactory typeFactory = binding.getTypeFactory();
        if (datetimeType.getSqlTypeName().equals(SqlTypeName.DATE)) {
            Set<DatetimeFnUtils.DateTimePart> DATE_UNITS =
                    new HashSet<>(Arrays.asList(
                            DatetimeFnUtils.DateTimePart.YEAR,
                            DatetimeFnUtils.DateTimePart.QUARTER,
                            DatetimeFnUtils.DateTimePart.MONTH,
                            DatetimeFnUtils.DateTimePart.WEEK,
                            DatetimeFnUtils.DateTimePart.DAY));
            if (DATE_UNITS.contains(datePartEnum)) {
                returnType = typeFactory.createSqlType(SqlTypeName.DATE);
            } else {
                // Date always gets upcast to precision 9, even if we only add 1 hour.
                returnType = typeFactory.createSqlType(SqlTypeName.TIMESTAMP, precision);
            }
        } else {
            // The output always needs to be cast to precision 9, so we cannot just return
            // the same return type.
            returnType = typeFactory.createSqlType(datetimeType.getSqlTypeName(), precision);
        }
        return returnType;
    }

    /**
     * Determine the return type of the MySQL DATEADD, DATE_ADD, ADDDATE, DATE_SUB, SUBDATE function
     * without considering nullability.
     *
     * @param binding The operand bindings for the function signature.
     * @return The return type of the function
     */
    private static RelDataType mySqlDateaddReturnType(SqlOperatorBinding binding) {
        List<RelDataType> operandTypes = binding.collectOperandTypes();
        // Determine if the output is nullable.
        RelDataType datetimeType = operandTypes.get(0);
        final RelDataType returnType;

        if (SqlTypeFamily.INTEGER.contains(operandTypes.get(1))) {
            // when the second argument is integer, it is equivalent to adding day interval
            if (datetimeType.getSqlTypeName().equals(SqlTypeName.DATE)) {
                returnType = binding.getTypeFactory().createSqlType(SqlTypeName.DATE);
            } else {
                returnType = datetimeType;
            }
        } else {
            // if the first argument is date, the return type depends on the interval type
            if (datetimeType.getSqlTypeName().equals(SqlTypeName.DATE)) {
                Set<SqlTypeName> DATE_INTERVAL_TYPES =
                        Sets.immutableEnumSet(
                                SqlTypeName.INTERVAL_YEAR_MONTH,
                                SqlTypeName.INTERVAL_YEAR,
                                SqlTypeName.INTERVAL_MONTH,
                                SqlTypeName.INTERVAL_DAY);
                if (DATE_INTERVAL_TYPES.contains(operandTypes.get(1).getSqlTypeName())) {
                    returnType = binding.getTypeFactory().createSqlType(SqlTypeName.DATE);
                } else {
                    returnType = binding.getTypeFactory().createSqlType(SqlTypeName.TIMESTAMP);
                }
            } else {
                returnType = datetimeType;
            }
        }
        return returnType;
    }

    public static SqlReturnTypeInference dateAddReturnType(String fnName) {
        SqlReturnTypeInference returnTypeInference = opBinding -> {
            List<RelDataType> operandTypes = opBinding.collectOperandTypes();
            if (operandTypes.size() == 2) {
                return mySqlDateaddReturnType(opBinding);
            } else {
                return snowflakeDateaddReturnType(opBinding, fnName);
            }
        };
        return returnTypeInference.andThen(SqlTypeTransforms.TO_NULLABLE);
    }


    public static SqlReturnTypeInference ARRAY_MAP_GETITEM = opBinding -> inferReturnTypeArrayMapGetItem(opBinding);

    /**
     * Directly copied from
     * org.apache.calcite.sql.fun.SqlItemOperator#inferReturnType, with no changes.
     * So that we can correctly type our "GET" function, which is syntactic sugar for the getItem operator.
     *
     * Since this is copied, I'm pasting the Calcite License:
     *
     *  * Licensed to the Apache Software Foundation (ASF) under one or more
     *  * contributor license agreements.  See the NOTICE file distributed with
     *  * this work for additional information regarding copyright ownership.
     *  * The ASF licenses this file to you under the Apache License, Version 2.0
     *  * (the "License"); you may not use this file except in compliance with
     *  * the License.  You may obtain a copy of the License at
     *  *
     *  * http://www.apache.org/licenses/LICENSE-2.0
     *  *
     *  * Unless required by applicable law or agreed to in writing, software
     *  * distributed under the License is distributed on an "AS IS" BASIS,
     *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     *  * See the License for the specific language governing permissions and
     *  * limitations under the License.
     */
    static public RelDataType inferReturnTypeArrayMapGetItem(SqlOperatorBinding opBinding) {
        final RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
        final RelDataType operandType = opBinding.getOperandType(0);
        switch (operandType.getSqlTypeName()) {
            case ARRAY:
                return typeFactory.createTypeWithNullability(
                        getComponentTypeOrThrow(operandType), true);
            case MAP:
                return typeFactory.createTypeWithNullability(
                        requireNonNull(operandType.getValueType(),
                                () -> "operandType.getValueType() is null for " + operandType),
                        true);
            case ROW:
                RelDataType fieldType;
                RelDataType indexType = opBinding.getOperandType(1);

                if (SqlTypeUtil.isString(indexType)) {
                    final String fieldName = getOperandLiteralValueOrThrow(opBinding, 1, String.class);
                    RelDataTypeField field = operandType.getField(fieldName, false, false);
                    if (field == null) {
                        throw new AssertionError("Cannot infer type of field '"
                                + fieldName + "' within ROW type: " + operandType);
                    } else {
                        fieldType = field.getType();
                    }
                } else if (SqlTypeUtil.isIntType(indexType)) {
                    Integer index = opBinding.getOperandLiteralValue(1, Integer.class);
                    if (index == null || index < 1 || index > operandType.getFieldCount()) {
                        throw new AssertionError("Cannot infer type of field at position "
                                + index + " within ROW type: " + operandType);
                    } else {
                        fieldType = operandType.getFieldList().get(index - 1).getType(); // 1 indexed
                    }
                } else {
                    throw new AssertionError("Unsupported field identifier type: '"
                            + indexType + "'");
                }
                if (fieldType != null && operandType.isNullable()) {
                    fieldType = typeFactory.createTypeWithNullability(fieldType, true);
                }
                return fieldType;
            case ANY:
            case DYNAMIC_STAR:
                return typeFactory.createTypeWithNullability(
                        typeFactory.createSqlType(SqlTypeName.ANY), true);
            default:
                throw new AssertionError();
        }
    }

    public static final SqlReturnTypeInference TO_NUMBER_RET_TYPE =  ToNumberRetTypeHelper(true);

    public static final SqlReturnTypeInference TRY_TO_NUMBER_RET_TYPE = ToNumberRetTypeHelper(false);

    public static SqlReturnTypeInference ToNumberRetTypeHelper(boolean errorOnFailedConversion) {
        //First, generate the return type without considering output nullability
        SqlReturnTypeInference retType = opBinding -> {
            RelDataTypeFactory factory = opBinding.getTypeFactory();

            //TODO: the binding.getOperandCount() checks will need to be changed if/when we allow the 'format'
            // optional argument
            if (opBinding.getOperandCount() == 3) {
                //We have a fractional value, return Double
                assert opBinding.isOperandLiteral(2, true) : "Internal error in ToNumberRetTypeHelper: scale argument is not a literal";
                int scale = opBinding.getIntLiteralOperand(2);

                if (scale > 0) {
                    return factory.createSqlType(SqlTypeName.DOUBLE);
                }
            }

            //precicion defaults to 38 if not specified
            int precicion = 38;
            if (opBinding.getOperandCount() >= 2) {
                assert opBinding.isOperandLiteral(2, true) :
                        "Internal error in ToNumberRetTypeHelper: scale argument is not a literal";
                precicion = opBinding.getIntLiteralOperand(1);
            }
            return factory.createSqlType(getMinIntegerSize(precicion));
        };

        //Now handle nullability
        if (errorOnFailedConversion) {
            return retType.andThen(SqlTypeTransforms.TO_NULLABLE);
        } else {
            return retType.andThen(SqlTypeTransforms.FORCE_NULLABLE);

        }
    }

}
