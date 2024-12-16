package org.apache.calcite.sql.type;

import com.bodosql.calcite.sql.BodoSqlUtil;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.util.Util;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.IntFunction;
import java.util.function.Predicate;

/**
 * OperandMetadataImpl extension for builtin Table Functions with Snowflake
 * rules and named arguments
 */
public class SnowflakeNamedOperandMetadataImpl extends OperandMetadataImpl {

    private final @NotNull Predicate<Integer> requiresLiteral;
    private final @NotNull LiteralMatchFunction literalMatchFn;
    private final @NotNull DefaultFunction defaultFn;

    /**
     * Package private. Use SnowflakeNamedOperandMetadataImpl.create() instead
     */
    SnowflakeNamedOperandMetadataImpl(@NotNull List<SqlTypeFamily> families,
                                      @NotNull Function<RelDataTypeFactory, List<RelDataType>> paramTypesFactory,
                                      @NotNull IntFunction<String> paramNameFn, @NotNull Predicate<Integer> optional,
                                      @NotNull Predicate<Integer> requiresLiteral, @NotNull LiteralMatchFunction literalMatchFn,
                                      @NotNull DefaultFunction defaultFn) {
        super(families, paramTypesFactory, paramNameFn, optional);
        this.defaultFn = defaultFn;
        this.literalMatchFn = literalMatchFn;
        this.requiresLiteral = requiresLiteral;
    }

    /**
     * Gets a default value for the given argument number. If this argument
     * is not optional then this throws a RuntimeException.
     * @param builder Builder used for creating the default value.
     * @param argNumber Which argument is being checked.
     * @return A RexNode for the default value.
     */
    public @NotNull RexNode getDefaultValue(RexBuilder builder, int argNumber) {
        return defaultFn.getDefaultValue(builder, argNumber);
    }

    /**
     * Determine if the input is a valid literal argument. An argument
     * is a valid literal if it is not required to be a literal or it is a
     * Literal, and it matches one of the available literal values.
     *
     * @param argNumber
     * @return True if the argument is either not required to be a literal or
     * the value is one of the legal literal options.
     */
    public boolean isValidLiteral(SqlNode literal, int argNumber) {
        if (!requiresLiteral.test(argNumber)) {
            return true;
        }
        // Default is always a matching literal.
        if (BodoSqlUtil.isDefaultCall(literal)) {
            return true;
        }
        if (literal instanceof SqlLiteral) {
            return literalMatchFn.matchesLiteral((SqlLiteral) literal, argNumber);
        } else {
            return false;
        }
    }


    public static SnowflakeNamedOperandMetadataImpl create(List<SqlTypeFamily> families, List<String> paramNames,
                                                           int numRequired, List<Boolean> requiresLiteral,
                                                           @NotNull LiteralMatchFunction literalMatchFn, @NotNull DefaultFunction defaultFn) {
        Function<RelDataTypeFactory, List<RelDataType>> paramTypesFactory = relDataTypeFactory -> {
            List<RelDataType> types = new ArrayList<>();
            for (int i = 0; i < paramNames.size(); i++) {
                // TODO(njriasan): Replace with a real implementation
                types.add(relDataTypeFactory.createSqlType(SqlTypeName.ANY));
            }
            return types;
        };
        IntFunction<String> paramNameFn = i -> paramNames.get(i);
        // TODO: Determine if there is any Snowflake API where a required argument
        // can follow an optional argument. If so this implementation needs to be revised.
        Predicate<Integer> optional = i -> i >= numRequired;
        Predicate<Integer> requiresLiteralFn = i -> requiresLiteral.get(i);
        return new SnowflakeNamedOperandMetadataImpl(families, paramTypesFactory, paramNameFn,
                optional, requiresLiteralFn, literalMatchFn, defaultFn);
    }

    /**
     * Interface for checking if a literal is valid.
     */
    public interface LiteralMatchFunction {
        /**
         * Function call used to determine if a SqlLiteral is valid. This API is primarily
         * meant for arguments that only require a couple actual literal values, not type
         * checking, which will be handled by a separate check. To simplify the code if an
         * argument doesn't restrict the values it defaults to True.
         *
         * @param literal The Sql literal to check.
         * @param argNumber The formal argument number.
         * @return True if there are no restrictions on the input/literal. Otherwise returns
         * True if the literal is one of the allowed values that can't be verified simply by
         * type.
         */
        @NotNull boolean matchesLiteral(SqlLiteral literal, int argNumber);
    }

    /**
     * Interface for indicating default values.
     */
    public interface DefaultFunction {
        /**
         * Function call used to provide a default value for a given argument
         * with a builder. This should raise an exception if the argument doesn't
         * support a default.
         * @param builder A RexBuilder for building a default.
         * @param argNumber The formal argument number.
         * @return The RexNode matching the default value
         */
        @NotNull RexNode getDefaultValue(RexBuilder builder, int argNumber);
    }

    // Extend checkSingleOperandType for FamilyOperandType to allow
    // having multiple type families.
    @Override public boolean checkSingleOperandType(
            SqlCallBinding callBinding,
            SqlNode node,
            int iFormalOperand,
            boolean throwOnFailure) {
        return checkSingleOperandType(
                callBinding, node, iFormalOperand, families.get(iFormalOperand), throwOnFailure);
    }
}
