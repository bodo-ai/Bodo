package org.apache.calcite.sql;

import com.google.common.collect.ImmutableMap;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlReturnTypeInference;
import org.apache.commons.lang3.NotImplementedException;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.jetbrains.annotations.NotNull;

import java.util.Map;

/**
 * Implementation of a SqlTableFunction used for builtin Table functions
 * defined by Snowflake.
 */
public class SnowflakeSqlTableFunction extends SqlFunction
        implements SqlTableFunction {

    private final SqlReturnTypeInference rowTypeInference;
    private final FunctionType type;

    // Map that tracks which input(s) hold the table characteristic
     private final Map<Integer, TableCharacteristic> characteristicMap;

    /**
     * Package private. Use SnowflakeSqlTableFunction.create() instead.
     */
     SnowflakeSqlTableFunction(String name, SqlReturnTypeInference rowTypeInference, @Nullable SqlOperandTypeChecker operandTypeChecker, FunctionType type, int tableOperandNum, TableCharacteristic.Semantics semantics) {
        super(name, SqlKind.OTHER_FUNCTION, ReturnTypes.CURSOR, null, operandTypeChecker, SqlFunctionCategory.USER_DEFINED_TABLE_FUNCTION);
        this.rowTypeInference = rowTypeInference;
        this.type = type;
        this.characteristicMap = createSingleCharacteristicMap(tableOperandNum, semantics);
    }

    /**
     * Returns the record type of the table yielded by this function when
     * applied to given arguments. Only literal arguments are passed,
     * non-literal are replaced with default values (null, 0, false, etc).
     *
     * @return strategy to infer the row type of a call to this function
     */
    @Override
    public SqlReturnTypeInference getRowTypeInference() {
        return rowTypeInference;
    }


    /**
     * Returns the table parameter characteristics for <code>ordinal</code>th
     * parameter to this table function.
     *
     * <p>Returns <code>null</code> if the <code>ordinal</code>th argument is
     * not table parameter or the <code>ordinal</code> is smaller than 0 or
     * the <code>ordinal</code> is greater than or equals to the number of
     * parameters.
     */
     @Override
     public @Nullable TableCharacteristic tableCharacteristic(int ordinal) {
        return characteristicMap.get(ordinal);
    }

    /**
     * Return the "TYPE" of Snowflake function. This is used for mapping the generic
     * table function type to more specific RelNodes.
     *
     * @return The type property stored on initialization.
     */
    public FunctionType getType() {
        return type;
    }

    /**
     * Gets a default value for the given argument number. If this argument
     * is not optional then this throws a RuntimeException.
     * @param builder Builder used for creating the default value.
     * @param argNumber Which argument is being checked.
     * @return A RexNode for the default value.
     */
    public @NotNull RexNode getDefaultValue(RexBuilder builder, int argNumber) {
        throw new NotImplementedException("getDefaultValue() must be implemented by a subclass");
    }

    public enum FunctionType {
        // Place-holder for all non-specialized functions.
        DEFAULT,
        FLATTEN,
        GENERATOR,
    }

    /**
     * Helper function to create a TABLE FUNCTION and perform the necessary checks.
     *
     * @param name The name of the function.
     * @param rowTypeInference How to infer the return types for each row of the input.
     * @param operandTypeChecker The type checker for the operands.
     * @param type The type of function, which is used for converting RelNodes later.
     * @param tableOperandNum Which input operand is an input table/array. This input
     *                        decides the semantics of the function (does it require some
     *                        aggregation or is it a mappable function)?
     * @param semantics The semantics of the table input. This is either row based
     *                  (mappable function) or Set based (aggregation like).
     */
    public static SnowflakeSqlTableFunction create(
            String name,
            SqlReturnTypeInference rowTypeInference,
            @Nullable SqlOperandTypeChecker operandTypeChecker,
            FunctionType type,
            int tableOperandNum,
            TableCharacteristic.Semantics semantics) {
        return new SnowflakeSqlTableFunction(name, rowTypeInference, operandTypeChecker, type, tableOperandNum, semantics);
    }

    public static Map<Integer, TableCharacteristic> createSingleCharacteristicMap(int operandNum, TableCharacteristic.Semantics semantics) {
        return ImmutableMap.of(
                operandNum,
                TableCharacteristic
                        .builder(semantics)
                        .passColumnsThrough()
                        .build()
        );
    }
}
