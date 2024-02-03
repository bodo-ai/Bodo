package org.apache.calcite.sql;

import com.bodosql.calcite.schema.SnowflakeUDFFunctionParameter;
import org.apache.calcite.linq4j.Ord;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.Function;
import org.apache.calcite.schema.FunctionParameter;
import org.apache.calcite.sql.type.SnowflakeNamedOperandMetadataImpl;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlReturnTypeInference;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

import static org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE;
import static org.apache.calcite.util.Static.RESOURCE;


/**
 * Subclass of Snowflake Sql Table Function for functions that allow for named
 * arguments. This is given its own class to reduce errors made in constructing
 * a SQL function
 */
public class SnowflakeNamedArgumentSqlTableFunction extends SnowflakeSqlTableFunction {
    // Keep a copy of operandTypeChecker for safer API calls
    final @NotNull SnowflakeNamedOperandMetadataImpl operandTypeChecker;

    /**
     * Package private. Use SnowflakeNamedArgumentSqlTableFunction.create() instead.
     */
    SnowflakeNamedArgumentSqlTableFunction(String name, SqlReturnTypeInference rowTypeInference, @NotNull SnowflakeNamedOperandMetadataImpl operandTypeChecker, FunctionType type, int tableOperandNum, TableCharacteristic.Semantics semantics) {
        super(name, rowTypeInference, operandTypeChecker, type, tableOperandNum, semantics);
        this.operandTypeChecker = operandTypeChecker;
    }

    @Override
    public SnowflakeNamedOperandMetadataImpl getOperandTypeChecker() {
        return operandTypeChecker;
    }



    /**
     * Gets a default value for the given argument number. If this argument
     * is not optional then this throws a RuntimeException.
     * @param builder Builder used for creating the default value.
     * @param argNumber Which argument is being checked.
     * @return A RexNode for the default value.
     */
    @Override
    public @NotNull RexNode getDefaultValue(RexBuilder builder, int argNumber) {
        return getOperandTypeChecker().getDefaultValue(builder, argNumber);
    }

    /**
     * Checks that the operand values in a {@link SqlCall} to this operator are
     * valid. Subclasses must either override this method or supply an instance
     * of {@link SqlOperandTypeChecker} to the constructor.
     *
     * @param callBinding    description of call
     * @param throwOnFailure whether to throw an exception if check fails
     *                       (otherwise returns false in that case)
     * @return whether check succeeded
     */
    @Override
    public boolean checkOperandTypes(
            SqlCallBinding callBinding,
            boolean throwOnFailure) {
        final List<String> paramNames = operandTypeChecker.paramNames();
        for (Ord<SqlNode> operand : Ord.zip(callBinding.operands())) {
            if (operand.e != null
                    && operand.e.getKind() == SqlKind.DEFAULT
                    && !operandTypeChecker.isOptional(operand.i)) {
                throw callBinding.newValidationError(RESOURCE.defaultForOptionalParameter());
            }

            boolean isValidLiteral = operandTypeChecker.isValidLiteral(operand.e, operand.i);
            if (!isValidLiteral) {
                if (throwOnFailure) {
                    throw callBinding.newError(
                            BODO_SQL_RESOURCE.invalidLiteral(this.getName(),paramNames.get(operand.i)));
                }
                return false;
            }
            if (!operandTypeChecker.checkSingleOperandType(callBinding, operand.e, operand.i, throwOnFailure)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Helper function to create a TABLE FUNCTION allowing named Arguments perform the necessary checks. This ensures the
     * table can supported named arguments.
     *
     * @param name The name of the function.
     * @param rowTypeInference How to infer the return types for each row of the input.
     * @param operandTypeChecker The type checker for the operands with support for named operands.
     * @param type The type of function, which is used for converting RelNodes later.
     * @param tableOperandNum Which input operand is an input table/array. This input
     *                        decides the semantics of the function (does it require some
     *                        aggregation or is it a mappable function)?
     * @param semantics The semantics of the table input. This is either row based
     *                  (mappable function) or Set based (aggregation like).
     */
    public static SnowflakeNamedArgumentSqlTableFunction create(
            String name,
            SqlReturnTypeInference rowTypeInference,
            @NotNull SnowflakeNamedOperandMetadataImpl operandTypeChecker,
            FunctionType type,
            int tableOperandNum,
            TableCharacteristic.Semantics semantics) {
        return new SnowflakeNamedArgumentSqlTableFunction(name, rowTypeInference, operandTypeChecker, type, tableOperandNum, semantics);
    }
}
