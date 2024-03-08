package org.apache.calcite.sql;

import com.bodosql.calcite.table.ColumnDataTypeInfo;
import com.google.common.collect.ImmutableList;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.ScalarFunction;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.checkerframework.checker.nullness.qual.Nullable;

import static com.bodosql.calcite.catalog.SnowflakeCatalog.snowflakeTypeNameToTypeInfo;

/**
 * Function definition used for all UDFs defined inside Snowflake. These functions
 * need to abide by Snowflake calling conventions and rules.
 */
public class SnowflakeUserDefinedFunction extends SnowflakeUserDefinedBaseFunction implements ScalarFunction {
    // Temporary field to enable testing JavaScript UDFs.
    public static boolean enableJavaScript = false;


    // -- Fields --
    private final ColumnDataTypeInfo returnTypeInfo;

    // -- Constructors --

    /**
     * Creates a new Function for a call to a Snowflake User Defined Function.
     *
     * @param functionPath         The full path to this function including the function name.
     * @param args                 A string containing the args output from describe function.
     * @param numOptional          How many of the arguments are optional. Optional args must be at the end.
     * @param returns              A string containing the returns output from describe function
     * @param body                 A string containing the body output from describe function or NULL if the function is not a SQL UDF.
     * @param isSecure             Is the function a secure function?
     * @param isExternal           Is the function an external function?
     * @param language             What is the UDF's source language?
     * @param isMemoizable         Is the UDF memoizable?
     */
    protected SnowflakeUserDefinedFunction(ImmutableList<String> functionPath, String args, int numOptional, String returns, @Nullable String body, boolean isSecure, boolean isExternal, String language, boolean isMemoizable, BodoTZInfo tzInfo, java.sql.Timestamp createdOn) {
        super("UDF", functionPath, args, numOptional, body, isSecure, isExternal, language, isMemoizable, tzInfo, createdOn);
        this.returnTypeInfo = snowflakeTypeNameToTypeInfo(returns, true, tzInfo);
    }

    // -- Methods
    /**
     * Returns the return type of this function, constructed using the given
     * type factory.
     *
     * @param typeFactory Type factory
     */
    @Override
    public RelDataType getReturnType(RelDataTypeFactory typeFactory) {
        return returnTypeInfo.convertToSqlType(typeFactory);
    }

    // -- static creators
    public static SnowflakeUserDefinedFunction create(ImmutableList<String> functionPath, String args, int numOptional, String returns, @Nullable String body, boolean isSecure, boolean isExternal, String language, boolean isMemoizable, BodoTZInfo tzInfo, java.sql.Timestamp createdOn) {
        return new SnowflakeUserDefinedFunction(functionPath, args, numOptional, returns, body, isSecure, isExternal, language, isMemoizable, tzInfo, createdOn);
    }

    /**
     * Determine if this function implementation can support JavaScript.
     * We provide a type factory to allow for type checking the parameters
     * and return type.
     *
     * @param typeFactory The type factory to use for type checking.
     * @return True if the function can support JavaScript, false otherwise.
     */
    @Override
    boolean canSupportJavaScript(RelDataTypeFactory typeFactory) {
        return enableJavaScript;
    }
}
