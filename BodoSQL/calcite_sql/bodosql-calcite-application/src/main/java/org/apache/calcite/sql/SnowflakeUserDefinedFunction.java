package org.apache.calcite.sql;

import com.bodosql.calcite.catalog.SnowflakeCatalog;
import com.bodosql.calcite.schema.SnowflakeUDFFunctionParameter;
import com.bodosql.calcite.table.BodoSQLColumn;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.google.common.collect.ImmutableList;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.FunctionParameter;
import org.apache.calcite.schema.ScalarFunction;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.sql.type.SqlOperandCountRanges;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlReturnTypeInference;
import org.apache.calcite.sql.validate.SqlUserDefinedFunction;
import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static com.bodosql.calcite.catalog.SnowflakeCatalog.snowflakeTypeNameToTypeInfo;

/**
 * Function definition used for all UDFs defined inside Snowflake. These functions
 * need to abide by Snowflake calling conventions and rules.
 */
public class SnowflakeUserDefinedFunction implements ScalarFunction {
    // -- Fields --

    // Full function path, including the name.
    public final ImmutableList<String> functionPath;

    private final SnowflakeCatalog.SnowflakeTypeInfo returnTypeInfo;

    private final List<FunctionParameter> parameters;

    // The function body
    private @Nullable String body;

    // Is this function a table function. Eventually this should be removed
    // when we add actual UserDefinedTableFunction support, but this
    private final boolean isTableFunction;
    // Hold information to determine functions we cannot support or may produce
    // performance concerns.
    private final SnowflakeUserDefinedFunctionErrorInfo errorInfo;
    private final int minNumArgs;
    private final int maxNumArgs;

    // -- Constructors --

    /**
     * Creates a new Function for a call to a Snowflake User Defined Function.
     *
     * @param functionPath         The full path to this function including the function name.
     * @param args                 A string containing the args output from describe function.
     * @param returns              A string containing the returns output from describe function
     * @param body                 A string containing the body output from describe function or NULL if the function is not a SQL UDF.
     * @param minNumArgs           The minimum number of required arguments. This is used for error handling while we have partial support.
     * @param maxNumArgs           The maximum number of required arguments. This is used for error handling while we have partial support.
     * @param isTableFunction      Is this function actually a table function.
     * @param errorInfo            Other snowflake fields used for generating errors.
     */
    protected SnowflakeUserDefinedFunction(ImmutableList<String> functionPath, String args, String returns, @Nullable String body, int minNumArgs, int maxNumArgs, boolean isTableFunction, SnowflakeUserDefinedFunctionErrorInfo errorInfo) {
        this.functionPath = functionPath;
        this.parameters = parseParameters(args);
        this.returnTypeInfo = snowflakeTypeNameToTypeInfo(returns);
        this.body = body;
        // TODO: Remove arg counts when we support default arguments.
        this.minNumArgs = minNumArgs;
        this.maxNumArgs = maxNumArgs;
        this.isTableFunction = isTableFunction;
        this.errorInfo = errorInfo;
    }

    // -- static creators
    public static SnowflakeUserDefinedFunction create(ImmutableList<String> functionPath, String args, String returns, @Nullable String body, int minNumArgs, int maxNumArgs, boolean isTableFunction, boolean isSecure, boolean isExternal, String language, boolean isMemoizable) {
        SnowflakeUserDefinedFunctionErrorInfo errorInfo = new SnowflakeUserDefinedFunctionErrorInfo(isSecure, isExternal, language, isMemoizable);
        return new SnowflakeUserDefinedFunction(functionPath, args, returns, body, minNumArgs, maxNumArgs, isTableFunction, errorInfo);
    }

    /**
     * Returns the return type of this function, constructed using the given
     * type factory.
     *
     * @param typeFactory Type factory
     */
    @Override
    public RelDataType getReturnType(RelDataTypeFactory typeFactory) {
        // Assume nullable for now.
        final boolean nullable = true;
        final BodoTZInfo tzInfo = BodoTZInfo.getDefaultTZInfo(typeFactory.getTypeSystem());
        // Make a dummy BodoSQLColumn for now.
        BodoSQLColumn.BodoSQLColumnDataType type = returnTypeInfo.columnDataType;
        BodoSQLColumn.BodoSQLColumnDataType elemType = returnTypeInfo.elemType;
        int precision = returnTypeInfo.precision;
        BodoSQLColumn column = new BodoSQLColumnImpl("", "", type, elemType, nullable, tzInfo, precision);
        return column.convertToSqlType(typeFactory, nullable, tzInfo, precision);
    }

    /**
     * Returns the parameters of this function.
     *
     * @return Parameters; never null
     */
    @Override
    public List<FunctionParameter> getParameters() {
       return parameters;
    }


    /**
     * Parse the args returned by described function.
     * @param args The arguments that are of the form:
     *             (name1 arg1, name2, arg2, ... nameN argN)
     */
    private List<FunctionParameter> parseParameters(String args) {
        List<FunctionParameter> parameters = new ArrayList<>();
        // Remove the starting ( and trailing )
        String trimmedArgs = args.substring(1, args.length() - 1).trim();
        if (!trimmedArgs.isEmpty()) {
            String[] inputs = trimmedArgs.split(",");
            for (int i = 0; i < inputs.length; i++) {
                String input = inputs[i];
                String[] typeParts = input.split(" ");
                String argName = typeParts[0];
                String type = typeParts[1];
                SnowflakeCatalog.SnowflakeTypeInfo typeInfo = snowflakeTypeNameToTypeInfo(type);
                parameters.add(new SnowflakeUDFFunctionParameter(argName, i, typeInfo));
            }
        }
        return parameters;
    }

    // -- Helper Classes --
    /**
     * Class that holds additional information returned by show functions that are not yet supported in Bodo.
     * This is used to compactly hold individualized error and warning messages if the Snowflake API changes.
     */
    protected static class SnowflakeUserDefinedFunctionErrorInfo {

        // -- Fields --
        private final boolean isSecure;
        private final boolean isExternal;
        private final String language;
        private final boolean isMemoizable;

        protected SnowflakeUserDefinedFunctionErrorInfo(boolean isSecure, boolean isExternal, String language, boolean isMemoizable) {
            this.isSecure = isSecure;
            this.isExternal = isExternal;
            // Normalize to upper case to simplify checks
            this.language = language.toUpperCase(Locale.ROOT);
            this.isMemoizable = isMemoizable;
        }
    }
}
