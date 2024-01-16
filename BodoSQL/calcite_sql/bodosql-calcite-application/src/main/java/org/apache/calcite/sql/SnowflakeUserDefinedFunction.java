package org.apache.calcite.sql;

import com.bodosql.calcite.schema.SnowflakeUDFFunctionParameter;
import com.bodosql.calcite.sql.BodoSqlUtil;
import com.bodosql.calcite.table.ColumnDataTypeInfo;
import com.google.common.collect.ImmutableList;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.runtime.Resources;
import org.apache.calcite.schema.FunctionParameter;
import org.apache.calcite.schema.ScalarFunction;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.sql.validate.SqlValidatorException;
import org.apache.calcite.util.ImmutableBitSet;
import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static com.bodosql.calcite.application.PythonLoggers.VERBOSE_LEVEL_TWO_LOGGER;
import static com.bodosql.calcite.catalog.SnowflakeCatalog.snowflakeTypeNameToTypeInfo;
import static org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE;

/**
 * Function definition used for all UDFs defined inside Snowflake. These functions
 * need to abide by Snowflake calling conventions and rules.
 */
public class SnowflakeUserDefinedFunction implements ScalarFunction {
    // -- Fields --

    // Full function path, including the name.
    public final ImmutableList<String> functionPath;

    private final ColumnDataTypeInfo returnTypeInfo;

    private final List<FunctionParameter> parameters;

    // The function body
    private @Nullable String body;

    // Hold information to determine functions we cannot support or may produce
    // performance concerns.
    private final SnowflakeUserDefinedFunctionErrorInfo errorInfo;

    // -- Constructors --

    /**
     * Creates a new Function for a call to a Snowflake User Defined Function.
     *
     * @param functionPath         The full path to this function including the function name.
     * @param args                 A string containing the args output from describe function.
     * @param numOptional          How many of the arguments are optional. Optional args must be at the end.
     * @param returns              A string containing the returns output from describe function
     * @param body                 A string containing the body output from describe function or NULL if the function is not a SQL UDF.
     * @param isTableFunction      Is this function actually a table function.
     * @param isSecure             Is the function a secure function?
     * @param isExternal           Is the function an external function?
     * @param language             What is the UDF's source language?
     * @param isMemoizable         Is the UDF memoizable?
     */
    protected SnowflakeUserDefinedFunction(ImmutableList<String> functionPath, String args, int numOptional, String returns, @Nullable String body, boolean isTableFunction, boolean isSecure, boolean isExternal, String language, boolean isMemoizable, BodoTZInfo tzInfo) {
        this.functionPath = functionPath;
        this.parameters = parseParameters(args, numOptional, tzInfo);
        // Assume nullable for now
        this.returnTypeInfo = snowflakeTypeNameToTypeInfo(returns, true, tzInfo);
        this.body = body;
        this.errorInfo = new SnowflakeUserDefinedFunctionErrorInfo(isTableFunction, isSecure, isExternal, language, isMemoizable);
    }

    // -- static creators
    public static SnowflakeUserDefinedFunction create(ImmutableList<String> functionPath, String args, int numOptional, String returns, @Nullable String body, boolean isTableFunction, boolean isSecure, boolean isExternal, String language, boolean isMemoizable, BodoTZInfo tzInfo) {
        return new SnowflakeUserDefinedFunction(functionPath, args, numOptional, returns, body, isTableFunction, isSecure, isExternal, language, isMemoizable, tzInfo);
    }

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
    private List<FunctionParameter> parseParameters(String args, int numOptional, BodoTZInfo tzInfo) {
        List<FunctionParameter> parameters = new ArrayList<>();
        // Remove the starting ( and trailing )
        String trimmedArgs = args.substring(1, args.length() - 1).trim();
        if (!trimmedArgs.isEmpty()) {
            String[] inputs = trimmedArgs.split(",");
            int startOptionalIdx = inputs.length - numOptional;
            for (int i = 0; i < inputs.length; i++) {
                String input = inputs[i].trim();
                String[] typeParts = input.split(" ");
                String argName = typeParts[0];
                String type = typeParts[1];
                // Assume nullable for now
                ColumnDataTypeInfo typeInfo = snowflakeTypeNameToTypeInfo(type, true, tzInfo);
                parameters.add(new SnowflakeUDFFunctionParameter(argName, i, typeInfo, i >= startOptionalIdx));
            }
        }
        return parameters;
    }


    /**
     * Returns an error or creates warning if the Snowflake UDF contains features that we
     * are unable to support. A warning should only be raised if the function
     * doesn't error.
     * @return Resources.ExInst<SqlValidatorException>: This is converted to a proper error
     * by the validator with node context.
     */
    public Resources.ExInst<SqlValidatorException> errorOrWarn() {
        Resources.ExInst<SqlValidatorException> retVal = this.errorInfo.error();
        if (retVal != null) {
            return retVal;
        }
        this.errorInfo.warn();
        return null;
    }

    /**
     * Returns an error if any function argument is omitted. We cannot support
     * default values yet because this information is not yet available in
     * Snowflake metadata.
     * @return Resources.ExInst<SqlValidatorException>: This is converted to a proper error
     * by the validator with node context.
     */
    public Resources.ExInst<SqlValidatorException> errorOnDefaults(List<SqlNode> operandList) {
        ImmutableBitSet.Builder builder = ImmutableBitSet.builder();
        for (int i = 0; i < operandList.size(); i++) {
            SqlNode operand = operandList.get(i);
            if (BodoSqlUtil.isDefaultCall(operand)) {
                builder.set(i);
            }
        }
        ImmutableBitSet bits = builder.build();
        // If there is at least 1 bit set we need to throw an
        // exception because we cannot support default arguments.
        if (!bits.isEmpty()) {
            List<String> argumentNames = new ArrayList();
            for (int bit: bits) {
                argumentNames.add(parameters.get(bit).getName());
            }
            String databaseName = functionPath.get(0);
            String schemaName = functionPath.get(1);
            String functionName = functionPath.get(2);
            String badArguments = String.join(", ", argumentNames);
            return BODO_SQL_RESOURCE.snowflakeUDFContainsDefaultArguments(databaseName, schemaName, functionName, badArguments);
        }
        return null;
    }

    // -- Helper Classes --
    /**
     * Class that holds additional information returned by show functions that are not yet supported in Bodo.
     * This is used to compactly hold individualized error and warning messages if the Snowflake API changes.
     */
    private class SnowflakeUserDefinedFunctionErrorInfo {

        // -- Fields --

        // Is this function a table function. Eventually this should be removed
        // when we add actual UserDefinedTableFunction support.
        private final boolean isTableFunction;
        private final boolean isSecure;
        private final boolean isExternal;
        private final String language;
        private final boolean isMemoizable;

        private SnowflakeUserDefinedFunctionErrorInfo(boolean isTableFunction, boolean isSecure, boolean isExternal, String language, boolean isMemoizable) {
            this.isTableFunction = isTableFunction;
            this.isSecure = isSecure;
            this.isExternal = isExternal;
            // Normalize to upper case to simplify checks
            this.language = language.toUpperCase(Locale.ROOT);
            this.isMemoizable = isMemoizable;
        }

        /**
         * Returns an error if we cannot support this UDF.
         * @return Resources.ExInst<SqlValidatorException>: This is converted to a proper error
         * by the validator with node context.
         */
        private Resources.ExInst<SqlValidatorException> error() {
            // Note this order is not strictly required, but in general we aim to go from least
            // to most recoverable.
            final String errorMsg;
            if (!this.language.equals("SQL")) {
                errorMsg = String.format(Locale.ROOT, "Unsupported source language. Bodo only support SQL UDFs, but found %s", this.language);
            } else if (this.isExternal) {
                errorMsg = "Bodo does not support external UDFs";
            } else if (this.isSecure) {
                errorMsg = "Bodo does not support secure UDFs";
            } else if (this.isTableFunction) {
                errorMsg = "UDF is a table function. Bodo does not support UDTFs yet.";
            } else if (SnowflakeUserDefinedFunction.this.body == null) {
                errorMsg = "Unable to determine the function body for the UDF.";
            } else {
                errorMsg = null;
            }
            if (errorMsg != null) {
                ImmutableList<String> functionPath = SnowflakeUserDefinedFunction.this.functionPath;
                String databaseName = functionPath.get(0);
                String schemaName = functionPath.get(1);
                String functionName = functionPath.get(2);
                return BODO_SQL_RESOURCE.snowflakeUDFContainsUnsupportedFeature(databaseName, schemaName, functionName, errorMsg);
            }
            return null;
        }

        /**
         * Log a Level 2 warning if this UDF contains a feature that could impact performance.
         */
        private void warn() {
            if (this.isMemoizable) {
                ImmutableList<String> functionPath = SnowflakeUserDefinedFunction.this.functionPath;
                String databaseName = functionPath.get(0);
                String schemaName = functionPath.get(1);
                String functionName = functionPath.get(2);
                String msg = String.format(Locale.ROOT, "Function \"%s\".\"%s\".\"%s\" is defined to be Memoized, but BodoSQL doesn't support UDF memoization yet. This may impact performance", databaseName, schemaName, functionName);
                VERBOSE_LEVEL_TWO_LOGGER.warning(msg);
            }
        }
    }
}
