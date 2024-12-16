package org.apache.calcite.sql;

import com.bodosql.calcite.schema.SnowflakeUDFFunctionParameter;
import com.bodosql.calcite.sql.BodoSqlUtil;
import com.bodosql.calcite.table.ColumnDataTypeInfo;
import com.google.common.collect.ImmutableList;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.runtime.Resources;
import org.apache.calcite.schema.Function;
import org.apache.calcite.schema.FunctionParameter;
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
 * Base function for UDFs and UDTFs.
 */
public abstract class SnowflakeUserDefinedBaseFunction implements Function {

    // -- Fields --
    private final String functionType;

    // Full function path, including the name.
    private final ImmutableList<String> functionPath;

    private final List<FunctionParameter> parameters;

    // The function body.
    private @Nullable String body;

    // What language is the UDF written in?
    private final String language;

    // Hold information to determine functions we cannot support or may produce
    // performance concerns.
    private final SnowflakeUserDefinedFunction.SnowflakeUserDefinedFunctionErrorInfo errorInfo;

    // When was the UDF created in Snowflake.
    private final java.sql.Timestamp createdOn;

    /**
     * Creates a new Function for a call to a Snowflake User Defined Function.
     *
     * @param functionType         What type of function do we have for generating error messages.
     * @param functionPath         The full path to this function including the function name.
     * @param args                 A string containing the args output from describe function.
     * @param numOptional          How many of the arguments are optional. Optional args must be at the end.
     * @param body                 A string containing the body output from describe function or NULL if the function is not a SQL UDF.
     * @param isSecure             Is the function a secure function?
     * @param isExternal           Is the function an external function?
     * @param language             What is the UDF's source language?
     * @param isMemoizable         Is the UDF memoizable?
     * @param createdOn            When was the UDF created?
     */
    SnowflakeUserDefinedBaseFunction(String functionType, ImmutableList<String> functionPath, String args, int numOptional, @Nullable String body, boolean isSecure, boolean isExternal, String language, boolean isMemoizable, java.sql.Timestamp createdOn) {
        this.functionType = functionType;
        this.functionPath = functionPath;
        this.parameters = parseParameters(args, numOptional);
        // Assume nullable for now
        this.body = body;
        // Normalize to upper case to simplify checks
        this.language = language.toUpperCase(Locale.ROOT);;
        this.errorInfo = new SnowflakeUserDefinedBaseFunction.SnowflakeUserDefinedFunctionErrorInfo(isSecure, isExternal, isMemoizable);
        this.createdOn = createdOn;
    }

    // Abstract methods
    public ImmutableList<String> getFunctionPath() {
        return functionPath;
    }
    public String getBody() {
        return body;
    }

    public String getLanguage() {
        return language;
    }

    /**
     * Return if this function type can be inlined.
     * @return True if the function can be inlined, false otherwise.
     */
    public boolean canInlineFunction() {
        return language.equals("SQL");
    }

    /**
     * Determine if this function implementation can support JavaScript.
     * We provide a type factory to allow for type checking the parameters
     * and return type.
     * @param typeFactory The type factory to use for type checking.
     * @return True if the function can support JavaScript, false otherwise.
     */
    abstract boolean canSupportJavaScript(RelDataTypeFactory typeFactory);

    /**
     * Determine if we can support the language of this function.
     * We provide a type factory to allow for type checking the parameters
     * and return type.
     * @param typeFactory The type factory to use for type checking.
     * @return True if BodoSQL supports this UDF language, false otherwise.
     */
    public boolean canSupportLanguage(RelDataTypeFactory typeFactory) {
        return this.language.equals("SQL") || (this.language.equals("JAVASCRIPT") && canSupportJavaScript(typeFactory));
    }

    public SnowflakeUserDefinedFunctionErrorInfo getErrorInfo() {
        return errorInfo;
    }

    public List<FunctionParameter> getParameters() {
        return parameters;
    }

    public java.sql.Timestamp getCreatedOn() {
        return createdOn;
    }

    // Implemented methods

    /**
     * Parse the args returned by described function.
     * @param args The arguments that are of the form:
     *             (name1 arg1, name2, arg2, ... nameN argN)
     */
    protected List<FunctionParameter> parseParameters(String args, int numOptional) {
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
                ColumnDataTypeInfo typeInfo = snowflakeTypeNameToTypeInfo(type, true);
                parameters.add(new SnowflakeUDFFunctionParameter(argName, i, typeInfo, i >= startOptionalIdx));
            }
        }
        return parameters;
    }

    /**
     * Returns an error or creates warning if the Snowflake UDF contains features that we
     * are unable to support. A warning should only be raised if the function
     * doesn't error.
     * @param typeFactory The type factory to use for type checking in case that impacts if
     *                    we can support the function.
     * @return Resources.ExInst<SqlValidatorException>: This is converted to a proper error
     * by the validator with node context.
     */
    public Resources.ExInst<SqlValidatorException> errorOrWarn(RelDataTypeFactory typeFactory) {
        Resources.ExInst<SqlValidatorException> retVal = getErrorInfo().error(typeFactory);
        if (retVal != null) {
            return retVal;
        }
        getErrorInfo().warn();
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
                argumentNames.add(getParameters().get(bit).getName());
            }
            String databaseName = getFunctionPath().get(0);
            String schemaName = getFunctionPath().get(1);
            String functionName = getFunctionPath().get(2);
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
    protected class SnowflakeUserDefinedFunctionErrorInfo {

        // -- Fields --
        private final boolean isSecure;
        private final boolean isExternal;
        private final boolean isMemoizable;

        protected SnowflakeUserDefinedFunctionErrorInfo(boolean isSecure, boolean isExternal, boolean isMemoizable) {
            this.isSecure = isSecure;
            this.isExternal = isExternal;
            this.isMemoizable = isMemoizable;
        }

        /**
         * Returns an error if we cannot support this UDF.
         * @return Resources.ExInst<SqlValidatorException>: This is converted to a proper error
         * by the validator with node context.
         */
        public Resources.ExInst<SqlValidatorException> error(RelDataTypeFactory typeFactory) {
            // Note this order is not strictly required, but in general we aim to go from least
            // to most recoverable.
            final String errorMsg;
            if (!SnowflakeUserDefinedBaseFunction.this.canSupportLanguage(typeFactory)) {
                // Add a message about JavaScript UDFs if we are a UDF. UDTFs don't support
                // JavaScript, so we don't want to confuse the user.
                final String javascriptMessage;
                if (functionType.equals("UDF")) {
                    javascriptMessage = " and has limited support for JavaScript UDFs";
                } else {
                    javascriptMessage = "";
                }
                errorMsg = String.format(Locale.ROOT, "Unsupported source language. Bodo supports SQL %ss%s. Source language found was %s.", functionType, javascriptMessage, SnowflakeUserDefinedBaseFunction.this.getLanguage());
            } else if (this.isExternal) {
                errorMsg = String.format(Locale.ROOT, "Bodo does not support external %ss", functionType);
            } else if (this.isSecure) {
                errorMsg = String.format(Locale.ROOT,"Bodo does not support secure %ss", functionType);
            } else if (SnowflakeUserDefinedBaseFunction.this.getBody() == null) {
                errorMsg = String.format(Locale.ROOT,"Unable to determine the function body for the %s.", functionType);
            } else {
                errorMsg = null;
            }
            if (errorMsg != null) {
                ImmutableList<String> functionPath = SnowflakeUserDefinedBaseFunction.this.getFunctionPath();
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
        public void warn() {
            if (this.isMemoizable) {
                ImmutableList<String> functionPath = SnowflakeUserDefinedBaseFunction.this.getFunctionPath();
                String databaseName = functionPath.get(0);
                String schemaName = functionPath.get(1);
                String functionName = functionPath.get(2);
                String msg = String.format(Locale.ROOT, "Function \"%s\".\"%s\".\"%s\" is defined to be Memoized, but BodoSQL doesn't support %s memoization yet. This may impact performance", databaseName, schemaName, functionName, functionType);
                VERBOSE_LEVEL_TWO_LOGGER.warning(msg);
            }
        }
    }
}
