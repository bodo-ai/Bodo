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
import org.apache.calcite.sql.validate.SqlValidatorException;
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

    private final SnowflakeCatalog.SnowflakeTypeInfo returnTypeInfo;

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
     * @param returns              A string containing the returns output from describe function
     * @param body                 A string containing the body output from describe function or NULL if the function is not a SQL UDF.
     * @param isTableFunction      Is this function actually a table function.
     * @param isSecure             Is the function a secure function?
     * @param isExternal           Is the function an external function?
     * @param language             What is the UDF's source language?
     * @param isMemoizable         Is the UDF memoizable?
     */
    protected SnowflakeUserDefinedFunction(ImmutableList<String> functionPath, String args, String returns, @Nullable String body, boolean isTableFunction, boolean isSecure, boolean isExternal, String language, boolean isMemoizable) {
        this.functionPath = functionPath;
        this.parameters = parseParameters(args);
        this.returnTypeInfo = snowflakeTypeNameToTypeInfo(returns);
        this.body = body;
        this.errorInfo = new SnowflakeUserDefinedFunctionErrorInfo(isTableFunction, isSecure, isExternal, language, isMemoizable);
    }

    // -- static creators
    public static SnowflakeUserDefinedFunction create(ImmutableList<String> functionPath, String args, String returns, @Nullable String body, boolean isTableFunction, boolean isSecure, boolean isExternal, String language, boolean isMemoizable) {
        return new SnowflakeUserDefinedFunction(functionPath, args, returns, body, isTableFunction, isSecure, isExternal, language, isMemoizable);
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


    /**
     * Raise an error or warning if the Snowflake UDF contains features that we
     * are unable to support. A warning should only be raised if the function
     * doesn't error.
     */
    public void errorOrWarn() throws SqlValidatorException {
        this.errorInfo.error();
        this.errorInfo.warn();
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
         * Raise an error if we cannot support this UDF.
         * @throws SqlValidatorException the Validation error.
         */
        private void error() throws SqlValidatorException {
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
                throw BODO_SQL_RESOURCE.snowflakeUDFContainsUnsupportedFeature(databaseName, schemaName, functionName, errorMsg).ex();
            }
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
