package org.apache.calcite.sql;

import com.bodosql.calcite.catalog.SnowflakeCatalog;
import com.google.common.collect.ImmutableList;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.FunctionParameter;
import org.apache.calcite.schema.ScalarFunction;
import org.apache.calcite.sql.type.SqlOperandCountRanges;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlReturnTypeInference;
import org.apache.calcite.sql.validate.SqlUserDefinedFunction;

import java.util.List;
import java.util.Locale;

/**
 * Function definition used for all UDFs defined inside Snowflake. These functions
 * need to abide by Snowflake calling conventions and rules.
 */
public class SnowflakeUserDefinedFunction implements ScalarFunction {
    // -- Fields --

    // Full function path, including the name.
    private final ImmutableList<String> functionPath;
    // Hold a reference to the catalog to enable inlining the function.
    private final SnowflakeCatalog catalog;

    // The Snowflake function signature.
    // TODO: Replace this with a parsed function.
    private final String signature;

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
     * @param catalog              A link to the Snowflake catalog for finding implementation information
     * @param signature            The argument/return value signature from Snowflake
     * @param minNumArgs           The minimum number of required arguments. This is used for error handling while we have partial support.
     * @param maxNumArgs           The maximum number of required arguments. This is used for error handling while we have partial support.
     * @param isTableFunction      Is this function actually a table function.
     * @param errorInfo            Other snowflake fields used for generating errors.
     */
    protected SnowflakeUserDefinedFunction(ImmutableList<String> functionPath, SnowflakeCatalog catalog, String signature, int minNumArgs, int maxNumArgs, boolean isTableFunction, SnowflakeUserDefinedFunctionErrorInfo errorInfo) {
        this.functionPath = functionPath;
        this.catalog = catalog;
        this.signature = signature;
        // TODO: Remove arg counts when we support default arguments.
        this.minNumArgs = minNumArgs;
        this.maxNumArgs = maxNumArgs;
        this.isTableFunction = isTableFunction;
        this.errorInfo = errorInfo;
    }

    // -- static creators
    public static SnowflakeUserDefinedFunction create(ImmutableList<String> functionPath, SnowflakeCatalog catalog, String signature, int minNumArgs, int maxNumArgs, boolean isTableFunction, boolean isSecure, boolean isExternal, String language, boolean isMemoizable) {
        SnowflakeUserDefinedFunctionErrorInfo errorInfo = new SnowflakeUserDefinedFunctionErrorInfo(isSecure, isExternal, language, isMemoizable);
        return new SnowflakeUserDefinedFunction(functionPath, catalog, signature, minNumArgs, maxNumArgs, isTableFunction, errorInfo);
    }

    /**
     * Returns the return type of this function, constructed using the given
     * type factory.
     *
     * @param typeFactory Type factory
     */
    @Override
    public RelDataType getReturnType(RelDataTypeFactory typeFactory) {
        throw new RuntimeException(
                String.format(
                        Locale.ROOT,
                        "Unable to resolve function: %s.%s.%s. BodoSQL does not have support for Snowflake"
                                + " UDFs yet",
                        functionPath.get(0),
                        functionPath.get(1),
                        functionPath.get(2)));
    }

    /**
     * Returns the parameters of this function.
     *
     * @return Parameters; never null
     */
    @Override
    public List<FunctionParameter> getParameters() {
        throw new RuntimeException(
                String.format(
                        Locale.ROOT,
                        "Unable to resolve function: %s.%s.%s. BodoSQL does not have support for Snowflake"
                                + " UDFs yet",
                        functionPath.get(0),
                        functionPath.get(1),
                        functionPath.get(2)));
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
