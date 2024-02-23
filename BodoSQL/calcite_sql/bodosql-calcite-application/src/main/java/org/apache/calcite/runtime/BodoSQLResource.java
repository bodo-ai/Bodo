package org.apache.calcite.runtime;

import org.apache.calcite.sql.validate.SqlValidatorException;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * Equivalent interface to CalciteResource but for BodoSQL's custom
 * error messages.
 */
public interface BodoSQLResource {

   @Resources.BaseMessage("Invalid Trim Syntax. We support TRIM([BOTH/TRAILING/LEADING] trimchars from Y ) and TRIM(X [, trimchars])")
   Resources.ExInst<CalciteException> genericTrimError();

   @Resources.BaseMessage("TABLESAMPLE argument must be between {0,number,#} and {1,number,#}, inclusive")
   @Resources.Property(name = "SQLSTATE", value = "2202H")
   Resources.ExInst<CalciteException> invalidSampleSize(Number min, Number max);

   @Resources.BaseMessage("Encountered an unconditional condition prior to a conditional condition in a MERGE INTO statement.")
   Resources.ExInst<SqlValidatorException> mergeClauseUnconditionalPrecedesConditional();

    @Resources.BaseMessage("CreateTable statements currently require an 'AS query'")
    Resources.ExInst<SqlValidatorException> createTableRequiresAsQuery();

   @Resources.BaseMessage("Encountered an unsupported join condition in an outer join that cannot be rewritten.")
   Resources.ExInst<CalciteException> unsupportedJoinCondition();

    @Resources.BaseMessage("Create Table statements cannot contain both ''OR REPLACE'' and ''IF NOT EXISTS''")
    Resources.ExInst<SqlValidatorException> createTableInvalidSyntax();

   @Resources.BaseMessage("Named Parameter table is not registered. To use named parameters in a query "
           + "please  register a table name in the configuration.")
   Resources.ExInst<SqlValidatorException> namedParamTableNotRegistered();

   @Resources.BaseMessage("Named Parameter table is registered with the name ''{0}'' but no table exists "
           + "with that name. To use namedParameters you must supply a namedParameters table.")
   Resources.ExInst<SqlValidatorException> namedParamTableNotFound(String tableName);

   @Resources.BaseMessage("SQL query contains a unregistered parameter: ''@{0}''")
   Resources.ExInst<SqlValidatorException> namedParamParameterNotFound(String paramName);

   @Resources.BaseMessage("Duplicate LIMIT: {0}")
   Resources.ExInst<CalciteException> duplicateLimit(String kind);

   @Resources.BaseMessage("Invalid time unit input for {0}: {1}")
   Resources.ExInst<SqlValidatorException> wrongTimeUnit(String fnName, String msg);

   @Resources.BaseMessage("Unsupported date/time unit \"{1}\" for function {0}")
   Resources.ExInst<SqlValidatorException> illegalDatePartTimeUnit(String functionName, String dateTimeUnit);

   @Resources.BaseMessage("Function \"{0}\" requires a valid literal for argument \"{1}\".")
   Resources.ExInst<SqlValidatorException> invalidLiteral(String fnName, String argName);

   @Resources.BaseMessage("Function \"{0}\" requires a valid array or json value for argument \"{1}\".")
   Resources.ExInst<SqlValidatorException> requiresArrayOrJson(String fnName, String argName);

   @Resources.BaseMessage("Function \"{0}\".\"{1}\".\"{2}\" contains an unsupported feature. Error message: \"{3}\"")
   Resources.ExInst<SqlValidatorException> snowflakeUDFContainsUnsupportedFeature(String databaseName, String schemaName, String functionName, String errorMessage);

   @Resources.BaseMessage("Function \"{0}\".\"{1}\".\"{2}\" uses default arguments, which are not supported on Snowflake UDFs because the default values cannot be found in Snowflake metadata. Missing argument(s): {3}")
   Resources.ExInst<SqlValidatorException> snowflakeUDFContainsDefaultArguments(String databaseName, String schemaName, String functionName, String argumentList);

   @Resources.BaseMessage("Encountered a table without read permissions when attempting to expand {0}.")
   Resources.ExInst<SqlValidatorException> noReadPermissionExpandingView(String viewQualifiedName);
}
