package org.apache.calcite.sql.fun;

import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.validate.SqlNameMatcher;

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;

/**
 * Operator table used for aggregate functions. This needs access to package private methods.
 */
public class SqlAggOperatorTable implements SqlOperatorTable {
    private static @Nullable SqlAggOperatorTable instance;

    /** Returns the Aggregation operator table, creating it if necessary. */
    public static synchronized SqlAggOperatorTable instance() {
        SqlAggOperatorTable instance = SqlAggOperatorTable.instance;
        if (instance == null) {
            // Creates and initializes the standard operator table.
            // Uses two-phase construction, because we can't initialize the
            // table until the constructor of the sub-class has completed.
            instance = new SqlAggOperatorTable();
            SqlAggOperatorTable.instance = instance;
        }
        return instance;
    }


    // Override LISTAGG because it has the wrong precision
    public static final SqlFunction LISTAGG = new SqlListaggAggFunction(SqlKind.LISTAGG, BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_NULLABLE);

    private List<SqlOperator> aggOperatorList = Arrays.asList(LISTAGG);

    @Override
    public void lookupOperatorOverloads(
            SqlIdentifier opName,
            @Nullable SqlFunctionCategory category,
            SqlSyntax syntax,
            List<SqlOperator> operatorList,
            SqlNameMatcher nameMatcher) {
        // Heavily copied from Calcite:
        // https://github.com/apache/calcite/blob/4bc916619fd286b2c0cc4d5c653c96a68801d74e/core/src/main/java/org/apache/calcite/sql/util/ListSqlOperatorTable.java#L57
        for (SqlOperator operator : aggOperatorList) {
            // All String Operators added are functions so far.

            if (syntax != operator.getSyntax()) {
                continue;
            }
            // Check that the name matches the desired names.
            if (!opName.isSimple() || !nameMatcher.matches(operator.getName(), opName.getSimple())) {
                continue;
            }
            // TODO: Check the category. The Lexing currently thinks
            //  all of these functions are user defined functions.
            operatorList.add(operator);
        }
    }

    @Override
    public List<SqlOperator> getOperatorList() {
        return aggOperatorList;
    }
}
