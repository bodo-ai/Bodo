package org.apache.calcite.rel.rel2sql;

import org.apache.calcite.plan.volcano.RelSubset;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.sql.SqlDialect;

/**
 * Bodo implementation of RelToSqlConverter that includes support
 * for handling RelSubset information.
 */
public class BodoRelToSqlConverter extends RelToSqlConverter {
    public BodoRelToSqlConverter(SqlDialect dialect) {
        super(dialect);
    }

    public Result visit(RelSubset e) {
        return dispatch(e.getBestOrOriginal());
    }
}
