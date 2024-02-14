package org.apache.calcite.rel.rel2sql;

import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan;
import com.bodosql.calcite.adapter.snowflake.SnowflakeToPandasConverter;
import com.google.common.collect.ImmutableList;
import org.apache.calcite.adapter.jdbc.JdbcTable;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.volcano.RelSubset;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.hint.RelHint;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlDialect;
import org.apache.calcite.sql.SqlHint;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.SqlTableRef;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.util.ImmutableBitSet;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.Objects.requireNonNull;

/**
 * Bodo implementation of RelToSqlConverter that includes support
 * for handling RelSubset information and our custom nodes.
 */
public class BodoRelToSqlConverter extends RelToSqlConverter {
    public BodoRelToSqlConverter(SqlDialect dialect) {
        super(dialect);
    }

    public Result visit(SnowflakeToPandasConverter e){
        return dispatch(e.getInput());
    }

    public Result visit(RelSubset e) {
        return dispatch(e.getBestOrOriginal());
    }

    /** Visits a SnowflakeTableScan; called by {@link #dispatch} via reflection. */
    public Result visit(SnowflakeTableScan e) {
        final SqlIdentifier identifier = getSqlTargetTable(e);
        final SqlNode sqlTable;
        final ImmutableList<RelHint> hints = e.getHints();
        if (!hints.isEmpty()) {
            SqlParserPos pos = identifier.getParserPosition();
            sqlTable =
                    new SqlTableRef(pos, identifier,
                            SqlNodeList.of(pos,
                                    hints.stream()
                                            .map(h -> toSqlHint(h, pos))
                                            .collect(Collectors.toList())));
        } else {
            sqlTable = identifier;
        }
        // Next apply column pruning
        final SqlNode node;
        if (!e.prunesColumns()) {
            // If we use every column just use the table.
            node = sqlTable;
        } else {
            final List<SqlNode> selectList = new ArrayList<>();
            final RelDataType prunedType = e.getRowType();
            for (int i = 0; i < prunedType.getFieldCount(); i++) {
                addSelect(selectList, new SqlIdentifier(prunedType.getFieldNames().get(i), SqlParserPos.ZERO), prunedType);
            }
            final SqlNodeList selectNodeList = new SqlNodeList(selectList, POS);
            node = new SqlSelect(SqlParserPos.ZERO, null, selectNodeList, sqlTable,
                    null, null, null, null, null, null, null, null);
        }
        return result(node, ImmutableList.of(Clause.FROM), e, null);

    }

    // Copied private methods
    private static SqlHint toSqlHint(RelHint hint, SqlParserPos pos) {
        if (hint.kvOptions != null) {
            return new SqlHint(pos, new SqlIdentifier(hint.hintName, pos),
                    SqlNodeList.of(pos, hint.kvOptions.entrySet().stream()
                            .flatMap(
                                    e -> Stream.of(new SqlIdentifier(e.getKey(), pos),
                                            SqlLiteral.createCharString(e.getValue(), pos)))
                            .collect(Collectors.toList())),
                    SqlHint.HintOptionFormat.KV_LIST);
        } else if (hint.listOptions != null) {
            return new SqlHint(pos, new SqlIdentifier(hint.hintName, pos),
                    SqlNodeList.of(pos, hint.listOptions.stream()
                            .map(e -> SqlLiteral.createCharString(e, pos))
                            .collect(Collectors.toList())),
                    SqlHint.HintOptionFormat.LITERAL_LIST);
        }
        return new SqlHint(pos, new SqlIdentifier(hint.hintName, pos),
                SqlNodeList.EMPTY, SqlHint.HintOptionFormat.EMPTY);
    }

    private static SqlIdentifier getSqlTargetTable(RelNode e) {
        // Use the foreign catalog, schema and table names, if they exist,
        // rather than the qualified name of the shadow table in Calcite.
        final RelOptTable table = requireNonNull(e.getTable());
        return table.maybeUnwrap(JdbcTable.class)
                .map(JdbcTable::tableName)
                .orElseGet(() ->
                        new SqlIdentifier(table.getQualifiedName(), SqlParserPos.ZERO));
    }


}
