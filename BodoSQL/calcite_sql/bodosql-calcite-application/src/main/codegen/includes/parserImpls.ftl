<#--
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to you under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
-->


boolean OrReplaceOpt() :
{
}
{
    <OR> <REPLACE> { return true; }
    |
    { return false; }
}


boolean IfNotExistsOpt() :
{
}
{
    <IF> <NOT> <EXISTS> { return true; }
    |
    { return false; }
}


SqlSnowflakeCreateTable.CreateTableType TableTypeOpt() :
{
}
{
// From https://docs.snowflake.com/en/sql-reference/sql/create-table#optional-parameters
// The synonyms and abbreviations for TEMPORARY (e.g. GLOBAL TEMPORARY) are provided for
// compatibility with other databases...
// Tables created with any of these keywords appear and behave identically to tables created
// using TEMPORARY.
// Default: No value. If a table is not declared as TRANSIENT or TEMPORARY, the table is permanent.

    <TRANSIENT> { return SqlSnowflakeCreateTable.CreateTableType.TRANSIENT; }
        |
    (<VOLATILE> | [<GLOBAL> | <LOCAL>] (<TEMP> | <TEMPORARY>)) { return SqlSnowflakeCreateTable.CreateTableType.TEMPORARY; }
         | { return SqlSnowflakeCreateTable.CreateTableType.DEFAULT; }
}

SqlNodeList ExtendColumnList() :
{
    final Span s;
    List<SqlNode> list = new ArrayList<SqlNode>();
    SqlNode col;
}
{
    <LPAREN> { s = span(); }
    col = ColumnWithType() { list.add(col); }
    (
        <COMMA> col = ColumnWithType() { list.add(col); }
    )*
    <RPAREN> {
        return new SqlNodeList(list, s.end(this));
    }
}

SqlNode ColumnWithType() :
{
    SqlIdentifier id;
    SqlDataTypeSpec type;
    boolean nullable = true;
    final Span s = Span.of();

    SqlNode defaultExpr = null;
    Pair<SqlLiteral,SqlLiteral> incrementExpr = null;
    SqlLiteral incrementStart = null;
    SqlLiteral incrementStep = null;
}
{
    id = CompoundIdentifier()
    type = DataType()
    // Any of the following additional column modifiers can appear in any order
    (
        ( <NOT> <NULL> { nullable = false; } )
    |   ( <DEFAULT_> defaultExpr = Expression(ExprContext.ACCEPT_SUB_QUERY) )
    |   ( 
          ( <AUTOINCREMENT> | <IDENTITY> )
          ( ( <LPAREN> incrementStart = NumericLiteral() <COMMA> incrementStep = NumericLiteral() <RPAREN> )
          | ( <START> incrementStart = NumericLiteral() <INCREMENT> incrementStep = NumericLiteral() )
          )
          { incrementExpr = new Pair<SqlLiteral, SqlLiteral>(incrementStart, incrementStep); }
        )
    )*
    {
        return new SqlSnowflakeColumnDeclaration(
            s.add(id).end(this), 
            id,
            type.withNullable(nullable),
            defaultExpr,
            incrementExpr,
            null
        );
    }
}

/* Parse a CLUSTER BY clause for a CREATE TABLE statement */
SqlNodeList ClusterBy() :
{
List<SqlNode> clusterExprsList;
}
{
    <CLUSTER> <BY> <LPAREN> clusterExprsList = SelectList() <RPAREN>
    { 
        return new SqlNodeList(clusterExprsList, getPos()); 
    }
}

SqlCreate SqlCreateTable(Span s, boolean replace) :
{
    final SqlSnowflakeCreateTable.CreateTableType tableType;
    final SqlIdentifier id;
    final boolean ifNotExists;
    SqlNode query = null;
    SqlNodeList columnList = null;
    SqlNodeList clusterExprs = null;
    boolean copyGrants = false;
}
{
    tableType = TableTypeOpt()
    <TABLE>
    ifNotExists = IfNotExistsOpt()
    id = CompoundIdentifier()
    (
        /* LIKE syntax
         * Column list banned
         * Following qualifiers optionally allowed in any order (after the LIKE clause):
         * - CLUSTER BY
         * - COPY GRANTS
         */
        (
            <LIKE> query = CompoundIdentifier()
            (

              clusterExprs = ClusterBy()
            | ( <COPY> <GRANTS> { copyGrants = true; } )
            )*
            {
                return new SqlSnowflakeCreateTableLike(s.end(this), replace, tableType,
                    ifNotExists, id, query, clusterExprs, copyGrants);
            }
        )
    |   /* CLONE syntax
         * Column list banned
         * Following qualifiers optionally allowed in any order (after the CLONE clause):
         * - COPY GRANTS
         */
        (
            <CLONE> query = CompoundIdentifier()
            (
              <COPY> <GRANTS> { copyGrants = true; } 
            )*
            {
                return new SqlSnowflakeCreateTableClone(s.end(this), replace, tableType,
                    ifNotExists, id, query, copyGrants);
            }
        )
        /* Regular syntax
         * Column list required
         * Following qualifiers optionally allowed in any order:
         * - CLUSTER BY
         * - COPY GRANTS (if the table is being replaced)
         * 
         * CTAS syntax
         * Column list optional
         * Following qualifiers optionally allowed in any order (before the AS clause):
         * - CLUSTER BY
         * - COPY GRANTS (if the table is being replaced)
         */
    |    ( 
            [ columnList = ExtendColumnList() ]
            (
              clusterExprs = ClusterBy()
            | ( <COPY> <GRANTS> { copyGrants = true; } )
            ) *
            [ <AS> query = OrderedQueryOrExpr(ExprContext.ACCEPT_QUERY) ]
            {
                // The query is only provided for CTAS statements
                if (query != null) {
                    return new SqlSnowflakeCreateTableAs(s.end(this), replace, tableType,
                        ifNotExists, id, columnList, query, clusterExprs, copyGrants);
                } else {
                    // Non-CTAS statements require the column names/types to be provided
                    assert columnList != null;
                    return new SqlSnowflakeCreateTable(s.end(this), replace, tableType,
                        ifNotExists, id, columnList, clusterExprs, copyGrants);
                }
            }
        )
    )
}

SqlAlterTable SqlAlterTable() :
{
    boolean ifExists;
    SqlIdentifier table;
    SqlIdentifier renameName = null;
    SqlIdentifier swapName = null;
    SqlNode addCol  = null;
    SqlIdentifier renameColOriginal = null;
    SqlIdentifier renameColNew = null;
    SqlNodeList dropCols = null;
}
{
    <ALTER> <TABLE>
    ifExists = IfExistsOpt()
    table = CompoundIdentifier()
    (
        ( <RENAME> <TO> renameName = SimpleIdentifier() 
        { return new SqlAlterTableRenameTable(getPos(), ifExists, table, renameName); })
    |   ( <SWAP> <WITH> swapName = CompoundIdentifier() 
        { return new SqlAlterTableSwapTable(getPos(), ifExists, table, swapName); })
    |   ( <ADD> [ <COLUMN> ] addCol = ColumnWithType() 
        { return new SqlAlterTableAddCol(getPos(), ifExists, table, addCol); })
    |   ( <RENAME> [ <COLUMN> ] renameColOriginal = SimpleIdentifier() <TO> renameColNew = SimpleIdentifier() 
        { return new SqlAlterTableRenameCol(getPos(), ifExists, table, renameColOriginal, renameColNew); })
    |   ( <DROP> [ <COLUMN> ] { dropCols = new SqlNodeList(getPos()); } SimpleIdentifierCommaList(dropCols)
        { return new SqlAlterTableDropCol(getPos(), ifExists, table, dropCols); })
    )
}

boolean IfExistsOpt() :
{
}
{
    <IF> <EXISTS> { return true; }
    |
    { return false; }
}

boolean CascadeOpt() :
{
    final boolean cascade;
}
{
    (
        <CASCADE> { cascade = true; }
    |
        <RESTRICT> { cascade = false; }
    |
        { cascade = true; }
    )
    { return cascade; }
}

SqlDdl SqlTruncate() :
{
    final Span s;
    final SqlDdl ddl;
}
{
    <TRUNCATE> { s = span(); }
    (
        ddl = SqlTruncateTable(s)
    )
    {
        return ddl;
    }
}

SqlTruncateTable SqlTruncateTable(Span s) :
{
    final boolean ifExists;
    final SqlIdentifier id;
}
{
    <TABLE>
    ifExists = IfExistsOpt()
    id = CompoundIdentifier()
    {
        return new SqlTruncateTable(s.end(this), ifExists, id);
    }
}

SqlDrop SqlDropTable(Span s, boolean replace) :
{
    final boolean ifExists;
    final boolean cascade;
    final SqlIdentifier id;
}
{
    <TABLE>
    ifExists = IfExistsOpt()
    id = CompoundIdentifier()
    cascade = CascadeOpt()
    {
        return new SqlDropTable(s.end(this), ifExists, id, cascade);
    }
}

SqlDrop SqlDropView(Span s, boolean replace) :
{
    final boolean ifExists;
    final SqlIdentifier id;
}
{
    <VIEW>
    ifExists = IfExistsOpt()
    id = CompoundIdentifier()
    {
        return SqlDdlNodes.dropView(s.end(this), ifExists, id);
    }
}




// /** Parses the infix "::" cast operator used in PostgreSQL. */
// void InfixCast(List<Object> list, ExprContext exprContext, Span s) :
// {
//     final SqlDataTypeSpec dt;
// }
// {
//     <INFIX_CAST> {
//         checkNonQueryExpression(exprContext);
//     }
//     dt = DataType() {
//         list.add(
//             new SqlParserUtil.ToTreeListItem(SqlLibraryOperators.INFIX_CAST,
//                 s.pos()));
//         list.add(dt);
//     }
// }


/** Parses the NULL-safe "<=>" equal operator used in MySQL. */
// void NullSafeEqual(List<Object> list, ExprContext exprContext, Span s):
// {
// }
// {
//     <NULL_SAFE_EQUAL> {
//         checkNonQueryExpression(exprContext);
//         list.add(new SqlParserUtil.ToTreeListItem(SqlLibraryOperators.NULL_SAFE_EQUAL, getPos()));
//     }
//     Expression2b(ExprContext.ACCEPT_SUB_QUERY, list)
// }
