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


boolean IfNotExistsOpt() :
{
}
{
    <IF> <NOT> <EXISTS> { return true; }
    |
    { return false; }
}


SqlBodoCreateTable.CreateTableType TableTypeOpt() :
{
}
{
// From https://docs.snowflake.com/en/sql-reference/sql/create-table#optional-parameters
// The synonyms and abbreviations for TEMPORARY (e.g. GLOBAL TEMPORARY) are provided for
// compatibility with other databases...
// Tables created with any of these keywords appear and behave identically to tables created
// using TEMPORARY.
// Default: No value. If a table is not declared as TRANSIENT or TEMPORARY, the table is permanent.

    <TRANSIENT> { return SqlBodoCreateTable.CreateTableType.TRANSIENT; }
        |
    (<VOLATILE> | [<GLOBAL> | <LOCAL>] (<TEMP> | <TEMPORARY>)) { return SqlBodoCreateTable.CreateTableType.TEMPORARY; }
         | { return SqlBodoCreateTable.CreateTableType.DEFAULT; }
}

SqlNodeList ExtendColumnList() :
{
    final Span s;
    List<SqlNode> list = new ArrayList<SqlNode>();
}
{
    <LPAREN> { s = span(); }
    ColumnWithType(list)
    (
        <COMMA> ColumnWithType(list)
    )*
    <RPAREN> {
        return new SqlNodeList(list, s.end(this));
    }
}

void ColumnWithType(List<SqlNode> list) :
{
    SqlIdentifier id;
    SqlDataTypeSpec type;
    boolean nullable = true;
    final Span s = Span.of();
}
{
    id = CompoundIdentifier()
    type = DataType()
    [
        <NOT> <NULL> {
            nullable = false;
        }
    ]
    {
        list.add(SqlDdlNodes.column(s.add(id).end(this), id,
            type.withNullable(nullable), null, null));
    }
}

SqlCreate SqlCreateTable(Span s, boolean replace) :
{
    final SqlBodoCreateTable.CreateTableType tableType;
    final boolean ifNotExists;
    final SqlIdentifier id;
    final SqlNodeList columnList;
    final SqlNode query;
}
{
    tableType = TableTypeOpt()
    <TABLE>
    ifNotExists = IfNotExistsOpt()
    id = CompoundIdentifier()
    (
        columnList = ExtendColumnList()
    |
        { columnList = null; }
    )
    (
        <AS> query = OrderedQueryOrExpr(ExprContext.ACCEPT_QUERY)
    |
        <LIKE> query = TableRef()
    |
        { query = null; }
    )
    {
        return new SqlBodoCreateTable(s.end(this), replace, tableType,
            ifNotExists, id, columnList, query);
    }
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
