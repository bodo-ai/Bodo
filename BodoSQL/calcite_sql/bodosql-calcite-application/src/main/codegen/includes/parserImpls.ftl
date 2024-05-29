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

/**
 * Argument names that are otherwise reserved but can be used
 * as the name of an argument for a function call.
 */
 SqlIdentifier ArgumentNameUnreservedKeyword() :
{
    final String name;
    final SqlParserPos pos;
}
{
    (
        <#list (parser.argumentNameUnreserveList!default.parser.argumentNameUnreserveList) as keyword>
        <#if keyword?index == 0>
                <${keyword}>
        <#else>
            |   <${keyword}>
        </#if>
        </#list>
    )
    {
        name = unquotedIdentifier();
        pos = getPos();
        return new SqlIdentifier(name, pos);
    }
}

/**
 * Parses a superset of Named Argument to include reserved keywords
 * that can be used as argument names.
 */
SqlIdentifier NamedArgumentIdentifier() :
{
    final SqlIdentifier identifier;
    final String name;
}
{

    (
        identifier = SimpleIdentifier()
    |
        identifier = ArgumentNameUnreservedKeyword()

    )
    {
        return identifier;
    }
}



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
    SqlNode comment = null;
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
    |   ( <COMMENT> comment = StringLiteral() )
    )*
    {
        return new SqlSnowflakeColumnDeclaration(
            s.add(id).end(this),
            id,
            type.withNullable(nullable),
            defaultExpr,
            incrementExpr,
            comment,
            null
        );
    }
}

SqlNodeList ViewColumns() :
{
final Span s;
List<SqlNode> list = new ArrayList<SqlNode>();
    SqlNode col;
}
{
    <LPAREN> { s = span(); }
    col = ColumnWithOptionalType() { list.add(col); }
    (
        <COMMA> col = ColumnWithOptionalType() { list.add(col); }
    )*
    <RPAREN>
    {
        return new SqlNodeList(list, s.end(this));
    }
}

void WithTags() :
{
 // TODO: add way to store tags
}
{
    [ <WITH> ] <TAG>
    <LPAREN>
        CompoundIdentifier() <EQ> StringLiteral()
        ( <COMMA> CompoundIdentifier() <EQ> StringLiteral() )*
    <RPAREN>
}

void WithMaskingPolicy() :
{
// TODO: add way to store masking policy
}
{
    [ <WITH> ] <MASKING> <POLICY>
    CompoundIdentifier()
    [
        <USING>
        <LPAREN>
        CompoundIdentifier()
        ( <COMMA> Expression(ExprContext.ACCEPT_SUB_QUERY) )+
        <RPAREN>
    ]
}

void WithRowAccessPolicy() :
{
// TODO: add way to store row access policy
}
{
    [ <WITH> ] <ROW> <ACCESS> <POLICY>
    CompoundIdentifier()
    <ON>
    <LPAREN>
        CompoundIdentifier()
        ( <COMMA> CompoundIdentifier() )*
    <RPAREN>
}

SqlNode ColumnWithOptionalType() :
{
    SqlIdentifier id;
    SqlDataTypeSpec type = new SqlDataTypeSpec(new SqlBasicTypeNameSpec(SqlTypeName.UNKNOWN, SqlParserPos.ZERO), SqlParserPos.ZERO);
    boolean nullable = true;
    final Span s = Span.of();
    SqlNode defaultExpr = null;
    Pair<SqlLiteral,SqlLiteral> incrementExpr = null;
    SqlLiteral incrementStart = null;
    SqlLiteral incrementStep = null;
}
{
    id = CompoundIdentifier()
    [ type = DataType() ]
    // Any of the following additional column modifiers can appear in any order
    (
        ( <NOT> <NULL> { nullable = false; } )
    |   WithTags()
    |   WithMaskingPolicy()
    )*
    {
        return new SqlSnowflakeColumnDeclaration(
            s.add(id).end(this),
            id,
            type.withNullable(nullable),
            null,
            null,
            null,
            null
        );
    }
}

/* Parse a CLUSTER BY clause for a CREATE TABLE statement */
SqlNodeList ClusterBy() :
{
final List<SqlNode> clusterExprsList = new ArrayList<SqlNode>();
}
{
    <CLUSTER> <BY> <LPAREN>
    AddSelectItem(clusterExprsList)
    ( <COMMA> AddSelectItem(clusterExprsList) )*
    <RPAREN>
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
    SqlNode comment = null;
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
            | ( <COMMENT> <EQ> comment = StringLiteral() )
            )*
            {
                return new SqlSnowflakeCreateTableLike(s.end(this), replace, tableType,
                    ifNotExists, id, query, clusterExprs, copyGrants, comment);
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
            | ( <COMMENT> <EQ> comment = StringLiteral() )
            )*
            {
                return new SqlSnowflakeCreateTableClone(s.end(this), replace, tableType,
                    ifNotExists, id, query, copyGrants, comment);
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
            | ( <COMMENT> <EQ> comment = StringLiteral() )
            ) *
            [ <AS> query = OrderedQueryOrExpr(ExprContext.ACCEPT_QUERY) ]
            {
                // The query is only provided for CTAS statements
                if (query != null) {
                    return new SqlSnowflakeCreateTableAs(s.end(this), replace, tableType,
                        ifNotExists, id, columnList, query, clusterExprs, copyGrants, comment);
                } else {
                    // Non-CTAS statements require the column names/types to be provided
                    assert columnList != null;
                    return new SqlSnowflakeCreateTable(s.end(this), replace, tableType,
                        ifNotExists, id, columnList, clusterExprs, copyGrants, comment);
                }
            }
        )
    )
}

void CompressionType(HashMap<String, String> formatOptions) :
{
}
{
        <AUTO> { formatOptions.put("COMPRESSION", "AUTO"); }
    |   <GZIP> { formatOptions.put("COMPRESSION", "GZIP"); }
    |   <BZ2> { formatOptions.put("COMPRESSION", "BZ2"); }
    |   <BROTLI> { formatOptions.put("COMPRESSION", "BROTLI"); }
    |   <ZSTD> { formatOptions.put("COMPRESSION", "ZSTD"); }
    |   <DEFLATE> { formatOptions.put("COMPRESSION", "DEFLATE"); }
    |   <RAW_DEFLATE> { formatOptions.put("COMPRESSION", "RAW_DEFLATE"); }
    |   <SNAPPY> { formatOptions.put("COMPRESSION", "SNAPPY"); }
    |   <NONE> { formatOptions.put("COMPRESSION", "NONE"); }
}

void BinaryFormat(HashMap<String, String> formatOptions) :
{
}
{
        <HEX> { formatOptions.put("BINARY_FORMAT", "HEX"); }
    |   <BASE64> { formatOptions.put("BINARY_FORMAT", "BASE64"); }
    |   <UTF8> { formatOptions.put("BINARY_FORMAT", "UTF8"); }
}

void EncodingFormat(HashMap<String, String> formatOptions) :
{
    SqlNode stringArg;
}
{
        stringArg = StringLiteral() { formatOptions.put("ENCODING", stringArg.toString()); }
    |   <UTF8> { formatOptions.put("ENCODING", "UTF8"); }
}

void FormatArgStringOrNone(String argName, HashMap<String, String> formatOptions) :
{
    SqlNode stringArg;
}
{
        stringArg = StringLiteral() { formatOptions.put(argName, stringArg.toString()); }
    |   <NONE> { formatOptions.put(argName, "NONE"); }
}

void FormatArgStringOrAuto(String argName, HashMap<String, String> formatOptions) :
{
    SqlNode stringArg;
}
{
        stringArg = StringLiteral() { formatOptions.put(argName, stringArg.toString()); }
    |   <AUTO> { formatOptions.put(argName, "AUTO"); }
}

void FormatArgParenthesizedStrings(String argName, HashMap<String, String> formatOptions) :
{
    SqlNode stringArg;
    List<String> stringArgs = new ArrayList<String>();
}
{
    <LPAREN>
        stringArg = StringLiteral() { stringArgs.add(stringArg.toString()); }
        (
            <COMMA>
            stringArg = StringLiteral() { stringArgs.add(stringArg.toString()); }
        )*
    <RPAREN>
    { formatOptions.put(argName, "(" + String.join(", ", stringArgs) + ")"); }
}

void FormatArgBoolean(String argName, HashMap<String, String> formatOptions) :
{
}
{
        <TRUE> { formatOptions.put(argName, "TRUE"); }
    |   <FALSE> { formatOptions.put(argName, "FALSE"); }
}

void FormatArgInteger(String argName, HashMap<String, String> formatOptions) :
{
    SqlNode numericArg;
}
{
        <UNSIGNED_INTEGER_LITERAL> { formatOptions.put(argName, token.image.toString()); }
}

void FormatOption(HashMap<String, String> formatOptions) :
{
}
{
    /* The following format options can be provided in any order
    -- If FILE_FORMAT = ( TYPE = CSV ... )
        COMPRESSION = AUTO | GZIP | BZ2 | BROTLI | ZSTD | DEFLATE | RAW_DEFLATE | NONE
        RECORD_DELIMITER = '<character>' | NONE
        FIELD_DELIMITER = '<character>' | NONE
        PARSE_HEADER = TRUE | FALSE
        SKIP_HEADER = <integer>
        SKIP_BLANK_LINES = TRUE | FALSE
        DATE_FORMAT = '<string>' | AUTO
        TIME_FORMAT = '<string>' | AUTO
        TIMESTAMP_FORMAT = '<string>' | AUTO
        BINARY_FORMAT = HEX | BASE64 | UTF8
        ESCAPE = '<character>' | NONE
        ESCAPE_UNENCLOSED_FIELD = '<character>' | NONE
        TRIM_SPACE = TRUE | FALSE
        FIELD_OPTIONALLY_ENCLOSED_BY = '<character>' | NONE
        NULL_IF = ( '<string>' [ , '<string>' ... ] )
        ERROR_ON_COLUMN_COUNT_MISMATCH = TRUE | FALSE
        REPLACE_INVALID_CHARACTERS = TRUE | FALSE
        EMPTY_FIELD_AS_NULL = TRUE | FALSE
        SKIP_BYTE_ORDER_MARK = TRUE | FALSE
        ENCODING = '<string>' | UTF8
    -- If FILE_FORMAT = ( TYPE = JSON ... )
        COMPRESSION = AUTO | GZIP | BZ2 | BROTLI | ZSTD | DEFLATE | RAW_DEFLATE | NONE
        DATE_FORMAT = '<string>' | AUTO
        TIME_FORMAT = '<string>' | AUTO
        TIMESTAMP_FORMAT = '<string>' | AUTO
        BINARY_FORMAT = HEX | BASE64 | UTF8
        TRIM_SPACE = TRUE | FALSE
        NULL_IF = ( '<string>' [ , '<string>' ... ] )
        ENABLE_OCTAL = TRUE | FALSE
        ALLOW_DUPLICATE = TRUE | FALSE
        STRIP_OUTER_ARRAY = TRUE | FALSE
        STRIP_NULL_VALUES = TRUE | FALSE
        REPLACE_INVALID_CHARACTERS = TRUE | FALSE
        IGNORE_UTF8_ERRORS = TRUE | FALSE
        SKIP_BYTE_ORDER_MARK = TRUE | FALSE
    -- If FILE_FORMAT = ( TYPE = AVRO ... )
        COMPRESSION = AUTO | GZIP | BROTLI | ZSTD | DEFLATE | RAW_DEFLATE | NONE
        TRIM_SPACE = TRUE | FALSE
        NULL_IF = ( '<string>' [ , '<string>' ... ] )
    -- If FILE_FORMAT = ( TYPE = ORC ... )
        TRIM_SPACE = TRUE | FALSE
        NULL_IF = ( '<string>' [ , '<string>' ... ] )
    -- If FILE_FORMAT = ( TYPE = PARQUET ... )
        COMPRESSION = AUTO | SNAPPY | NONE
        BINARY_AS_TEXT = TRUE | FALSE
        TRIM_SPACE = TRUE | FALSE
        NULL_IF = ( '<string>' [ , '<string>' ... ] )
    -- If FILE_FORMAT = ( TYPE = XML ... )
        COMPRESSION = AUTO | GZIP | BZ2 | BROTLI | ZSTD | DEFLATE | RAW_DEFLATE | NONE
        IGNORE_UTF8_ERRORS = TRUE | FALSE
        PRESERVE_SPACE = TRUE | FALSE
        STRIP_OUTER_ELEMENT = TRUE | FALSE
        DISABLE_SNOWFLAKE_DATA = TRUE | FALSE
        DISABLE_AUTO_CONVERT = TRUE | FALSE
        SKIP_BYTE_ORDER_MARK = TRUE | FALSE
    */
    ( <COMPRESSION> <EQ> CompressionType(formatOptions) )
|   ( <RECORD_DELIMITER> <EQ> FormatArgStringOrNone("RECORD_DELIMITER", formatOptions) )
|   ( <FIELD_DELIMITER> <EQ> FormatArgStringOrNone("FIELD_DELIMITER", formatOptions) )
|   ( <PARSE_HEADER> <EQ> FormatArgBoolean("PARSE_HEADER", formatOptions) )
|   ( <SKIP_HEADER> <EQ> FormatArgInteger("SKIP_HEADER", formatOptions) )
|   ( <SKIP_BLANK_LINES> <EQ> FormatArgBoolean("SKIP_BLANK_LINES", formatOptions) )
|   ( <DATE_FORMAT> <EQ> FormatArgStringOrAuto("DATE_FORMAT", formatOptions) )
|   ( <TIME_FORMAT> <EQ> FormatArgStringOrAuto("TIME_FORMAT", formatOptions) )
|   ( <TIMESTAMP_FORMAT> <EQ> FormatArgStringOrAuto("TIMESTAMP_FORMAT", formatOptions) )
|   ( <BINARY_FORMAT> <EQ> BinaryFormat(formatOptions) )
|   ( <ESCAPE> <EQ> FormatArgStringOrNone("ESCAPE", formatOptions) )
|   ( <ESCAPE_UNENCLOSED_FIELD> <EQ> FormatArgStringOrNone("ESCAPE_UNENCLOSED_FIELD", formatOptions) )
|   ( <TRIM_SPACE> <EQ> FormatArgBoolean("TRIM_SPACE", formatOptions) )
|   ( <FIELD_OPTIONALLY_ENCLOSED_BY> <EQ> FormatArgStringOrNone("FIELD_OPTIONALLY_ENCLOSED_BY", formatOptions) )
|   ( <NULL_IF>  <EQ> FormatArgParenthesizedStrings("NULL_IF", formatOptions))
|   ( <ERROR_ON_COLUMN_COUNT_MISMATCH> <EQ> FormatArgBoolean("ERROR_ON_COLUMN_COUNT_MISMATCH", formatOptions) )
|   ( <REPLACE_INVALID_CHARACTERS> <EQ> FormatArgBoolean("REPLACE_INVALID_CHARACTERS", formatOptions) )
|   ( <EMPTY_FIELD_AS_NULL> <EQ> FormatArgBoolean("EMPTY_FIELD_AS_NULL", formatOptions) )
|   ( <SKIP_BYTE_ORDER_MARK> <EQ> FormatArgBoolean("SKIP_BYTE_ORDER_MARK", formatOptions) )
|   ( <ENCODING> <EQ> EncodingFormat(formatOptions) )
|   ( <ENABLE_OCTAL> <EQ> FormatArgBoolean("ENABLE_OCTAL", formatOptions) )
|   ( <ALLOW_DUPLICATE> <EQ> FormatArgBoolean("ALLOW_DUPLICATE", formatOptions) )
|   ( <STRIP_OUTER_ARRAY> <EQ> FormatArgBoolean("STRIP_OUTER_ARRAY", formatOptions) )
|   ( <STRIP_NULL_VALUES> <EQ> FormatArgBoolean("STRIP_NULL_VALUES", formatOptions) )
|   ( <IGNORE_UTF8_ERRORS> <EQ> FormatArgBoolean("IGNORE_UTF8_ERRORS", formatOptions) )
|   ( <BINARY_AS_TEXT> <EQ> FormatArgBoolean("BINARY_AS_TEXT", formatOptions) )
|   ( <PRESERVE_SPACE> <EQ> FormatArgBoolean("PRESERVE_SPACE", formatOptions) )
|   ( <STRIP_OUTER_ELEMENT> <EQ> FormatArgBoolean("STRIP_OUTER_ELEMENT", formatOptions) )
|   ( <DISABLE_SNOWFLAKE_DATA> <EQ> FormatArgBoolean("DISABLE_SNOWFLAKE_DATA", formatOptions) )
|   ( <DISABLE_AUTO_CONVERT> <EQ> FormatArgBoolean("DISABLE_AUTO_CONVERT", formatOptions) )
}

SqlSnowflakeFileFormat FileFormat() :
{
    SqlNode formatName = null;
    SqlNode formatType = null;
    HashMap<String, String> formatOptions = new HashMap<String, String>();
}
{
    <LPAREN>
    (
        ( <FORMAT_NAME> <EQ> formatName = StringLiteral() )
    |   ( <TYPE> <EQ> formatType = StringLiteral()
        [ FormatOption(formatOptions) ([<COMMA>] FormatOption(formatOptions))* ]
        )
    )
    <RPAREN>
    {
        return new SqlSnowflakeFileFormat(formatName, formatType, formatOptions);
    }
}

List<SqlNode> ParenthesizedColumnList() :
{
    SqlNode col;
    List<SqlNode> cols = new ArrayList<SqlNode>();
}
{
    <LPAREN>
    col = CompoundIdentifier() { cols.add(col); }
    (
        <COMMA>
        col = CompoundIdentifier() { cols.add(col); }
    )*
    <RPAREN>
    { return cols; }
}

List<SqlNode> TransformationColumns() :
{
    SqlNode col;
    List<SqlNode> cols = new ArrayList<SqlNode>();

}
{
    <TRANSFORM_COLUMN> { cols.add(new SqlIdentifier(token.image.toUpperCase(Locale.ROOT), getPos())); }
    (
        <COMMA>
        <TRANSFORM_COLUMN>
        { cols.add(new SqlIdentifier(token.image.toUpperCase(Locale.ROOT), getPos())); }
    )*
    { return cols; }
}

SqlNode SqlCopyInto() :
{
    SqlNode target;
    List<SqlNode> targetCols = null;
    SqlNode source = null;
    List<SqlNode> sourceCols = null;
    SqlNode sourceSource = null;
    SqlNode sourceAlias = null;
    SqlNode sourceQuery = null;
    SqlNode partition = null;
    SqlNode pattern = null;
    SqlSnowflakeFileFormat fileFormat = null;
    SqlCopyIntoTable.CopyIntoTableSource tableSourceType = null;
    SqlCopyIntoLocation.CopyIntoLocationTarget locationTargetType = null;
    SqlCopyIntoLocation.CopyIntoLocationSource locationSourceType = null;
}
{
    <COPY>
    <INTO>
(
    (
        // COPY INTO <table>
        target = CompoundIdentifier()
        (
            (
                // Version 1: COPY INTO [namespace.]table_name (col_name_1[, col_name_2, ...])
                //            FROM (SELECT [<alias>.]$<file_col_num>[.<element>] [ , [<alias>.]$<file_col_num>[.<element>] ... ])
                //                  FROM { internalStage | externalStage })
                targetCols = ParenthesizedColumnList()
                <FROM>
                <LPAREN>
                <SELECT>
                sourceCols = TransformationColumns()
                <FROM>
                ( <AT_PARAM_IDENTIFIER>  | <INTERNAL_OR_EXTERNAL_STAGE> )
                { sourceSource = new SqlIdentifier(token.image, getPos()); }
                [
                    sourceAlias = SimpleIdentifier()
                    {
                        sourceSource = SqlStdOperatorTable.AS.createCall(getPos(), sourceSource, sourceAlias);
                    }
                ]
                <RPAREN>
                {
                    tableSourceType = SqlCopyIntoTable.CopyIntoTableSource.QUERY;
                    source = new SqlSelect(
                        getPos(), null,
                        new SqlNodeList(sourceCols, Span.of(sourceCols).pos()),
                        sourceSource, null, null, null, null, null, null, null, null, null);
                }
            )
        |  (
                <FROM>
                (
                    // Version 2: COPY INTO [namespace.]table_name
                    //            FROM (SELECT [<alias>.]$<file_col_num>[.<element>] [ , [<alias>.]$<file_col_num>[.<element>] ... ])
                    //                  FROM { internalStage | externalStage })
                    (
                        <LPAREN>
                        <SELECT>
                        sourceCols = TransformationColumns()
                        <FROM>
                        ( <AT_PARAM_IDENTIFIER>  | <INTERNAL_OR_EXTERNAL_STAGE> )
                        { sourceSource = new SqlIdentifier(token.image, getPos()); }
                        [
                            sourceAlias = SimpleIdentifier()
                            {
                                sourceSource = SqlStdOperatorTable.AS.createCall(getPos(), sourceSource, sourceAlias);
                            }
                        ]
                        <RPAREN>
                        {
                            tableSourceType = SqlCopyIntoTable.CopyIntoTableSource.QUERY;
                            source = new SqlSelect(
                                getPos(), null,
                                new SqlNodeList(sourceCols, Span.of(sourceCols).pos()),
                                sourceSource, null, null, null, null, null, null, null, null, null);
                        }
                    )
                    // Version 3: COPY INTO [namespace.]table_name FROM {internalStage | externalStage | externalLocation}
                |   ( ( <AT_PARAM_IDENTIFIER>  | <INTERNAL_OR_EXTERNAL_STAGE> )
                    { source = new SqlIdentifier(token.image, getPos());
                     tableSourceType = SqlCopyIntoTable.CopyIntoTableSource.STAGE; })
                |   ( source = StringLiteral()
                    { tableSourceType = SqlCopyIntoTable.CopyIntoTableSource.LOCATION; } )
                )
            )
        )
    // The following clauses are optionally allowed in any order
    (
        <PATTERN> <EQ> pattern = StringLiteral()
    |   <FILE_FORMAT> <EQ> fileFormat = FileFormat()
    )*
    {  return new SqlCopyIntoTable(getPos(), target, targetCols, tableSourceType, source, pattern, fileFormat); }
    )
|   (
        // COPY INTO <location>
        (
            ( target = StringLiteral()
            { locationTargetType = SqlCopyIntoLocation.CopyIntoLocationTarget.LOCATION; } )
        |   ( ( <AT_PARAM_IDENTIFIER>  | <INTERNAL_OR_EXTERNAL_STAGE> )
            { target = new SqlIdentifier(token.image, getPos());
              locationTargetType = SqlCopyIntoLocation.CopyIntoLocationTarget.STAGE; } )
        )
        <FROM>
        // The source is either a table or a query
        (
            ( source = CompoundIdentifier()
            { locationSourceType = SqlCopyIntoLocation.CopyIntoLocationSource.TABLE; } )
        |   ( <LPAREN> source = SqlSelect() <RPAREN>
            { locationSourceType = SqlCopyIntoLocation.CopyIntoLocationSource.QUERY; } )
        )
        // The following clauses are optionally allowed in any order
        (
            <PARTITION> <BY> partition = Expression(ExprContext.ACCEPT_NON_QUERY)
        |   <FILE_FORMAT> <EQ> fileFormat = FileFormat()
        )*
        {  return new SqlCopyIntoLocation(getPos(), locationTargetType, target, locationSourceType, source, partition, fileFormat); }
    )
)

}

// This will attempt to match on all ALTER TABLE nodes we currently have implemented.
// If it encounters a ALTER TABLE statement that it doesn't know how to parse,
// it will raise a SqlParseError.

SqlAlterTable SqlAlterTable(Span s) :
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
    <TABLE>
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
    |   ( <DROP> [ <COLUMN> ] { dropCols = new SqlNodeList(getPos()); } AddSimpleIdentifiers(dropCols)
        { return new SqlAlterTableDropCol(getPos(), ifExists, table, dropCols); })
    )
}

// This will attempt to match on all ALTER VIEW nodes we currently have implemented.
// If it encounters a ALTER TABLE statement that it doesn't know how to parse,
// it will raise a SqlParseError.

SqlAlterView SqlAlterView(Span s) :
{
    boolean ifExists;
    SqlIdentifier view;
    SqlIdentifier renameName = null;
}
{
    <VIEW>
    ifExists = IfExistsOpt()
    view = CompoundIdentifier()
    (
        ( <RENAME> <TO> renameName = SimpleIdentifier()
        { return new SqlAlterViewRenameView(getPos(), ifExists, view, renameName); })
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

// Note: This is the Calcite implementation that doesn't match Snowflake.
SqlTruncate CalciteSqlTruncateTable(Span s) :
{
    final SqlIdentifier id;
    final boolean continueIdentity;
}
{
      <TABLE> id = CompoundIdentifier()
    (
      <CONTINUE> <IDENTITY> { continueIdentity = true; }
      |
      <RESTART> <IDENTITY> { continueIdentity = false; }
      |
      { continueIdentity = true; }
    )
    {
        return SqlDdlNodes.truncateTable(s.end(this), id, continueIdentity);
    }
}

SqlDrop SqlDropTable(Span s) :
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

SqlDrop SqlDropView(Span s) :
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

SqlDrop SqlDropSchema(Span s) :
{
    final boolean ifExists;
    final SqlIdentifier id;
}
{
    <SCHEMA>
    ifExists = IfExistsOpt()
    id = CompoundIdentifier()
    {
        return SqlDdlNodes.dropSchema(s.end(this), false, ifExists, id);
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
//     AddExpression2b(ExprContext.ACCEPT_SUB_QUERY, list)
// }

SqlNode DatePartFunctionCall() :
{
    final Span s;
    final SqlNode e;
    final SqlLiteral interval;
    final List<SqlNode> args = new ArrayList<SqlNode>();
}
{
    <DATE_PART> { s = span(); }
    <LPAREN>
    interval = SnowflakeDateTimeInterval()
    <COMMA>
    e = Expression(ExprContext.ACCEPT_SUB_QUERY) { args.add(e); }
    <RPAREN> {
        return SqlBodoParserUtil.createDatePartFunction("DATE_PART", s.end(this), interval, args);
    }
}

SqlLiteral SnowflakeDateInterval() : {
    final SqlLiteral e;
    final String s;
}
{
    (
        s = SimpleStringLiteral() {e = SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.FromString(s), getPos()); }
    |
        e = SnowflakeDateUnquotedInterval()
    )
    {
        return e;
    }
}

SqlLiteral SnowflakeTimeInterval() : {
    final SqlLiteral e;
    final String s;
}
{
    (
        s = SimpleStringLiteral() {e = SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.FromString(s), getPos()); }
    |
        e = SnowflakeTimeUnquotedInterval()
    )
    {
        return e;
    }
}

SqlLiteral SnowflakeDateTimeInterval() :
{
    final SqlLiteral e;
    final String s;
}
{
    (
        s = SimpleStringLiteral() {e = SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.FromString(s), getPos()); }
    |
        e = SnowflakeDateTimeUnquotedInterval()
    )
    {
        return e;
    }
}

/**
 * Parse an unquote Interval value that is legal
 * for DATE_PART. We convert a string literal to standardize
 * with the string implementation.
 *
 * See: https://docs.snowflake.com/sql-reference/functions-date-time#label-supported-date-time-parts
 */
SqlLiteral SnowflakeDateTimeUnquotedInterval() :
{
    final SqlLiteral e;
}
{
    (
        e = SnowflakeDateUnquotedInterval()
    |
        e = SnowflakeTimeUnquotedInterval()
    )
    {
        return e;
    }
}

SqlLiteral SnowflakeDateUnquotedInterval() : {
}{
    <YEAR> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.YEAR, getPos()); }
    | <Y> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.YEAR, getPos()); }
    | <YY> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.YEAR, getPos()); }
    | <YYY> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.YEAR, getPos()); }
    | <YYYY> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.YEAR, getPos()); }
    | <YR> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.YEAR, getPos()); }
    | <YEARS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.YEAR, getPos()); }
    | <YRS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.YEAR, getPos()); }
    | <MONTH> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MONTH, getPos()); }
    | <MM> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MONTH, getPos()); }
    | <MON> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MONTH, getPos()); }
    | <MONS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MONTH, getPos()); }
    | <MONTHS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MONTH, getPos()); }
    | <DAY> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAY, getPos()); }
    | <D> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAY, getPos()); }
    | <DD> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAY, getPos()); }
    | <DAYS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAY, getPos()); }
    | <DAYOFMONTH> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAY, getPos()); }
    | <DAYOFWEEK> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAYOFWEEK, getPos()); }
    | <WEEKDAY> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAYOFWEEK, getPos()); }
    | <DOW> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAYOFWEEK, getPos()); }
    | <DW> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAYOFWEEK, getPos()); }
    | <DAYOFWEEKISO> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAYOFWEEKISO, getPos()); }
    | <WEEKDAY_ISO> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAYOFWEEKISO, getPos()); }
    | <DOW_ISO> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAYOFWEEKISO, getPos()); }
    | <DW_ISO> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAYOFWEEKISO, getPos()); }
    | <DAYOFYEAR> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAYOFYEAR, getPos()); }
    | <YEARDAY> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAYOFYEAR, getPos()); }
    | <DOY> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAYOFYEAR, getPos()); }
    | <DY> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.DAYOFYEAR, getPos()); }
    | <WEEK> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.WEEK, getPos()); }
    | <W> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.WEEK, getPos()); }
    | <WK> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.WEEK, getPos()); }
    | <WEEKOFYEAR> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.WEEK, getPos()); }
    | <WOY> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.WEEK, getPos()); }
    | <WY> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.WEEK, getPos()); }
    | <WEEKISO> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.WEEKISO, getPos()); }
    | <WEEK_ISO> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.WEEKISO, getPos()); }
    | <WEEKOFYEARISO> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.WEEKISO, getPos()); }
    | <WEEKOFYEAR_ISO> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.WEEKISO, getPos()); }
    | <QUARTER> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.QUARTER, getPos()); }
    | <Q> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.QUARTER, getPos()); }
    | <QTR> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.QUARTER, getPos()); }
    | <QTRS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.QUARTER, getPos()); }
    | <QUARTERS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.QUARTER, getPos()); }
    | <YEAROFWEEK> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.YEAROFWEEK, getPos()); }
    | <YEAROFWEEKISO> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.YEAROFWEEKISO, getPos()); }
}

SqlLiteral SnowflakeTimeUnquotedInterval() : {
}{
    <HOUR> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.HOUR, getPos()); }
    | <H> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.HOUR, getPos()); }
    | <HH> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.HOUR, getPos()); }
    | <HR> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.HOUR, getPos()); }
    | <HOURS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.HOUR, getPos()); }
    | <HRS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.HOUR, getPos()); }
    | <MINUTE> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MINUTE, getPos()); }
    | <M> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MINUTE, getPos()); }
    | <MI> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MINUTE, getPos()); }
    | <MIN> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MINUTE, getPos()); }
    | <MINUTES> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MINUTE, getPos()); }
    | <MINS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MINUTE, getPos()); }
    | <SECOND> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.SECOND, getPos()); }
    | <S> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.SECOND, getPos()); }
    | <SEC> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.SECOND, getPos()); }
    | <SECONDS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.SECOND, getPos()); }
    | <SECS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.SECOND, getPos()); }
    | <MS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MILLISECOND, getPos()); }
    | <MSEC> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MILLISECOND, getPos()); }
    | <US> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MICROSECOND, getPos()); }
    | <USEC> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MICROSECOND, getPos()); }
    | <MICROSECOND> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MICROSECOND, getPos()); }
    | <MICROSECONDS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MICROSECOND, getPos()); }
    | <MILLISECOND> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MILLISECOND, getPos()); }
    | <MILLISECONDS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.MILLISECOND, getPos()); }
    | <NANOSECOND> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.NANOSECOND, getPos()); }
    | <NS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.NANOSECOND, getPos()); }
    | <NSEC> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.NANOSECOND, getPos()); }
    | <NANOSEC> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.NANOSECOND, getPos()); }
    | <NSECOND> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.NANOSECOND, getPos()); }
    | <NANOSECONDS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.NANOSECOND, getPos()); }
    | <NANOSECS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.NANOSECOND, getPos()); }
    | <NSECONDS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.NANOSECOND, getPos()); }
    | <EPOCH_SECOND> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.EPOCH_SECOND, getPos()); }
    | <EPOCH> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.EPOCH_SECOND, getPos()); }
    | <EPOCH_SECONDS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.EPOCH_SECOND, getPos()); }
    | <EPOCH_MILLISECOND> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.EPOCH_MILLISECOND, getPos()); }
    | <EPOCH_MILLISECONDS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.EPOCH_MILLISECOND, getPos()); }
    | <EPOCH_MICROSECOND> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.EPOCH_MICROSECOND, getPos()); }
    | <EPOCH_MICROSECONDS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.EPOCH_MICROSECOND, getPos()); }
    | <EPOCH_NANOSECOND> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.EPOCH_NANOSECOND, getPos()); }
    | <EPOCH_NANOSECONDS> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.EPOCH_NANOSECOND, getPos()); }
    | <TIMEZONE_HOUR> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.TIMEZONE_HOUR, getPos()); }
    | <TZH> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.TIMEZONE_HOUR, getPos()); }
    | <TIMEZONE_MINUTE> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.TIMEZONE_MINUTE, getPos()); }
    | <TZM> { return SqlLiteral.createSymbol(DatetimeFnUtils.DateTimePart.TIMEZONE_MINUTE, getPos()); }
}

SqlNode LastDayFunctionCall() :
{
    final Span s;
    final SqlNode e;
    final SqlLiteral interval;
    final List<SqlNode> args = new ArrayList<SqlNode>();
}
{
    <LAST_DAY> { s = span(); }
    <LPAREN>
    e = Expression(ExprContext.ACCEPT_SUB_QUERY) { args.add(e); }
(
    <COMMA>
    interval = SnowflakeDateInterval()
    <RPAREN>
|
    <RPAREN> { return SqlBodoOperatorTable.LAST_DAY.createCall(s.end(this), args); }
)
    {
        return SqlBodoParserUtil.createLastDayFunction(s.end(this), interval, args);
    }
}

/**
 * Parses a call to TIMESTAMPADD.
 * Bodo change: Allow parsing all supported snowflake units unquoted
 */
SqlCall TimestampAddFunctionCall() :
{
    final List<SqlNode> args = new ArrayList<SqlNode>();
    final Span s;
    final SqlLiteral interval;
}
{
    <DATEADD> { s = span(); }
    <LPAREN>
    (
        interval = SnowflakeDateTimeInterval() { args.add(interval); }
        <COMMA>
        AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    |
        AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    )
    <COMMA>
    AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    <RPAREN> {
        return DatetimeOperatorTable.DATEADD.createCall(
            s.end(this), args);
    }
|
    <TIMEADD> { s = span(); }
    <LPAREN>
    interval = SnowflakeDateTimeInterval() { args.add(interval); }
    <COMMA>
    AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    <COMMA>
    AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    <RPAREN> {
        return DatetimeOperatorTable.TIMEADD.createCall(
            s.end(this), args);
    }
|
    <TIMESTAMPADD> { s = span(); }
    <LPAREN>
    interval = SnowflakeDateTimeInterval() { args.add(interval); }
    <COMMA>
    AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    <COMMA>
    AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    <RPAREN> {
        return SqlBodoOperatorTable.TIMESTAMP_ADD.createCall(
            s.end(this), args);
    }
}

/**
 * Parses a call to TIMESTAMPDIFF.
 * Bodo change: allow all snowflake supported quoted/unquoted intervals
 */
SqlCall TimestampDiffFunctionCall() :
{
    final List<SqlNode> args = new ArrayList<SqlNode>();
    final Span s;
    final SqlLiteral interval;
}
{
    <DATEDIFF> { s = span(); }
    <LPAREN>
    (
      interval = SnowflakeDateTimeInterval() { args.add(interval); }
      <COMMA>
      AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    |
      AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    )
    <COMMA>
    AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    <RPAREN> {
        return DatetimeOperatorTable.DATEDIFF.createCall(
            s.end(this), args);
    }
|
    <TIMEDIFF> { s = span(); }
    <LPAREN>
    interval = SnowflakeDateTimeInterval() { args.add(interval); }
    <COMMA>
    AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    <COMMA>
    AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    <RPAREN> {
        return DatetimeOperatorTable.TIMEDIFF.createCall(
            s.end(this), args);
    }
|
    <TIMESTAMPDIFF> { s = span(); }
    <LPAREN>
    interval = SnowflakeDateTimeInterval() { args.add(interval); }
    <COMMA>
    AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    <COMMA>
    AddExpression(args, ExprContext.ACCEPT_SUB_QUERY)
    <RPAREN> {
        return SqlBodoOperatorTable.TIMESTAMP_DIFF.createCall(
            s.end(this), args);
    }
}

/**
 * Parses a call to DATE_TRUNC.
 * Bodo change: allow all snowflake supported quoted/unquoted intervals
 */
SqlCall DateTruncFunctionCall() :
{
    final List<SqlNode> args = new ArrayList<SqlNode>();
    final Span s;
    final SqlLiteral interval;
    final SqlNode e;
}
{
    <DATE_TRUNC> { s = span(); }
    <LPAREN>
    interval = SnowflakeDateTimeInterval() {  args.add(interval); }
    <COMMA>
    e = Expression(ExprContext.ACCEPT_SUB_QUERY) { args.add(e); }
    <RPAREN> {
        return DatetimeOperatorTable.DATE_TRUNC.createCall(
            s.end(this), args);
    }
}

TimeUnit BodoTimeUnit() : {
}{
    <YEAR> { return TimeUnit.YEAR; }
    | <Y> { return TimeUnit.YEAR; }
    | <YY> { return TimeUnit.YEAR; }
    | <YYY> { return TimeUnit.YEAR; }
    | <YYYY> { return TimeUnit.YEAR; }
    | <YR> { return TimeUnit.YEAR; }
    | <YEARS> { return TimeUnit.YEAR; }
    | <YRS> { return TimeUnit.YEAR; }
    | <MONTH> { return TimeUnit.MONTH; }
    | <MM> { return TimeUnit.MONTH; }
    | <MON> { return TimeUnit.MONTH; }
    | <MONS> { return TimeUnit.MONTH; }
    | <MONTHS> { return TimeUnit.MONTH; }
    | <DAY> { return TimeUnit.DAY; }
    | <D> { return TimeUnit.DAY; }
    | <DD> { return TimeUnit.DAY; }
    | <DAYS> { return TimeUnit.DAY; }
    | <DAYOFMONTH> { return TimeUnit.DAY; }
    | <WEEK> { return TimeUnit.WEEK; }
    | <W> { return TimeUnit.WEEK; }
    | <WK> { return TimeUnit.WEEK; }
    | <WEEKOFYEAR> { return TimeUnit.WEEK; }
    | <WOY> { return TimeUnit.WEEK; }
    | <WY> { return TimeUnit.WEEK; }
    | <QUARTER> { return TimeUnit.QUARTER; }
    | <Q> { return TimeUnit.QUARTER; }
    | <QTR> { return TimeUnit.QUARTER; }
    | <QTRS> { return TimeUnit.QUARTER; }
    | <QUARTERS> { return TimeUnit.QUARTER; }
    | <HOUR> { return TimeUnit.HOUR; }
    | <H> { return TimeUnit.HOUR; }
    | <HH> { return TimeUnit.HOUR; }
    | <HR> { return TimeUnit.HOUR; }
    | <HOURS> { return TimeUnit.HOUR; }
    | <HRS> { return TimeUnit.HOUR; }
    | <MINUTE> { return TimeUnit.MINUTE; }
    | <M> { return TimeUnit.MINUTE; }
    | <MI> { return TimeUnit.MINUTE; }
    | <MIN> { return TimeUnit.MINUTE; }
    | <MINUTES> { return TimeUnit.MINUTE; }
    | <MINS> { return TimeUnit.MINUTE; }
    | <SECOND> { return TimeUnit.SECOND; }
    | <S> { return TimeUnit.SECOND; }
    | <SEC> { return TimeUnit.SECOND; }
    | <SECONDS> { return TimeUnit.SECOND; }
    | <SECS> { return TimeUnit.SECOND; }
    | <MS> { return TimeUnit.MILLISECOND; }
    | <MSEC> { return TimeUnit.MILLISECOND; }
    | <MILLISECOND> { return TimeUnit.MILLISECOND; }
    | <MILLISECONDS> { return TimeUnit.MILLISECOND; }
    | <US> { return TimeUnit.MICROSECOND; }
    | <USEC> { return TimeUnit.MICROSECOND; }
    | <MICROSECOND> { return TimeUnit.MICROSECOND; }
    | <MICROSECONDS> { return TimeUnit.MICROSECOND; }
    | <NANOSECOND> { return TimeUnit.NANOSECOND; }
    | <NS> { return TimeUnit.NANOSECOND; }
    | <NSEC> { return TimeUnit.NANOSECOND; }
    | <NANOSEC> { return TimeUnit.NANOSECOND; }
    | <NSECOND> { return TimeUnit.NANOSECOND; }
    | <NANOSECONDS> { return TimeUnit.NANOSECOND; }
    | <NANOSECS> { return TimeUnit.NANOSECOND; }
    | <NSECONDS> { return TimeUnit.NANOSECOND; }
}

SqlIntervalQualifier BodoIntervalQualifier() : {
    final TimeUnit intervalUnit;
}{
    intervalUnit = BodoTimeUnit() {
        return new SqlIntervalQualifier(intervalUnit, null, getPos());
    }
}


/**
 * Copied from Calcite. Generates a call to createView.
 * NOTE: TAG and MASKING should be non-reserved, but are reserved for the time being since
 * they create a parsing ambiguity if they are non-reserved in the CREATE VIEW syntax.
 */
SqlCreate SqlCreateView(Span s, boolean replace) :
{
    final SqlIdentifier id;
    SqlNodeList columnList = null;
    final SqlNode query;
}
{
    [ <SECURE> ]
    [ [ <LOCAL> | <GLOBAL> ] ( <TEMP> | <TEMPORARY> | <VOLATILE> ) ]
    [ <RECURSIVE> ]
    <VIEW>
    IfNotExistsOpt()
    id = CompoundIdentifier()
    [ columnList = ViewColumns() ]
    (
      ( <COPY> <GRANTS> )
    |   WithTags()
    |   WithRowAccessPolicy()
    | ( <COMMENT> <EQ> StringLiteral() )
    ) *
    <AS> query = OrderedQueryOrExpr(ExprContext.ACCEPT_QUERY) {
        return SqlDdlNodes.createView(s.end(this), replace, id, columnList,
            query);
    }
}

SqlCreate SqlCreateSchema(Span s, boolean replace) :
{
    final SqlIdentifier id;
    final boolean ifNotExists;
}
{
    <SCHEMA>
    ifNotExists = IfNotExistsOpt()
    id = CompoundIdentifier()
    {
        return SqlDdlNodes.createSchema(s.end(this), replace, ifNotExists, id);
    }
}

SqlNode BodoArrayLiteral() :
{
    final List<SqlNode> list;
    SqlNode e;
    final Span s;
}
{
    <LBRACKET> { s = span(); }
    (
        e = Expression(ExprContext.ACCEPT_SUB_QUERY) { list = startList(e); }
        ( <COMMA> e = Expression(ExprContext.ACCEPT_SUB_QUERY) { list.add(e); } )*
    |
        { list = Collections.emptyList(); }
    )
    <RBRACKET> {
       return ArrayOperatorTable.ARRAY_CONSTRUCT.createCall(s.end(this), list);
    }
}


SqlNode BodoObjectLiteral() :
{
final List<SqlNode> list;
SqlNode k;
SqlNode v;
final Span s;
}
{
    <LBRACE> { s = span(); }
    (
        k = StringLiteral()
        <COLON>
        v = Expression(ExprContext.ACCEPT_SUB_QUERY)
        { list = startList(k); list.add(v); }
        (
            <COMMA>
            k = StringLiteral()
            <COLON>
            v = Expression(ExprContext.ACCEPT_SUB_QUERY)
            { list.add(k); list.add(v); }
        )*
    |
        { list = Collections.emptyList(); }
    )
    <RBRACE>
    {
        return ObjectOperatorTable.OBJECT_CONSTRUCT.createCall(s.end(this), list);
    }
}

// Transaction Queries - Just simple wrappers
SqlBegin SqlBegin() :
{
    final Span s;
}
{
    <BEGIN> { s = span(); }
    [ <WORK> | <TRANSACTION> ]
    {
        return new SqlBegin(s.end(this));
    }
}

SqlCommit SqlCommit() :
{
    final Span s;
}
{
    <COMMIT> { s = span(); }
    [ <WORK> ]
    {
        return new SqlCommit(s.end(this));
    }
}

SqlRollback SqlRollback() :
{
    final Span s;
}
{
    <ROLLBACK> { s = span(); }
    [ <WORK> ]
    {
        return new SqlRollback(s.end(this));
    }
}

// Responsible for parsing all SHOW statements.
// Will attempt to parse a TERSE option; for statements that do not support this
// it will just be set to false.

SqlShow SqlShow() :
{
    final Span s;
    final SqlShow showNode;
    boolean ifTerse;
}
{
    <SHOW>
    ifTerse = TerseOpt()
    { s = span(); }
    (
        <#list (parser.showStatementParserMethods!default.parser.showStatementParserMethods) as method>
                showNode = ${method}(s, ifTerse)
            <#sep>
                |
            </#sep>
        </#list>
    )
    {
        return showNode;
    }
}

/**
 * Parses a SHOW OBJECTS statement.
 */
SqlSnowflakeShowObjects SqlShowObjects(Span s, boolean ifTerse) :
{
   final SqlIdentifier schemaName;
}
{
    <OBJECTS> <IN>
    schemaName = CompoundIdentifier()
    {
        return new SqlSnowflakeShowObjects(s.end(schemaName), schemaName, ifTerse);
    }
}
/**
 * Parses a SHOW SCHEMAS statement.
 */
SqlSnowflakeShowSchemas SqlShowSchemas(Span s, boolean ifTerse) :
{
   final SqlIdentifier dbName;
}
{
    <SCHEMAS> <IN>
    dbName = CompoundIdentifier()
    {
        return new SqlSnowflakeShowSchemas(s.end(dbName), dbName, ifTerse);
    }
}

/**
 * Parses a SHOW TABLES statement.
 */
SqlShowTables SqlShowTables(Span s, boolean ifTerse) :
{
   final SqlIdentifier schemaName;
}
{
    <TABLES> <IN>
    schemaName = CompoundIdentifier()
    {
        return new SqlShowTables(s.end(schemaName), schemaName, ifTerse);
    }
}

/**
 * Parses a SHOW VIEWS statement.
 */
SqlShowViews SqlShowViews(Span s, boolean ifTerse) :
{
   final SqlIdentifier schemaName;
}
{
    <VIEWS> <IN>
    schemaName = CompoundIdentifier()
    {
        return new SqlShowViews(s.end(schemaName), schemaName, ifTerse);
    }
}

/**
 * Helper to parse a TERSE keyword (for SHOW statements).
 */
boolean TerseOpt() :
{
}
{
    <TERSE> { return true; }
    |
    { return false; }
}
