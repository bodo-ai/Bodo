package com.bodosql.calcite.sql.parser;

import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParserPos;

import com.bodosql.calcite.sql.fun.SqlNamedParameterOperator;

public abstract class SqlAbstractParserImpl extends org.apache.calcite.sql.parser.SqlAbstractParserImpl {
  protected SqlNode makeNamedParam(String name, String prefix, SqlParserPos pos) {
    return SqlNamedParameterOperator.INSTANCE.createCall(
        pos,
        SqlLiteral.createCharString(name, pos),
        SqlLiteral.createCharString(prefix, pos)
    );
  }
}
