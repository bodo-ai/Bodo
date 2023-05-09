package com.bodosql.calcite.application.Utils;

import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeUtil;

public class TypeEquivalentSimplifier {

  /**
   * Returns an SQLNode that has equivalent types to the original SqlNode/Type.
   *
   * <p>For example, if passed SELECT * from table and RecordType(BIGINT A, VARCHAR B, DATE C),
   *
   * <p>the returned expression would be:
   *
   * <p>SELECT null::BIGINT as A, null::varchar as B, null::DATE as C
   *
   * @param node The SqlNode
   * @param type The Type of the input SqlNode
   * @return The SqlNode that returns equivalent types to the input SqlNode
   */
  public static SqlNode reduceToSimpleSqlNode(SqlNode node, RelDataType type) {
    if (node instanceof SqlSelect || node instanceof SqlWith) {
      return reduceToSimpleSelect(type);
    } else {
      throw new RuntimeException("SqlNode not supported");
    }
  }

  /**
   * @param node Node to be casted/aliased
   * @param type Type to cast to
   * @param name Alias to assign
   * @return The input node, casted to the specified type, and assigned the given alias
   */
  public static SqlCall castToTypeWithAlias(SqlNode node, RelDataType type, String name) {
    SqlCall castedNode =
        SqlStdOperatorTable.CAST.createCall(
            SqlParserPos.ZERO, node, SqlTypeUtil.convertTypeToSpec(type));
    SqlCall aliasedNode =
        SqlStdOperatorTable.AS.createCall(
            SqlParserPos.ZERO, castedNode, new SqlIdentifier(name, SqlParserPos.ZERO));

    return aliasedNode;
  }

  /**
   * Returns a SELECT that returns the exact same fields as the input type. Requires that the type
   * has a non-null fields list.
   *
   * <p>For example, if the type passed is RecordType(BIGINT A, VARCHAR B, DATE C), the returned
   * expression would be:
   *
   * <p>SELECT null::BIGINT as A, null::varchar as B, null::DATE as C
   *
   * @param type The specified type
   * @return
   */
  public static SqlSelect reduceToSimpleSelect(RelDataType type) {
    SqlNodeList newNodeList = SqlNodeList.EMPTY.clone(SqlParserPos.ZERO);
    List<RelDataTypeField> fields = type.getFieldList();
    SqlNode nullLiteral = SqlLiteral.createNull(SqlParserPos.ZERO);
    for (int i = 0; i < type.getFieldCount(); i++) {
      RelDataTypeField field = fields.get(i);
      newNodeList.add(castToTypeWithAlias(nullLiteral, field.getType(), field.getName()));
    }

    return new SqlSelect(
        SqlParserPos.ZERO,
        null,
        newNodeList,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null);
  }
}
