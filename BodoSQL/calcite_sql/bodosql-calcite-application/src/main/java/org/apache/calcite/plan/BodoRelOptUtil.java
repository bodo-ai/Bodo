package org.apache.calcite.plan;

import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.fun.SqlInOperator;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql2rel.SqlToRelConverter;

import com.bodosql.calcite.sql.fun.SqlBodoOperatorTable;

import java.util.List;

public class BodoRelOptUtil {
  /**
   * Returns {@link SqlOperator} for given {@link SqlKind} or returns {@code operator}
   * when {@link SqlKind} is not known.
   * <p>
   * TODO(jsternberg): This method is mostly identical to {@link RelOptUtil#op}
   * but includes support for {@link SqlBodoOperatorTable#NULL_EQUALS}.
   * <p>
   * The proper way to support this would be to give NULL_EQUALS its own
   * custom {@link org.apache.calcite.sql.fun.SqlQuantifyOperator} so we
   * don't have to modify the upstream version and then use convertlets
   * to perform the conversion similar to {@link com.bodosql.calcite.sql.fun.SqlLikeQuantifyOperator}.
   * That's not necessarily straightforward and easy right now so copying
   * this method and utilizing it in {@link SqlToRelConverter} is more
   * straightforward.
   *
   * @param kind input kind
   * @param operator default operator value
   * @return SqlOperator for the given kind
   * @see RexUtil#op(SqlKind)
   */
  public static SqlOperator op(SqlKind kind, SqlOperator operator) {
    switch (kind) {
    case EQUALS:
      return SqlStdOperatorTable.EQUALS;
    case NOT_EQUALS:
      return SqlStdOperatorTable.NOT_EQUALS;
    case NULL_EQUALS:
      return SqlBodoOperatorTable.NULL_EQUALS;
    case GREATER_THAN:
      return SqlStdOperatorTable.GREATER_THAN;
    case GREATER_THAN_OR_EQUAL:
      return SqlStdOperatorTable.GREATER_THAN_OR_EQUAL;
    case LESS_THAN:
      return SqlStdOperatorTable.LESS_THAN;
    case LESS_THAN_OR_EQUAL:
      return SqlStdOperatorTable.LESS_THAN_OR_EQUAL;
    case IS_DISTINCT_FROM:
      return SqlStdOperatorTable.IS_DISTINCT_FROM;
    case IS_NOT_DISTINCT_FROM:
      return SqlStdOperatorTable.IS_NOT_DISTINCT_FROM;
    default:
      return operator;
    }
  }
}
