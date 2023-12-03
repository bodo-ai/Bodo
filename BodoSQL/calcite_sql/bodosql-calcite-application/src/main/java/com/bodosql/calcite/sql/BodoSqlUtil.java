package com.bodosql.calcite.sql;

import static org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE;
import static org.apache.calcite.util.Static.RESOURCE;

import java.math.BigDecimal;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSampleSpec;
import org.apache.calcite.sql.SqlUtil;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;

public class BodoSqlUtil {
  /** Creates a table sample spec from the parser information. */
  public static @Nullable SqlSampleSpec createTableSample(
      @NonNull SqlParserPos pos,
      boolean isBernoulli,
      BigDecimal rate,
      boolean sampleByRows,
      boolean isRepeatable,
      int repeatableSeed) {
    if (rate == null) {
      return null;
    }

    if (sampleByRows) {
      final BigDecimal maxRows = BigDecimal.valueOf(1000000L);
      if (rate.compareTo(BigDecimal.ZERO) < 0 || rate.compareTo(maxRows) > 0) {
        throw SqlUtil.newContextException(
            pos, BODO_SQL_RESOURCE.invalidSampleSize(BigDecimal.ZERO, maxRows));
      }

      return isRepeatable
          ? SqlTableSampleRowLimitSpec.createTableSample(isBernoulli, rate, repeatableSeed)
          : SqlTableSampleRowLimitSpec.createTableSample(isBernoulli, rate);
    }

    final BigDecimal oneHundred = BigDecimal.valueOf(100L);
    if (rate.compareTo(BigDecimal.ZERO) < 0 || rate.compareTo(oneHundred) > 0) {
      throw SqlUtil.newContextException(
          pos, BODO_SQL_RESOURCE.invalidSampleSize(BigDecimal.ZERO, oneHundred));
    }

    BigDecimal samplePercentage = rate.divide(oneHundred);
    if (samplePercentage.compareTo(BigDecimal.ZERO) < 0
        || samplePercentage.compareTo(BigDecimal.ONE) > 0) {
      throw SqlUtil.newContextException(pos, RESOURCE.invalidSampleSize());
    }
    return isRepeatable
        ? SqlSampleSpec.createTableSample(isBernoulli, samplePercentage, repeatableSeed)
        : SqlSampleSpec.createTableSample(isBernoulli, samplePercentage);
  }

  /**
   * Return if this OP represents a SQL default value.
   *
   * @param op The Sql operator.
   * @return Is this SqlStdOperatorTable.DEFAULT
   */
  public static boolean isDefaultCall(SqlOperator op) {
    return op == SqlStdOperatorTable.DEFAULT;
  }

  /**
   * Return if this SqlNode represents a SQL default value.
   *
   * @param node The SqlNode.
   * @return Is this call the default?
   */
  public static boolean isDefaultCall(SqlNode node) {
    if (node instanceof SqlCall) {
      return isDefaultCall(((SqlCall) node).getOperator());
    }
    return false;
  }

  /**
   * Return if this RexNode represents a SQL default value.
   *
   * @param node The RexNode.
   * @return Is this call the default?
   */
  public static boolean isDefaultCall(RexNode node) {
    if (node instanceof RexCall) {
      return isDefaultCall(((RexCall) node).getOperator());
    }
    return false;
  }
}
