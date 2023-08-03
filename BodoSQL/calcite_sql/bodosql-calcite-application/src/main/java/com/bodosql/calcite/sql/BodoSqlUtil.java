package com.bodosql.calcite.sql;

import org.apache.calcite.sql.SqlNumericLiteral;
import org.apache.calcite.sql.SqlSampleSpec;
import org.apache.calcite.sql.SqlUtil;
import org.apache.calcite.sql.parser.SqlParserPos;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;

import java.math.BigDecimal;

import static org.apache.calcite.util.Static.RESOURCE;

public class BodoSqlUtil {
  /**
   * Creates a table sample spec from the parser information.
   */
  public static @Nullable SqlSampleSpec createTableSample(@NonNull SqlParserPos pos,
      boolean isBernoulli, @NonNull SqlNumericLiteral sampleRate, boolean sampleByRows,
      boolean isRepeatable, int repeatableSeed) {
    BigDecimal rate = sampleRate.bigDecimalValue();
    if (rate == null) {
      return null;
    }

    if (sampleByRows) {
      final BigDecimal maxRows = BigDecimal.valueOf(1000000L);
      if (rate.compareTo(BigDecimal.ZERO) < 0 || rate.compareTo(maxRows) > 0) {
        throw SqlUtil.newContextException(
            pos, RESOURCE.invalidSampleSize(BigDecimal.ZERO, maxRows));
      }

      return isRepeatable
          ? SqlTableSampleRowLimitSpec.createTableSample(
          isBernoulli, rate, repeatableSeed)
          : SqlTableSampleRowLimitSpec.createTableSample(isBernoulli, rate);
    }

    final BigDecimal oneHundred = BigDecimal.valueOf(100L);
    if (rate.compareTo(BigDecimal.ZERO) < 0
        || rate.compareTo(oneHundred) > 0) {
      throw SqlUtil.newContextException(
          pos, RESOURCE.invalidSampleSize(BigDecimal.ZERO, oneHundred));
    }

    // Treat TABLESAMPLE(0) and TABLESAMPLE(100) as no table
    // sampling at all.  Not strictly correct: TABLESAMPLE(0)
    // should produce no output, but it simplifies implementation
    // to know that some amount of sampling will occur.
    // In practice values less than ~1E-43% are treated as 0.0 and
    // values greater than ~99.999997% are treated as 1.0
    float fRate = rate.divide(oneHundred).floatValue();
    if (fRate <= 0.0f || fRate >= 1.0f) {
      return null;
    }
    return isRepeatable
        ? SqlSampleSpec.createTableSample(
        isBernoulli, fRate, repeatableSeed)
        : SqlSampleSpec.createTableSample(isBernoulli, fRate);
  }
}
