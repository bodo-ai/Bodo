package com.bodosql.calcite.sql;

import java.math.BigDecimal;
import org.apache.calcite.sql.SqlSampleSpec;

/** Sample specification for row based sampling. */
public class SqlTableSampleRowLimitSpec extends SqlSampleSpec {
  private final boolean isBernoulli;
  private final BigDecimal numberOfRows;
  private final boolean isRepeatable;
  private final int repeatableSeed;

  SqlTableSampleRowLimitSpec(boolean isBernoulli, BigDecimal numberOfRows) {
    this.isBernoulli = isBernoulli;
    this.numberOfRows = numberOfRows;
    this.isRepeatable = false;
    this.repeatableSeed = 0;
  }

  SqlTableSampleRowLimitSpec(boolean isBernoulli, BigDecimal numberOfRows, int repeatableSeed) {
    this.isBernoulli = isBernoulli;
    this.numberOfRows = numberOfRows;
    this.isRepeatable = true;
    this.repeatableSeed = repeatableSeed;
  }

  /** Indicates Bernoulli vs. System sampling. */
  public boolean isBernoulli() {
    return isBernoulli;
  }

  /** Returns number of rows to sample. Must be zero or positive. */
  public BigDecimal getNumberOfRows() {
    return numberOfRows;
  }

  /** Indicates whether repeatable seed should be used. */
  public boolean isRepeatable() {
    return isRepeatable;
  }

  /** Seed to produce repeatable samples. */
  public int getRepeatableSeed() {
    return repeatableSeed;
  }

  @Override
  public String toString() {
    StringBuilder b = new StringBuilder();
    b.append(isBernoulli ? "BERNOULLI" : "SYSTEM");
    b.append('(');
    b.append(numberOfRows);
    b.append(" ROWS)");

    if (isRepeatable) {
      b.append(" REPEATABLE(");
      b.append(repeatableSeed);
      b.append(')');
    }
    return b.toString();
  }

  /**
   * Creates a table sample with a row limit without repeatability.
   *
   * @param isBernoulli true if Bernoulli style sampling is to be used; false for implementation
   *     specific sampling
   * @param numberOfRows number of rows to sample from the table
   */
  public static SqlSampleSpec createTableSample(boolean isBernoulli, BigDecimal numberOfRows) {
    return new SqlTableSampleRowLimitSpec(isBernoulli, numberOfRows);
  }

  /**
   * Creates a table sample with repeatability.
   *
   * @param isBernoulli true if Bernoulli style sampling is to be used; false for implementation
   *     specific sampling
   * @param numberOfRows number of rows to sample from the table
   * @param repeatableSeed seed value used to reproduce the same sample
   */
  public static SqlSampleSpec createTableSample(
      boolean isBernoulli, BigDecimal numberOfRows, int repeatableSeed) {
    return new SqlTableSampleRowLimitSpec(isBernoulli, numberOfRows, repeatableSeed);
  }
}
