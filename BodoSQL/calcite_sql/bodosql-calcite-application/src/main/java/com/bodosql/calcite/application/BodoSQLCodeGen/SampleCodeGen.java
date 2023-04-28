package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.getBodoIndent;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import org.apache.calcite.plan.RelOptRowSamplingParameters;
import org.apache.calcite.plan.RelOptSamplingParameters;

/**
 * Class that returns the generated code for Sample operations after all inputs have been visited
 */
public class SampleCodeGen {

  /**
   * Function that returns the necessary generated code for a Sample expression.
   *
   * @param outVar The output variable.
   * @param expr The expression of the input table
   * @param params Parameters of the Sample Relnode
   * @return The code generated for the Sample expression.
   */
  public static String generateSampleCode(
      String outVar, String expr, RelOptSamplingParameters params) {
    if (!params.isBernoulli()) {
      throw new BodoSQLCodegenException("Error: SYSTEM/BLOCK sampling is not yet supported");
    }

    StringBuilder sampleBuilder = new StringBuilder();

    sampleBuilder
        .append(getBodoIndent())
        .append(outVar)
        .append(" = ")
        .append(expr)
        .append(".sample(frac=")
        .append(params.getSamplingPercentage());

    if (params.isRepeatable()) {
      sampleBuilder.append(", random_state=").append(params.getRepeatableSeed());
    }

    sampleBuilder.append(")\n");

    return sampleBuilder.toString();
  }

  /**
   * Function that returns the necessary generated code for a RowSample expression.
   *
   * @param outVar The output variable.
   * @param expr The expression of the input table
   * @param params Parameters of the RowSample Relnode
   * @return The code generated for the RowSample expression.
   */
  public static String generateRowSampleCode(
      String outVar, String expr, RelOptRowSamplingParameters params) {
    if (!params.isBernoulli()) {
      throw new BodoSQLCodegenException("Error: SYSTEM/BLOCK sampling is not yet supported");
    }

    StringBuilder rowSampleBuilder = new StringBuilder();

    rowSampleBuilder
        .append(getBodoIndent())
        .append(outVar)
        .append(" = ")
        .append(expr)
        .append(".sample(n=")
        .append(params.getNumberOfRows());

    if (params.isRepeatable()) {
      rowSampleBuilder.append(", random_state=").append(params.getRepeatableSeed());
    }

    rowSampleBuilder.append(")\n");

    return rowSampleBuilder.toString();
  }
}
