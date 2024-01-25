package com.bodosql.calcite.schema;

import com.google.common.collect.ImmutableList;
import java.util.List;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexNode;
import org.checkerframework.checker.nullness.qual.NonNull;

/**
 * Interface that defines the API used to "inline" or "expand" the contents of a function body. This
 * is intended to be used for inlining UDFs that are defined with a known function body.
 */
public interface FunctionExpander {

  /**
   * Inline the body of a function. This API is responsible for parsing the function body as a query
   * and validating the contents query for type stability.
   *
   * <p>This API is still under active development, so the return type is not yet finalized and
   * additional arguments are likely to be added.
   *
   * @param functionBody Body of the function.
   * @param functionPath Path of the function.
   * @param paramNames The name of the function parameters.
   * @param arguments The RexNode argument inputs to the function.
   * @param correlatedArguments The arguments after replacing any input references with field
   *     accesses to a correlated variable.
   * @param returnType The expected function return type.
   * @param cluster The cluster used for generating Rex/RelNodes. This is shared with the caller so
   *     correlation ids are consistently updated.
   * @return The body of the function as a RexNode. This should either be a scalar sub-query or a
   *     simple expression.
   */
  RexNode expandFunction(
      @NonNull String functionBody,
      @NonNull ImmutableList<@NonNull String> functionPath,
      @NonNull List<@NonNull String> paramNames,
      @NonNull List<@NonNull RexNode> arguments,
      @NonNull List<@NonNull RexNode> correlatedArguments,
      @NonNull RelDataType returnType,
      @NonNull RelOptCluster cluster);
}
