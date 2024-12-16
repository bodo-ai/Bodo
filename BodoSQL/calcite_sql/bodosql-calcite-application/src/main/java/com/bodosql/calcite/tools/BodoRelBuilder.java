package com.bodosql.calcite.tools;

import org.apache.calcite.plan.Context;
import org.apache.calcite.plan.Contexts;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptSchema;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * Extension of {@link RelBuilder} that allows for additional functionality. We use a Java class
 * here because Kotlin can't shadow static objects/methods, and we want to avoid errors with proto.
 */
public class BodoRelBuilder extends RelBuilder {
  protected BodoRelBuilder(
      @Nullable Context context, RelOptCluster cluster, @Nullable RelOptSchema relOptSchema) {
    super(context, cluster, relOptSchema);
  }

  /**
   * Creates a {@link RelBuilderFactory}, a partially-created RelBuilder. Just add a {@link
   * RelOptCluster} and a {@link RelOptSchema}
   */
  public static RelBuilderFactory proto(final Context context) {
    return (cluster, schema) -> new BodoRelBuilder(context, cluster, schema);
  }

  /** Creates a {@link RelBuilderFactory} that uses a given set of factories. */
  public static RelBuilderFactory proto(Object... factories) {
    return proto(Contexts.of(factories));
  }
}
