package com.bodosql.calcite.traits;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTrait;
import org.apache.calcite.plan.RelTraitDef;

public class BatchingProperty implements RelTrait {

  public static BatchingProperty SINGLE_BATCH = new BatchingProperty(Type.SINGLE_BATCH);
  public static BatchingProperty STREAMING = new BatchingProperty(Type.STREAMING);

  private final Type type;

  /** Default value used for internal testing. */
  public static final int defaultBatchSize = 4000;

  @Override
  public RelTraitDef getTraitDef() {
    return BatchingPropertyTraitDef.INSTANCE;
  }

  @Override
  public String toString() {
    if (type.equals(Type.SINGLE_BATCH)) {
      return "SINGLE BATCH DATA";
    } else {
      return "STREAMING DATA";
    }
  }

  @Override
  public boolean satisfies(RelTrait toTrait) {
    BatchingProperty toBatching = (BatchingProperty) toTrait;
    return this.type.equals(toBatching.type);
  }

  @Override
  public void register(RelOptPlanner relOptPlanner) {}

  private BatchingProperty(Type type) {
    this.type = type;
  }

  enum Type {
    SINGLE_BATCH,
    STREAMING
  }
}
