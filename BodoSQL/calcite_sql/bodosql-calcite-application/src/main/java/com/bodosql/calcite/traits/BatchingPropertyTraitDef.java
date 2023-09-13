package com.bodosql.calcite.traits;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;

public class BatchingPropertyTraitDef extends RelTraitDef<BatchingProperty> {
  public static BatchingPropertyTraitDef INSTANCE = new BatchingPropertyTraitDef();

  @Override
  public Class<BatchingProperty> getTraitClass() {
    return BatchingProperty.class;
  }

  @Override
  public String getSimpleName() {
    return "BATCHING_PROPERTY";
  }

  @Override
  public RelNode convert(
      RelOptPlanner planner,
      RelNode rel,
      BatchingProperty toTrait,
      boolean allowInfiniteCostConverters) {
    BatchingProperty fromTrait = rel.getTraitSet().getTrait(BatchingPropertyTraitDef.INSTANCE);
    if (fromTrait.satisfies(toTrait)) {
      return rel;
    }
    if (toTrait.satisfies(BatchingProperty.SINGLE_BATCH)) {
      // If converting to a single batch create a combine streams
      return new CombineStreamsExchange(
          rel.getCluster(), rel.getTraitSet().replace(BatchingProperty.SINGLE_BATCH), rel);
    } else {
      return new SeparateStreamExchange(
          rel.getCluster(), rel.getTraitSet().replace(BatchingProperty.STREAMING), rel);
    }
  }

  @Override
  public boolean canConvert(
      RelOptPlanner planner, BatchingProperty fromTrait, BatchingProperty toTrait) {
    return true;
  }

  @Override
  public void registerConverterRule(RelOptPlanner planner, ConverterRule converterRule) {}

  @Override
  public BatchingProperty getDefault() {
    return BatchingProperty.NONE;
  }
}
