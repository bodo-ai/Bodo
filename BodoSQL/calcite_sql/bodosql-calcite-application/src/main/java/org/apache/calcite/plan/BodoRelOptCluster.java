package org.apache.calcite.plan;

import com.bodosql.calcite.rel.metadata.BodoRelMetadataQuery;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;

/**
 * Bodo's extension to RelOptCluster with specific configuration details
 * and additional state tracking.
 *
 * Note this class is in Java so we can shadow RelOptCluster.create, which is
 * not possible in Kotlin.
 */
public class BodoRelOptCluster extends RelOptCluster {
    private final AtomicInteger nextJoinId;
    private final AtomicInteger nextCacheId;


    BodoRelOptCluster(RelOptPlanner planner, RelDataTypeFactory typeFactory,
                      RexBuilder rexBuilder, AtomicInteger nextCorrel,
                      Map<String, RelNode> mapCorrelToRel, AtomicInteger nextJoinId, AtomicInteger nextCacheId) {
        super(planner, typeFactory, rexBuilder, nextCorrel, mapCorrelToRel);
        setMetadataQuerySupplier(BodoRelMetadataQuery.getSupplier());
        this.nextJoinId = nextJoinId;
        this.nextCacheId = nextCacheId;
    }

    public int nextJoinId() {
        return nextJoinId.getAndIncrement();
    }

    public int nextCacheId() {
        return nextCacheId.getAndIncrement();
    }

    public static BodoRelOptCluster create(RelOptPlanner planner,
                                       RexBuilder rexBuilder) {
        return new BodoRelOptCluster(planner, rexBuilder.getTypeFactory(),
                rexBuilder, new AtomicInteger(0), new HashMap<>(), new AtomicInteger(0), new AtomicInteger(0));
    }
}
