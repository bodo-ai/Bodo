package com.bodosql.calcite.application;

import java.util.HashMap;
import java.util.Map;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelVisitor;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * Create a mapping from Calcite's RelNode.id to a compact ID space. This ID will be used by bodo
 * for associating generated code with the source SQL nodes (e.g. for query profiling) Note that we
 * can't use the RelNode id's directly since Calcite uses an increasing counter to vend IDs for the
 * lifetime of the compiler. This would make tests sensitive to the order in which they run and
 * potentially have ill effects on caching.
 */
public class RelIDToOperatorIDVisitor extends RelVisitor {
  private final HashMap<Integer, Integer> relNodeIDToOperatorID = new HashMap<>();
  private int counter = 0;

  public Map<Integer, Integer> getIDMapping() {
    return relNodeIDToOperatorID;
  }

  @Override
  public void visit(RelNode node, int ordinal, @Nullable RelNode parent) {
    node.childrenAccept(this);
    int relNodeID = node.getId();
    if (!relNodeIDToOperatorID.containsKey(relNodeID)) {
      counter++;
      relNodeIDToOperatorID.put(relNodeID, counter);
    }
  }
}
