package com.bodosql.calcite.application.utils;

import com.bodosql.calcite.adapter.pandas.PandasRel;
import com.bodosql.calcite.adapter.snowflake.SnowflakeRel;
import com.bodosql.calcite.adapter.snowflake.SnowflakeToPandasConverter;
import com.bodosql.calcite.ir.BodoEngineTable;
import com.bodosql.calcite.traits.BatchingProperty;
import java.util.HashMap;
import org.apache.calcite.rel.RelNode;

/**
 * Class that handles caching the output of each relational operator (RelNode) if possible during
 * codegen. Whenever possible, we cache the outputs of nodes to avoid redundant computation. The set
 * of nodes for which we support caching is currently only a subset of all nodes.
 *
 * <p>Because the logical plan is a tree, Nodes that are at the bottom of the tree can be repeated,
 * even if they are identical. When calcite produces identical nodes, they (generally) use the same
 * node ID. Some false positives can occur as a result of optimization, but we do a pass to reassign
 * the same nodes the same node ID (see com.bodosql.calcite.prepare.BodoPrograms.MergeRelProgram).
 */
public class RelationalOperatorCache {

  // Maps each RelNode to it's output based on its RelNode ID
  private final HashMap<Integer, BodoEngineTable> tableCacheMap = new HashMap<>();

  /**
   * Determine if we have already visited this node and can reuse its output. We check if a node is
   * cached by checking if its id is stored in the cache.
   *
   * @param node the node that may be cached
   * @return If the node is cached
   */
  public boolean isNodeCached(RelNode node) {
    // Note that we can directly check the ID due to merging relNodes during
    // an optimization step.
    // see com.bodosql.calcite.prepare.BodoPrograms.MergeRelProgram
    return tableCacheMap.containsKey(node.getId());
  }

  /**
   * @param node
   * @return a BodoEngineTable if the node has been cached, and null otherwise
   */
  public BodoEngineTable getCachedTable(RelNode node) {
    if (!tableCacheMap.containsKey(node.getId())) {
      throw new RuntimeException(
          "Error in OperatorCache.getCachedTable: Argument table is not cached.");
    }
    return tableCacheMap.get(node.getId());
  }

  /**
   * Adds the specified node to the cache if possible. No-op if it is not valid to cache the node's
   * output.
   *
   * @param node The node to cache
   * @param table The output value to cache
   */
  public void tryCacheNode(RelNode node, BodoEngineTable table) {
    if (canCacheOutput(node)) {
      this.tableCacheMap.put(node.getId(), table);
    }
  }

  /**
   * Checks if it is valid to cache the output of the current node.
   *
   * @param node Node to check
   * @return If it is valid to cache the output of the current node.
   */
  public static boolean canCacheOutput(RelNode node) {

    if (node instanceof PandasRel) {
      // pandas rels have a built-in function that
      // determines if they should/shouldn't be cached
      return ((PandasRel) node).canUseNodeCache();
    } else if (node instanceof SnowflakeRel) {

      // This is a sanity check.
      // In the codegen visitor, we should never visit any snowflake rel other than
      // SnowflakeToPandasConverter.
      // Every child of SnowflakeToPandasConverter just gets converted into the SQL string that we
      // send
      // to snowflake. Since we shouldn't even visit another snowflake rel, something has gone
      // terribly wrong
      // if we're even trying to cache a snowflake Rel, so throw an explicit exception.
      assert (node instanceof SnowflakeToPandasConverter)
          : "Internal error in canCacheOutput: Visited a non SnowflakeToPandasConverter node";
      return !node.getTraitSet().contains(BatchingProperty.STREAMING);
    } else {
      throw new RuntimeException(
          "Internal error in TableCache.canCacheNode: Encountered an unsupported Rel type");
    }
  }
}
