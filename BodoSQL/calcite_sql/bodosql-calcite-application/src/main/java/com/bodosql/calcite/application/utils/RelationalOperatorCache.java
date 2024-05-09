package com.bodosql.calcite.application.utils;

import com.bodosql.calcite.adapter.pandas.PandasRel;
import com.bodosql.calcite.ir.BodoEngineTable;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Module;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.OperatorType;
import com.bodosql.calcite.ir.StreamingPipelineFrame;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.traits.BatchingProperty;
import java.util.HashMap;
import java.util.List;
import kotlin.Pair;

/**
 * Class that handles caching the output of each relational operator (RelNode) if possible during
 * codegen. Whenever possible, we cache the outputs of nodes to avoid redundant computation. The set
 * of nodes for which we support caching is currently only a subset of all nodes (PandasRels).
 *
 * <p>Because the logical plan is a tree, Nodes that are at the bottom of the tree can be repeated,
 * even if they are identical. When calcite produces identical nodes, they (generally) use the same
 * node ID. Some false positives can occur as a result of optimization, but we do a pass to reassign
 * the same nodes the same node ID (see com.bodosql.calcite.prepare.BodoPrograms.MergeRelProgram).
 */
public class RelationalOperatorCache {

  // Maps each PandasRel to it's output based on its RelNode ID
  // for non-streaming PandasRels
  private final HashMap<Integer, BodoEngineTable> tableCacheMap = new HashMap<>();

  // Maps each PandasRel to it's output, and the pipeline in which it exists based on its RelNode ID
  // This is used for streaming PandasRels
  private final HashMap<Integer, Pair<BodoEngineTable, StreamingPipelineFrame>>
      streamingTableCacheMap = new HashMap();

  // The module builder
  // The RelationalOperatorCache may update the generated code in order to handle
  // more complex caching (specifically, caching streaming outputs)
  private final Module.Builder builder;

  public RelationalOperatorCache(Module.Builder builder) {
    this.builder = builder;
  }

  /**
   * Determine if we have already visited this node and can reuse its output. We check if a node is
   * cached by checking if its id is stored in the cache.
   *
   * @param node the node that may be cached
   * @return If the node is cached
   */
  public boolean isNodeCached(PandasRel node, int key) {
    // Note that we can directly check the ID due to merging relNodes during
    // an optimization step.
    // see com.bodosql.calcite.prepare.BodoPrograms.MergeRelProgram
    return node.getTraitSet().contains(BatchingProperty.STREAMING)
        ? streamingTableCacheMap.containsKey(key)
        : tableCacheMap.containsKey(key);
  }

  /**
   * @brief Get the cached value for a streaming node
   * @param node streaming node
   * @param key the cache key
   * @return a BodoEngineTable if the node has been cached, and a runtime exception if it has not
   *     been.
   */
  private BodoEngineTable getCachedStreamingTable(PandasRel node, int key) {
    assert node.getTraitSet().contains(BatchingProperty.STREAMING);
    // If the node is streaming, we need to do some additional work.
    Pair<BodoEngineTable, StreamingPipelineFrame> tableAndFrame =
        this.streamingTableCacheMap.get(key);
    if (tableAndFrame == null) {
      throw new RuntimeException(
          "Error in OperatorCache.getCachedTable: Argument table is not cached.");
    }

    BodoEngineTable table = tableAndFrame.getFirst();
    StreamingPipelineFrame frame = tableAndFrame.getSecond();
    Variable tableBuilderState = builder.getSymbolTable().genStateVar();

    int operatorID = this.builder.newOperatorID(node);

    frame.initializeStreamingState(
        operatorID,
        new Op.Assign(
            tableBuilderState,
            new Expr.Call(
                "bodo.libs.table_builder.init_table_builder_state",
                List.of(new Expr.IntegerLiteral(operatorID)),
                List.of(new Pair<>("use_chunked_builder", Expr.True.INSTANCE)))),
        OperatorType.ACCUMULATE_TABLE,
        // ChunkedTableBuilder doesn't need pinned memory budget
        0);

    frame.add(
        new Op.Stmt(
            new Expr.Call(
                "bodo.libs.table_builder.table_builder_append",
                List.of(tableBuilderState, table))));

    // Note: The node that is lowest on the tree starts the streaming pipeline,
    // since the cached tree must contain the node responsible for starting the streaming
    // pipeline,
    // we have to start it here.

    Variable exitCond = builder.getSymbolTable().genFinishedStreamingFlag();
    Variable iterVar = builder.getSymbolTable().genIterVar();
    Variable newTableVar = builder.getSymbolTable().genTableVar();
    builder.startStreamingPipelineFrame(exitCond, iterVar);
    builder.add(
        new Op.TupleAssign(
            List.of(newTableVar, exitCond),
            new Expr.Call(
                "bodo.libs.table_builder.table_builder_pop_chunk", List.of(tableBuilderState))));
    builder
        .getCurrentStreamingPipeline()
        .deleteStreamingState(
            operatorID,
            new Op.Stmt(
                new Expr.Call(
                    "bodo.libs.table_builder.delete_table_builder_state",
                    List.of(tableBuilderState))));

    BodoEngineTable outputTable = new BodoEngineTable(newTableVar.emit(), node);
    // Update the streaming cache map, so that subsequent
    // caches use this pipeline, instead of the original.
    // This lets us "daisy chain" table buffers, reducing peak memory usage.
    streamingTableCacheMap.put(key, new Pair<>(outputTable, builder.getCurrentStreamingPipeline()));
    return outputTable;
  }

  /**
   * @brief Get the cached value for a non streaming node
   * @param node streaming node
   * @param key the cache key
   * @return a BodoEngineTable if the node has been cached, and a runtime exception if it has not
   *     been.
   */
  private BodoEngineTable getCachedNonStreamingTable(PandasRel node, int key) {
    assert node.getTraitSet().containsIfApplicable(BatchingProperty.SINGLE_BATCH);
    if (!tableCacheMap.containsKey(key)) {
      throw new RuntimeException(
          "Error in OperatorCache.getCachedTable: Argument table is not cached.");
    }
    return tableCacheMap.get(key);
  }

  /**
   * @param node The node to cache
   * @param key The cache key
   * @return a BodoEngineTable if the node has been cached, and a runtime exception if it has not
   *     been.
   */
  public BodoEngineTable getCachedTable(PandasRel node, int key) {

    if (node.getTraitSet().contains(BatchingProperty.STREAMING)) {
      return getCachedStreamingTable(node, key);
    } else {
      return getCachedNonStreamingTable(node, key);
    }
  }

  /**
   * Adds the specified streaming node to the cache if possible. No-op if it is not valid to cache
   * the node's output.
   *
   * @param node The node to cache
   * @param key The cache key
   * @param table The output value to cache
   */
  private void tryCacheNodeStreaming(PandasRel node, int key, BodoEngineTable table) {
    assert node.getTraitSet().contains(BatchingProperty.STREAMING);
    this.streamingTableCacheMap.put(key, new Pair<>(table, builder.getCurrentStreamingPipeline()));
  }

  /**
   * Adds the specified non streaming node to the cache if possible. No-op if it is not valid to
   * cache the node's output.
   *
   * @param node The node to cache
   * @param key The cache key
   * @param table The output value to cache
   */
  private void tryCacheNodeNonStreaming(PandasRel node, int key, BodoEngineTable table) {
    assert node.getTraitSet().containsIfApplicable(BatchingProperty.SINGLE_BATCH);
    this.tableCacheMap.put(key, table);
  }

  /**
   * Adds the specified node to the cache if possible. No-op if it is not valid to cache the node's
   * output.
   *
   * @param node The node to cache
   * @param key The cache key
   * @param table The output value to cache
   */
  public void tryCacheNode(PandasRel node, int key, BodoEngineTable table) {
    if (canCacheOutput(node)) {
      if (node.getTraitSet().contains(BatchingProperty.STREAMING)) {
        tryCacheNodeStreaming(node, key, table);
      } else {
        tryCacheNodeNonStreaming(node, key, table);
      }
    }
  }

  /**
   * Checks if it is valid to cache the output of the current node.
   *
   * @param node Node to check
   * @return If it is valid to cache the output of the current node.
   */
  public static boolean canCacheOutput(PandasRel node) {
    // pandas rels have a built-in function that
    // determines if they should/shouldn't be cached
    return node.canUseNodeCache();
  }
}
