package com.bodosql.calcite.application.utils;

import com.bodosql.calcite.ir.BodoEngineTable;
import com.bodosql.calcite.ir.StateVariable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import kotlin.Pair;

/**
 * Class that handles state information for cache nodes. This tracks information for mapping tables
 * + state to various consumers. The BodoPhysicalCachedSubPlan is responsible for generating the
 * corresponding code.
 */
public class RelationalOperatorCache {

  // Maps each cacheID to the original BodoEngineTable
  private final Map<Integer, BodoEngineTable> tableCacheMap;
  // Map cache id to the state variables used by each consumer.
  private final Map<Integer, Stack<Pair<StateVariable, Integer>>> stateInfoMap;

  public RelationalOperatorCache() {
    this.tableCacheMap = new HashMap<>();
    this.stateInfoMap = new HashMap<>();
  }

  /**
   * Determine if we have already generated code for a cache node.
   *
   * @param key The cache key
   * @return If the node is cached
   */
  public boolean isNodeCached(int key) {
    return tableCacheMap.containsKey(key);
  }

  /**
   * Cache a table
   *
   * @param key The cache key
   * @param table The table to cache.
   */
  public void cacheTable(int key, BodoEngineTable table) {
    tableCacheMap.put(key, table);
  }

  /**
   * Fetch the cached table for a key.
   *
   * @param key The cache key.
   * @return The cached table.
   */
  public BodoEngineTable getCachedTable(int key) {
    return tableCacheMap.get(key);
  }

  /**
   * Set the allocated state variables for a cache node. This should change once we reuse a single
   * buffer.
   *
   * @param key The cache key.
   * @param stateVars The state variables allocated for each consumer.
   */
  public void setStateInfo(int key, List<Pair<StateVariable, Integer>> stateVars) {
    Stack<Pair<StateVariable, Integer>> stack = new Stack<>();
    stack.addAll(stateVars);
    stateInfoMap.put(key, stack);
  }

  /**
   * Fetch the state values to use for the next consumer.
   *
   * @param key The cache key.
   * @return The state variable to use and its operator ID.
   */
  public Pair<StateVariable, Integer> getStateInfo(int key) {
    return stateInfoMap.get(key).pop();
  }
}
