package com.bodosql.calcite.application.Utils;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;

/** Class of helper functions used to process a Join RelNode. */
public class JoinHelpers {

  /**
   * Function to detect which columns will be renamed within a join/merge.
   *
   * @param names List of names used by a table.
   * @param mergedCols Set of Columns that should be omitted from collisions
   * @param seen Hashmap used to track collisions. A column with a collision between tables should
   *     have a value with true, those in only 1 table should have a value of False. mergeCols are
   *     omitted.
   */
  public static void preventColumnCollision(
      List<String> names, HashSet<String> mergedCols, HashMap<String, Boolean> seen) {
    for (int i = 0; i < names.size(); i += 1) {
      String name = names.get(i);
      if (!mergedCols.contains(name)) {
        if (seen.containsKey(name)) {
          seen.put(name, true);
        } else {
          seen.put(name, false);
        }
      }
    }
  }

  /**
   * Function that determines if we need to perform a filter on the results of the code produced by
   * generateJoinCode in BodoSQLCodeGen/JoinCodeGen
   *
   * @param equalityMergeExprs A Hashset of rexNodes where each rexNode is an equality condition
   * @param cond The original condition of the join in the Calcite Plan
   */
  public static boolean needsPostJoinFilter(HashSet<RexNode> equalityMergeExprs, RexNode cond) {

    boolean filterOutput = true;
    if (equalityMergeExprs.size() == 1) {
      // Note: If there is more than 1 key for merging, it may be possible to avoid
      // filtering, but this requires a more detailed update.
      RexNode firstEntry = equalityMergeExprs.iterator().next();
      RexCall condition = (RexCall) firstEntry;
      filterOutput = (RexCall) cond != condition;
    } else {
      /* We skip filtering if the cond is a boolean literal (false should be optimized out). */
      if (cond instanceof RexLiteral && ((RexLiteral) cond).getValue() instanceof Boolean) {
        filterOutput = false;
      }
    }
    return filterOutput;
  }
}
