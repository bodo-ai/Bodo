package com.bodosql.calcite.application.utils;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

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
}
