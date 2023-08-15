package com.bodosql.calcite.application.Utils;

import com.bodosql.calcite.adapter.snowflake.shouldPushDownMetadataQueryHelper;
import org.apache.calcite.rel.RelNode;

/** Helper function */
public class canTrustRowCountHelper {

  /**
   * Helper function that returns if the row count of the node is accurate, or an estimate based on
   * heuristics.
   *
   * <p>Currently, this only returns True if row count is obtained directly from a query pushed to
   * snowflake. In the future, we may extend this further.
   *
   * @param node The node whose row count we're checking.
   * @return true if the row count of the node is accurate, and false otherwise.
   */
  static boolean canTrustRowCount(RelNode node) {
    return shouldPushDownMetadataQueryHelper.shouldPushAsMetaDataQuery(node);
  }
}
