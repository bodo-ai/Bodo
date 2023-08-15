package com.bodosql.calcite.adapter.snowflake;

import org.apache.calcite.plan.volcano.RelSubset;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelVisitor;

/** Wrapper around shouldPushDownMetadataQueryVisitor, that provides a nicer interface */
public class shouldPushDownMetadataQueryHelper extends RelVisitor {

  /**
   * Returns true if we should push the node to SF to get the meta-data, and false otherwise.
   * Currently, this is true if the rel tree consists only of Snowflake filters/table scans, and
   * false otherwise.
   *
   * @param node the node to be checked
   * @return boolean value for if the rel tree should be pushed to SF as a metadata query
   */
  public static boolean shouldPushAsMetaDataQuery(RelNode node) {
    shouldPushDownMetadataQueryVisitor visitor = new shouldPushDownMetadataQueryVisitor();
    visitor.canPushDown = true;
    visitor.visit(node, 0, null);
    return visitor.canPushDown;
  }

  /**
   * Helper visitor for shouldPushDownMetadataQueryHelper. Visits each of the elements of the
   * RelNode tree, and sets its property "canPushDown" to true/false depending on if the query
   * should be pushed.
   */
  static class shouldPushDownMetadataQueryVisitor extends RelVisitor {

    Boolean canPushDown = true;

    @Override
    public void visit(RelNode node, int ordinal, RelNode parent) {
      // Don't bother visiting if we've already determined
      // that we can't push down the filter
      if (canPushDown) {

        // Only allow RelNode trees that contain only filters and snowflake
        // table scans. Note that SnowflakeFilter's are only created for a whitelisted
        // set of conditions, so this will limit the set of generated queries
        // to conditions set in
        // com.bodosql.calcite.adapter.snowflake.AbstractSnowflakeFilterRule.Companion.isPushableFilter
        if (node instanceof RelSubset) {
          // RelSubset is a set of possible equivalent nodes.
          // Since we don't know exactly the relSubset will
          // eventually resolve to, we can't
          // determine if we should/shouldn't push it down
          canPushDown = false;
        } else if (node instanceof SnowflakeTableScan) {
          // Do nothing
        } else if (node instanceof SnowflakeFilter) {
          node.childrenAccept(this);
        } else {
          canPushDown = false;
        }
      }
    }
  }
}
