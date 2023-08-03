package com.bodosql.calcite.sql2rel;

import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.rel.core.LogicalTableCreate;
import com.bodosql.calcite.rel.core.RowSample;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.sql2rel.RelStructuredTypeFlattener;
import org.apache.calcite.tools.RelBuilder;

public class BodoRelStructuredTypeFlattener extends RelStructuredTypeFlattener {
  public BodoRelStructuredTypeFlattener(final RelBuilder relBuilder,
      final RexBuilder rexBuilder, final RelOptTable.ToRelContext toRelContext,
      final boolean restructure) {
    super(relBuilder, rexBuilder, toRelContext, restructure);
  }

  public void rewriteRel(RowSample rel) {
    rewriteGeneric(rel);
  }

  // TODO(jsternberg): Should not be a relational node in general.
  public void rewriteRel(LogicalTableCreate rel) {
    LogicalTableCreate newRel =
        LogicalTableCreate.create(
            getNewForOldRel(rel.getInput()),
            rel.getSchema(),
            rel.getTableName(),
            rel.isReplace(),
            rel.getCreateTableType(),
            rel.getSchemaPath()
        );
    setNewForOldRel(rel, newRel);
  }
}
