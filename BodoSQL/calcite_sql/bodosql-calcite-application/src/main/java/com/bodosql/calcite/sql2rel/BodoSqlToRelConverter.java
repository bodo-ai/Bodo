package com.bodosql.calcite.sql2rel;

import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptSamplingParameters;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.ViewExpanders;
import org.apache.calcite.prepare.Prepare;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Sample;
import org.apache.calcite.rel.hint.RelHint;
import org.apache.calcite.rel.logical.LogicalTargetTableScan;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeFieldImpl;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlSampleSpec;
import org.apache.calcite.sql.SqlTableIdentifierWithID;
import org.apache.calcite.sql.SqlUtil;
import org.apache.calcite.sql.type.BasicSqlType;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorNamespace;
import org.apache.calcite.sql.validate.SqlValidatorTable;
import org.apache.calcite.sql.validate.SqlValidatorUtil;
import org.apache.calcite.sql2rel.RelStructuredTypeFlattener;
import org.apache.calcite.sql2rel.SqlRexConvertletTable;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.tools.RelBuilder;

import com.bodosql.calcite.plan.RelOptRowSamplingParameters;
import com.bodosql.calcite.rel.core.RowSample;
import com.bodosql.calcite.sql.SqlTableSampleRowLimitSpec;
import com.google.common.collect.ImmutableList;

import org.checkerframework.checker.nullness.qual.Nullable;

import java.util.ArrayList;
import java.util.List;

import static java.util.Objects.requireNonNull;
import static org.apache.calcite.linq4j.Nullness.castNonNull;

public class BodoSqlToRelConverter extends SqlToRelConverter {
  // Duplicated because SqlToRelConverter doesn't give access to the one
  // it created.
  private final RelBuilder relBuilder;

  public BodoSqlToRelConverter(final RelOptTable.ViewExpander viewExpander,
      @Nullable final SqlValidator validator,
      final Prepare.CatalogReader catalogReader, final RelOptCluster cluster,
      final SqlRexConvertletTable convertletTable, final Config config) {
    super(viewExpander, validator, catalogReader, cluster, convertletTable, config);
    this.relBuilder = config.getRelBuilderFactory().create(cluster, null)
        .transform(config.getRelBuilderConfigTransform());
  }

  @Override
  public RelNode flattenTypes(
      RelNode rootRel,
      boolean restructure) {
    RelStructuredTypeFlattener typeFlattener =
        new BodoRelStructuredTypeFlattener(relBuilder,
            rexBuilder, createToRelContext(ImmutableList.of()), restructure);
    return typeFlattener.rewrite(rootRel);
  }

  private RelOptTable.ToRelContext createToRelContext(List<RelHint> hints) {
    return ViewExpanders.toRelContext(viewExpander, cluster, hints);
  }

  @Override
  protected void convertFrom(
      Blackboard bb,
      @Nullable SqlNode from,
      @Nullable List<String> fieldNames) {
    if (from == null) {
      super.convertFrom(bb, null, fieldNames);
      return;
    }

    switch (from.getKind()) {
    case TABLESAMPLE:
      // Handle row sampling separately as this is not part of calcite core.
      final List<SqlNode> operands = ((SqlCall) from).getOperandList();
      SqlSampleSpec sampleSpec = SqlLiteral.sampleValue(
          requireNonNull(operands.get(1), () -> "operand[1] of " + from));
      if (sampleSpec instanceof SqlTableSampleRowLimitSpec) {
        SqlTableSampleRowLimitSpec tableSampleRowLimitSpec =
            (SqlTableSampleRowLimitSpec) sampleSpec;
        convertFrom(bb, operands.get(0));
        RelOptRowSamplingParameters params =
            new RelOptRowSamplingParameters(
                tableSampleRowLimitSpec.isBernoulli(),
                tableSampleRowLimitSpec.getNumberOfRows().intValue(),
                tableSampleRowLimitSpec.isRepeatable(),
                tableSampleRowLimitSpec.getRepeatableSeed());
        bb.setRoot(new RowSample(cluster, bb.root(), params), false);
        return;
      }

      // Let calcite core handle this conversion.
      break;
    }

    // Defer all other conversions to calcite core.
    super.convertFrom(bb, from, fieldNames);
  }
}
