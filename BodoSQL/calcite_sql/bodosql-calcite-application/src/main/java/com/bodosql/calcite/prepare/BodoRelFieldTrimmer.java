package com.bodosql.calcite.prepare;

import com.bodosql.calcite.adapter.snowflake.SnowflakeToPandasConverter;
import com.bodosql.calcite.rex.RexNamedParam;
import com.bodosql.calcite.traits.CombineStreamsExchange;
import com.bodosql.calcite.traits.SeparateStreamExchange;
import com.google.common.collect.ImmutableList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelFieldCollation;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Sort;
import org.apache.calcite.rel.core.TableCreate;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexDynamicParam;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql2rel.RelFieldTrimmer;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.mapping.Mapping;
import org.apache.calcite.util.mapping.Mappings;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Extension of RelFieldTrimmer that covers Bodo's added RelNodes. */
public class BodoRelFieldTrimmer extends RelFieldTrimmer {

  // Store a copy of the RelBuilder
  private final RelBuilder relBuilder;

  public BodoRelFieldTrimmer(@Nullable SqlValidator validator, RelBuilder relBuilder) {
    super(validator, relBuilder);
    this.relBuilder = relBuilder;
  }

  public TrimResult trimFields(
      TableCreate node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {

    TrimResult childResult = dispatchTrimFields(node.getInput(), fieldsUsed, extraFields);
    TableCreate newNode = (TableCreate) node.copy(node.getTraitSet(), List.of(childResult.left));
    return result(newNode, childResult.right);
  }

  public TrimResult trimFields(
      SnowflakeToPandasConverter node,
      ImmutableBitSet fieldsUsed,
      Set<RelDataTypeField> extraFields) {

    TrimResult childResult = dispatchTrimFields(node.getInput(), fieldsUsed, extraFields);
    SnowflakeToPandasConverter newNode =
        (SnowflakeToPandasConverter) node.copy(node.getTraitSet(), List.of(childResult.left));
    return result(newNode, childResult.right);
  }

  public TrimResult trimFields(
      SeparateStreamExchange node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {

    TrimResult childResult = dispatchTrimFields(node.getInput(), fieldsUsed, extraFields);
    SeparateStreamExchange newNode =
        (SeparateStreamExchange) node.copy(node.getTraitSet(), List.of(childResult.left));
    return result(newNode, childResult.right);
  }

  public TrimResult trimFields(
      CombineStreamsExchange node, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {

    TrimResult childResult = dispatchTrimFields(node.getInput(), fieldsUsed, extraFields);
    // Small optimization, if we end up with a CombineStreamExchange on top of a
    // separateStreamExchange,
    // just discard them both
    if (childResult.left instanceof SeparateStreamExchange) {
      return result(childResult.left.getInput(0), childResult.right);
    } else {
      CombineStreamsExchange newNode =
          (CombineStreamsExchange) node.copy(node.getTraitSet(), List.of(childResult.left));
      return result(newNode, childResult.right);
    }
  }

  /**
   * Variant of {@link #trimFields(RelNode, ImmutableBitSet, Set)} for {@link
   * org.apache.calcite.rel.core.Sort}.
   */
  public TrimResult trimFields(
      Sort sort, ImmutableBitSet fieldsUsed, Set<RelDataTypeField> extraFields) {
    final RelDataType rowType = sort.getRowType();
    final int fieldCount = rowType.getFieldCount();
    final RelCollation collation = sort.getCollation();
    final RelNode input = sort.getInput();

    // We use the fields used by the consumer, plus any fields used as sort
    // keys.
    final ImmutableBitSet.Builder inputFieldsUsed = fieldsUsed.rebuild();
    for (RelFieldCollation field : collation.getFieldCollations()) {
      inputFieldsUsed.set(field.getFieldIndex());
    }

    // Create input with trimmed columns.
    final Set<RelDataTypeField> inputExtraFields = Collections.emptySet();
    TrimResult trimResult = trimChild(sort, input, inputFieldsUsed.build(), inputExtraFields);
    RelNode newInput = trimResult.left;
    final Mapping inputMapping = trimResult.right;

    // If the input is unchanged, and we need to project all columns,
    // there's nothing we can do.
    if (newInput == input && inputMapping.isIdentity() && fieldsUsed.cardinality() == fieldCount) {
      return result(sort, Mappings.createIdentity(fieldCount));
    }

    // leave the Sort unchanged in case we have dynamic limits
    // Bodo Change: Include RexNamedParam.
    if (sort.offset instanceof RexDynamicParam
        || sort.fetch instanceof RexDynamicParam
        || sort.offset instanceof RexNamedParam
        || sort.fetch instanceof RexNamedParam) {
      return result(sort, inputMapping);
    }

    relBuilder.push(newInput);
    final int offset = sort.offset == null ? 0 : RexLiteral.intValue(sort.offset);
    final int fetch = sort.fetch == null ? -1 : RexLiteral.intValue(sort.fetch);
    final ImmutableList<RexNode> fields = relBuilder.fields(RexUtil.apply(inputMapping, collation));
    relBuilder.sortLimit(offset, fetch, fields);

    // The result has the same mapping as the input gave us. Sometimes we
    // return fields that the consumer didn't ask for, because the filter
    // needs them for its condition.
    return result(relBuilder.build(), inputMapping);
  }
}
