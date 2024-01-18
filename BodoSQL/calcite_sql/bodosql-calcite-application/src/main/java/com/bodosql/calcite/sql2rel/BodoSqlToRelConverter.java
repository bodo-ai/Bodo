package com.bodosql.calcite.sql2rel;

import static java.util.Objects.requireNonNull;

import com.bodosql.calcite.plan.RelOptRowSamplingParameters;
import com.bodosql.calcite.rel.core.RowSample;
import com.bodosql.calcite.schema.FunctionExpander;
import com.bodosql.calcite.sql.SqlTableSampleRowLimitSpec;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.ViewExpanders;
import org.apache.calcite.prepare.Prepare;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.hint.RelHint;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.runtime.Resources;
import org.apache.calcite.schema.Function;
import org.apache.calcite.schema.FunctionParameter;
import org.apache.calcite.sql.SnowflakeUserDefinedFunction;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlSampleSpec;
import org.apache.calcite.sql.validate.SqlUserDefinedFunction;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorException;
import org.apache.calcite.sql.validate.SqlValidatorScope;
import org.apache.calcite.sql2rel.RelStructuredTypeFlattener;
import org.apache.calcite.sql2rel.SqlRexConvertletTable;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.tools.RelBuilder;
import org.checkerframework.checker.nullness.qual.Nullable;

public class BodoSqlToRelConverter extends SqlToRelConverter {
  // Duplicated because SqlToRelConverter doesn't give access to the one
  // it created.
  private final RelBuilder relBuilder;

  // Function expander used for inlining Snowflake UDFs.
  private final FunctionExpander functionExpander;

  public BodoSqlToRelConverter(
      final RelOptTable.ViewExpander viewExpander,
      @Nullable final SqlValidator validator,
      final Prepare.CatalogReader catalogReader,
      final RelOptCluster cluster,
      final SqlRexConvertletTable convertletTable,
      final Config config,
      final FunctionExpander functionExpander) {
    super(viewExpander, validator, catalogReader, cluster, convertletTable, config);
    this.relBuilder =
        config
            .getRelBuilderFactory()
            .create(cluster, null)
            .transform(config.getRelBuilderConfigTransform());
    this.functionExpander = functionExpander;
  }

  @Override
  public RelNode flattenTypes(RelNode rootRel, boolean restructure) {
    RelStructuredTypeFlattener typeFlattener =
        new BodoRelStructuredTypeFlattener(
            relBuilder, rexBuilder, createToRelContext(ImmutableList.of()), restructure);
    return typeFlattener.rewrite(rootRel);
  }

  private RelOptTable.ToRelContext createToRelContext(List<RelHint> hints) {
    return ViewExpanders.toRelContext(viewExpander, cluster, hints);
  }

  @Override
  protected void convertFrom(
      Blackboard bb, @Nullable SqlNode from, @Nullable List<String> fieldNames) {
    if (from == null) {
      super.convertFrom(bb, null, fieldNames);
      return;
    }

    switch (from.getKind()) {
      case TABLESAMPLE:
        // Handle row sampling separately as this is not part of calcite core.
        final List<SqlNode> operands = ((SqlCall) from).getOperandList();
        SqlSampleSpec sampleSpec =
            SqlLiteral.sampleValue(requireNonNull(operands.get(1), () -> "operand[1] of " + from));
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

  protected class BodoBlackboard extends Blackboard {

    /**
     * Creates a Blackboard.
     *
     * @param scope Name-resolution scope for expressions validated within this query. Can be null
     *     if this Blackboard is for a leaf node, say
     * @param nameToNodeMap Map which translates the expression to map a given parameter into, if
     *     translating expressions; null otherwise
     * @param top Whether this is the root of the query
     */
    protected BodoBlackboard(
        @Nullable SqlValidatorScope scope,
        @Nullable Map<String, RexNode> nameToNodeMap,
        boolean top) {
      super(scope, nameToNodeMap, top);
    }

    /** Extension of Blackboard.convertExpression with support for inlining SnowflakeUDFs. */
    @Override
    public RexNode convertExpression(SqlNode expr) {
      if (expr instanceof SqlCall) {
        SqlCall call = ((SqlCall) expr);
        if (call.getOperator() instanceof SqlUserDefinedFunction) {
          SqlUserDefinedFunction udf = (SqlUserDefinedFunction) call.getOperator();
          Function function = udf.getFunction();
          if (function instanceof SnowflakeUserDefinedFunction) {
            SnowflakeUserDefinedFunction snowflakeUdf = (SnowflakeUserDefinedFunction) function;
            Resources.ExInst<SqlValidatorException> exInst = snowflakeUdf.errorOrWarn();
            if (exInst != null) {
              throw validator.newValidationError(expr, exInst);
            }
            // Expand the call to add default values.
            SqlCall extendedCall = new SqlCallBinding(validator, scope, call).permutedCall();
            Resources.ExInst<SqlValidatorException> defaultExInst =
                snowflakeUdf.errorOnDefaults(extendedCall.getOperandList());
            if (defaultExInst != null) {
              throw validator.newValidationError(expr, defaultExInst);
            }
            // Construct parameter type information.
            List<FunctionParameter> parameters = snowflakeUdf.getParameters();
            List<SqlNode> operands = extendedCall.getOperandList();
            List<String> names = new ArrayList<>();
            List<RexNode> arguments = new ArrayList<>();
            for (int i = 0; i < parameters.size(); i++) {
              FunctionParameter parameter = parameters.get(i);
              // Add the name
              String name = parameter.getName();
              names.add(name);
              // Add the RexNode
              SqlNode operand = operands.get(i);
              RexNode convertedOperand = convertExpression(operand);
              arguments.add(convertedOperand);
            }
            // Get the expected return type for casting the output
            RelDataType returnType = snowflakeUdf.getReturnType(typeFactory);
            return functionExpander.expandFunction(
                snowflakeUdf.getBody(),
                snowflakeUdf.getFunctionPath(),
                names,
                arguments,
                returnType);
          }
        }
      }
      // Bodo extension to handle Snowflake UDFs
      return super.convertExpression(expr);
    }
  }

  /** Factory method for creating translation workspace. */
  @Override
  protected Blackboard createBlackboard(
      SqlValidatorScope scope, @Nullable Map<String, RexNode> nameToNodeMap, boolean top) {
    return new BodoBlackboard(scope, nameToNodeMap, top);
  }
}
