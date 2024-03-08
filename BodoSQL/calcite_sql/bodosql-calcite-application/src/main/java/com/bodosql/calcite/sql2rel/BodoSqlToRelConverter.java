package com.bodosql.calcite.sql2rel;

import static java.util.Objects.requireNonNull;

import com.bodosql.calcite.application.logicalRules.BodoSQLReduceExpressionsRule;
import com.bodosql.calcite.application.operatorTables.SnowflakeNativeUDF;
import com.bodosql.calcite.plan.RelOptRowSamplingParameters;
import com.bodosql.calcite.rel.core.RowSample;
import com.bodosql.calcite.rel.logical.BodoLogicalTableCreate;
import com.bodosql.calcite.schema.FunctionExpander;
import com.bodosql.calcite.sql.SqlTableSampleRowLimitSpec;
import com.bodosql.calcite.sql.ddl.SqlSnowflakeCreateTableBase;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.ViewExpanders;
import org.apache.calcite.prepare.Prepare;
import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelCollations;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.hint.RelHint;
import org.apache.calcite.rel.logical.LogicalSort;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexFieldAccess;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.runtime.Resources;
import org.apache.calcite.schema.Function;
import org.apache.calcite.schema.FunctionParameter;
import org.apache.calcite.sql.SnowflakeNamedArgumentSqlCatalogTableFunction;
import org.apache.calcite.sql.SnowflakeUserDefinedBaseFunction;
import org.apache.calcite.sql.SnowflakeUserDefinedFunction;
import org.apache.calcite.sql.SnowflakeUserDefinedTableFunction;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlSampleSpec;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.util.SqlBasicVisitor;
import org.apache.calcite.sql.validate.ListScope;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.apache.calcite.sql.validate.SqlQualified;
import org.apache.calcite.sql.validate.SqlUserDefinedFunction;
import org.apache.calcite.sql.validate.SqlUserDefinedTableFunction;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorException;
import org.apache.calcite.sql.validate.SqlValidatorNamespace;
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

  // Map used to pass to all blackboards so parameters when
  // inlining UDFs can be globally referenced.
  private final Map<String, RexNode> paramNameToNodeMap;

  public BodoSqlToRelConverter(
      final RelOptTable.ViewExpander viewExpander,
      @Nullable final SqlValidator validator,
      final Prepare.CatalogReader catalogReader,
      final RelOptCluster cluster,
      final SqlRexConvertletTable convertletTable,
      final Config config,
      final FunctionExpander functionExpander,
      Map<String, RexNode> paramNameToNodeMap) {
    super(viewExpander, validator, catalogReader, cluster, convertletTable, config);
    this.relBuilder =
        config
            .getRelBuilderFactory()
            .create(cluster, null)
            .transform(config.getRelBuilderConfigTransform());
    this.functionExpander = functionExpander;
    this.paramNameToNodeMap = paramNameToNodeMap;
  }

  public BodoSqlToRelConverter(
      final RelOptTable.ViewExpander viewExpander,
      @Nullable final SqlValidator validator,
      final Prepare.CatalogReader catalogReader,
      final RelOptCluster cluster,
      final SqlRexConvertletTable convertletTable,
      final Config config,
      final FunctionExpander functionExpander) {
    this(
        viewExpander,
        validator,
        catalogReader,
        cluster,
        convertletTable,
        config,
        functionExpander,
        Map.of());
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

  /**
   * Recursively converts a query to a relational expression.
   *
   * @param query Query
   * @param top Whether this query is the top-level query of the statement
   * @param targetRowType Target row type, or null
   * @return Relational expression
   */
  @Override
  protected RelRoot convertQueryRecursive(
      SqlNode query, boolean top, @Nullable RelDataType targetRowType) {
    if (query instanceof SqlSnowflakeCreateTableBase) {
      // Create table has to be the topmost relnode
      assert top;
      // Bodo change: intercept logic for CREATE TABLE and call our own method
      return RelRoot.of(
          bodoConvertCreateTable((SqlSnowflakeCreateTableBase) query), SqlKind.CREATE_TABLE);
    }
    return super.convertQueryRecursive(query, top, targetRowType);
  }

  /**
   * Converts a CREATE TABLE SqlNode to a RelNode.
   *
   * @param createCall The CREATE TABLE call being converted
   */
  private RelNode bodoConvertCreateTable(SqlSnowflakeCreateTableBase createCall) {
    final SqlNode createTableDef = requireNonNull(createCall.getQuery());
    final RelNode inputRel =
        convertCreateTableDefinition(
            createTableDef, requireNonNull(this.validator).getCreateTableScope(createCall));
    return BodoLogicalTableCreate.create(
        inputRel,
        requireNonNull(createCall.getOutputTableSchema()),
        requireNonNull(createCall.getOutputTableName()),
        createCall.getReplace(),
        createCall.getCreateTableType(),
        requireNonNull(createCall.getOutputTableSchemaPath()),
        createCall.getMeta());
  }

  private RelNode convertCreateTableQuery(SqlNode createTableDef) {
    SqlNode nonNullCreateTableDef = requireNonNull(createTableDef, "createTableDef");
    RelRoot relRoot = convertQueryRecursive(nonNullCreateTableDef, false, null);
    // Map the root to the final row type. This is needed to prune intermediate columns needed
    // if the inner data requires intermediate columns. See testCreateTableOrderByExpr
    // as an example.
    RelDataType validatedRowType = this.validator.getValidatedNodeType(nonNullCreateTableDef);
    RelNode topNode = relRoot.project();
    return RelRoot.of(topNode, validatedRowType, nonNullCreateTableDef.getKind()).project();
  }

  private RelNode convertCreateTableIdentifier(
      SqlNode createTableDef, SqlValidatorScope createTableScope) {
    // The scope of the createTableCall should always just be the catalog scope,
    // which we use to convert this table

    final Blackboard bb = createBlackboard(createTableScope, null, false);
    convertFrom(bb, createTableDef);
    final RelCollation emptyCollation = cluster.traitSet().canonize(RelCollations.of());

    // Create Table like creates a table with an identical schema, copies no rows.
    // Therefore, we add a LIMIT 0 to the query.
    return LogicalSort.create(
        bb.root(),
        emptyCollation,
        null,
        relBuilder
            .getRexBuilder()
            .makeLiteral(0, typeFactory.createSqlType(SqlTypeName.BIGINT), false));
  }

  protected RelNode convertCreateTableDefinition(
      SqlNode createTableDef, SqlValidatorScope createTableScope) {

    if (createTableDef instanceof SqlIdentifier) {
      return convertCreateTableIdentifier(createTableDef, createTableScope);
    } else {
      return convertCreateTableQuery(createTableDef);
    }
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
            SnowflakeUserDefinedFunction snowflakeScalarUdf =
                (SnowflakeUserDefinedFunction) function;
            SqlCall extendedCall =
                validateSnowflakeUDFOperands(
                    call, snowflakeScalarUdf, validator, scope, typeFactory);
            // Construct RexNodes for the arguments.
            List<RexNode> arguments = new ArrayList<>();
            for (int i = 0; i < extendedCall.operandCount(); i++) {
              SqlNode operand = extendedCall.operand(i);
              RexNode convertedOperand = convertExpression(operand);
              arguments.add(convertedOperand);
            }
            // Get the expected return type
            RelDataType returnType = snowflakeScalarUdf.getReturnType(typeFactory);
            // Fetch the parameters
            List<FunctionParameter> parameters = snowflakeScalarUdf.getParameters();
            if (snowflakeScalarUdf.canInlineFunction()) {
              // Inline the function body.
              // Construct the name information.
              List<String> names = new ArrayList<>();
              for (int i = 0; i < parameters.size(); i++) {
                FunctionParameter parameter = parameters.get(i);
                // Add the name
                String name = parameter.getName();
                names.add(name);
              }
              // Generate correlated variables in case we have a query that references
              // the input row.
              CorrMapVisitor visitor = new CorrMapVisitor();
              call.accept(visitor);
              BodoSQLReduceExpressionsRule.RexReplacer replacer =
                  new BodoSQLReduceExpressionsRule.RexReplacer(
                      rexBuilder, visitor.getInputRefs(), visitor.getFieldRefs());
              List<RexNode> correlatedArguments =
                  arguments.stream().map(x -> x.accept(replacer)).collect(Collectors.toList());
              return functionExpander.expandFunction(
                  snowflakeScalarUdf.getBody(),
                  snowflakeScalarUdf.getFunctionPath(),
                  names,
                  arguments,
                  correlatedArguments,
                  returnType,
                  cluster);
            } else {
              // Generate a RexCall for our Native UDF operator.
              String body = snowflakeScalarUdf.getBody();
              String language = snowflakeScalarUdf.getLanguage();
              List<RelDataType> argTypes =
                  parameters.stream().map(x -> x.getType(typeFactory)).collect(Collectors.toList());
              SqlFunction op = SnowflakeNativeUDF.create(body, language, argTypes, returnType);
              return rexBuilder.makeCall(op, arguments);
            }
          }
        }
      }
      // Bodo extension to handle Snowflake UDFs
      return super.convertExpression(expr);
    }

    /** SqlBasicVisitor for preparing and updating the correlated variable arguments to a */
    protected class CorrMapVisitor extends SqlBasicVisitor<Void> {
      List<RexNode> inputRefs;
      List<RexNode> fieldRefs;

      CorrMapVisitor() {
        this.inputRefs = new ArrayList();
        this.fieldRefs = new ArrayList();
      }

      @Override
      public Void visit(SqlCall call) {
        // Don't recurse on the parameter name of an argument as an identifier.
        if (call.getKind() == SqlKind.ARGUMENT_ASSIGNMENT) {
          return call.getOperandList().get(0).accept(this);
        } else {
          return super.visit(call);
        }
      }

      @Override
      public Void visit(SqlIdentifier id) {
        // Add the input ref
        RexNode convertedNode = BodoBlackboard.this.convertExpression(id);
        if (!(convertedNode instanceof RexInputRef)) {
          return null;
        }
        RexInputRef inputRef = (RexInputRef) convertedNode;
        // Create the correlation ID
        CorrelationId correlId = cluster.createCorrel();
        // Find the corresponding table type.
        SqlQualified qualified = scope.fullyQualify(id);

        // Resolve the id's expression and find the from location. This is
        // needed to properly handle joins and is largely copied from lookupExp.
        final SqlNameMatcher nameMatcher = scope.getValidator().getCatalogReader().nameMatcher();
        final SqlValidatorScope.ResolvedImpl resolved = new SqlValidatorScope.ResolvedImpl();
        scope.resolve(qualified.prefix(), nameMatcher, false, resolved);
        if (resolved.count() != 1) {
          throw new AssertionError(
              "no unique expression found for " + qualified + "; count is " + resolved.count());
        }
        final SqlValidatorScope.Resolve resolve = resolved.only();
        final RelDataType rowType = resolve.rowType();

        // Build the type from the potential from list
        final RelDataType finalRowType;
        if (resolve.path.steps().get(0).i < 0) {
          finalRowType = rowType;
        } else {
          // We may have multiple sources. We need to build the row type from the columns
          // in the inputs. This is particular relevant if we are the output of a join
          // or a union.
          final RelDataTypeFactory.Builder builder = typeFactory.builder();
          final ListScope ancestorScope1 =
              (ListScope) requireNonNull(resolve.scope, "resolve.scope");
          final ImmutableMap.Builder<String, Integer> fields = ImmutableMap.builder();
          int i = 0;
          // The child namespaces build the type for this namespace.
          for (SqlValidatorNamespace c : ancestorScope1.getChildren()) {
            if (ancestorScope1.isChildNullable(i)) {
              for (final RelDataTypeField f : c.getRowType().getFieldList()) {
                builder.add(f.getName(), typeFactory.createTypeWithNullability(f.getType(), true));
              }
            } else {
              builder.addAll(c.getRowType().getFieldList());
            }
            ++i;
          }
          // Make sure field names don't conflict.
          finalRowType = builder.uniquify().build();
        }
        // Create the correlation variable for rewriting the arguments
        RexNode correlVariable = rexBuilder.makeCorrel(finalRowType, correlId);
        RexFieldAccess fieldAccess =
            (RexFieldAccess) rexBuilder.makeFieldAccess(correlVariable, inputRef.getIndex());

        // Update the outputs
        this.inputRefs.add(inputRef);
        this.fieldRefs.add(fieldAccess);
        // Update the maps for tracking correlation variables.
        mapCorrelToDeferred.put(
            correlId, new DeferredLookup(BodoBlackboard.this, qualified.identifier.names.get(0)));
        mapCorrelateToRex.put(correlId, fieldAccess);
        return null;
      }

      public List<RexNode> getInputRefs() {
        return this.inputRefs;
      }

      public List<RexNode> getFieldRefs() {
        return this.fieldRefs;
      }
    }
  }

  /** Factory method for creating translation workspace. */
  @Override
  protected Blackboard createBlackboard(
      SqlValidatorScope scope, @Nullable Map<String, RexNode> nameToNodeMap, boolean top) {
    if (paramNameToNodeMap.isEmpty()) {
      return new BodoBlackboard(scope, nameToNodeMap, top);
    } else if (nameToNodeMap == null) {
      return new BodoBlackboard(scope, paramNameToNodeMap, top);
    } else {
      Map<String, RexNode> finalMap = new HashMap<>();
      // Parameters, which occur when inlining a function,
      // are given precedence over other names. In practice
      // this shouldn't occur because most nodes don't set
      // nameToNodeMap.
      for (String key : paramNameToNodeMap.keySet()) {
        finalMap.put(key, paramNameToNodeMap.get(key));
      }
      for (String key : nameToNodeMap.keySet()) {
        if (!paramNameToNodeMap.containsKey(key)) {
          finalMap.put(key, nameToNodeMap.get(key));
        }
      }
      return new BodoBlackboard(scope, finalMap, top);
    }
  }

  /**
   * Handle converting table function calls from SQL nodes to RelNodes. This currently only provides
   * support for inlining Snowflake UDTFs.
   *
   * @param bb The blackboard used the convert expressions.
   * @param call The SQLCall.
   */
  @Override
  protected void convertCollectionTable(Blackboard bb, SqlCall call) {
    final SqlOperator operator = call.getOperator();
    if (operator instanceof SqlUserDefinedTableFunction) {
      SqlUserDefinedTableFunction udf = (SqlUserDefinedTableFunction) call.getOperator();
      Function function = udf.getFunction();
      if (function instanceof SnowflakeNamedArgumentSqlCatalogTableFunction) {
        super.convertCollectionTable(bb, call);
        return;
      } else if (udf instanceof SqlUserDefinedTableFunction) {
        replaceSubQueries(bb, call, RelOptUtil.Logic.TRUE_FALSE_UNKNOWN);
        SnowflakeUserDefinedTableFunction snowflakeTableUdf =
            (SnowflakeUserDefinedTableFunction) function;
        SqlCall extendedCall =
            validateSnowflakeUDFOperands(call, snowflakeTableUdf, validator, bb.scope, typeFactory);
        List<String> names =
            snowflakeTableUdf.getParameters().stream()
                .map(x -> x.getName())
                .collect(Collectors.toList());
        List<RexNode> arguments =
            extendedCall.getOperandList().stream()
                .map(x -> bb.convertExpression(x))
                .collect(Collectors.toList());
        RelNode expandedFunction =
            functionExpander.expandTableFunction(
                snowflakeTableUdf.getBody(),
                snowflakeTableUdf.getFunctionPath(),
                names,
                arguments,
                snowflakeTableUdf.getRowType(typeFactory, List.of()),
                cluster);
        bb.setRoot(expandedFunction, true);
        return;
      }
    }
    super.convertCollectionTable(bb, call);
  }

  /**
   * Perform the necessary error checks on a Snowflake User defined function and return the fully
   * expanded SQLOperands
   *
   * @param call The SQLCall expression we aim to expand.
   * @param snowflakeUdf The Snowflake UDF operator.
   * @param validator The validator used to validate the call.
   * @param scope The scope for evaluating the call.
   * @param typeFactory The type factory used to check we support the UDFs with the available types.
   * @return The expanded SQLOperands that insert any values like defaults.
   */
  private static SqlCall validateSnowflakeUDFOperands(
      SqlCall call,
      SnowflakeUserDefinedBaseFunction snowflakeUdf,
      SqlValidator validator,
      SqlValidatorScope scope,
      RelDataTypeFactory typeFactory) {
    Resources.ExInst<SqlValidatorException> exInst = snowflakeUdf.errorOrWarn(typeFactory);
    if (exInst != null) {
      throw validator.newValidationError(call, exInst);
    }
    // Expand the call to add default values.
    SqlCall extendedCall = new SqlCallBinding(validator, scope, call).permutedCall();
    Resources.ExInst<SqlValidatorException> defaultExInst =
        snowflakeUdf.errorOnDefaults(extendedCall.getOperandList());
    if (defaultExInst != null) {
      throw validator.newValidationError(call, defaultExInst);
    }
    return extendedCall;
  }
}
