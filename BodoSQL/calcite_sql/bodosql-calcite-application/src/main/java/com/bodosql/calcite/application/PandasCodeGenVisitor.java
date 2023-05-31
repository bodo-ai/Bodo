package com.bodosql.calcite.application;

import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.concatDataFrames;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.generateAggCodeNoAgg;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.generateAggCodeNoGroupBy;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.generateAggCodeWithGroupBy;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.generateApplyCodeWithGroupBy;
import static com.bodosql.calcite.application.BodoSQLCodeGen.FilterCodeGen.generateFilterCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.JoinCodeGen.generateJoinCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.LiteralCodeGen.generateLiteralCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.LogicalValuesCodeGen.generateLogicalValuesCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ProjectCodeGen.generateLocCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ProjectCodeGen.generateProjectCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ProjectCodeGen.generateProjectedDataframe;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ProjectCodeGen.generateRenameCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SampleCodeGen.generateRowSampleCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SampleCodeGen.generateSampleCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SetOpCodeGen.generateExceptCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SetOpCodeGen.generateIntersectCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SetOpCodeGen.generateUnionCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SortCodeGen.generateSortCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SortCodeGen.getAscendingExpr;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SortCodeGen.getNAPositionStringLiteral;
import static com.bodosql.calcite.application.BodoSQLCodeGen.WindowAggCodeGen.generateOptimizedEngineKernelCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.WindowAggCodeGen.generateWindowedAggFn;
import static com.bodosql.calcite.application.BodoSQLCodeGen.WindowAggCodeGen.revertSortColumnName;
import static com.bodosql.calcite.application.BodoSQLCodeGen.WindowAggCodeGen.usesOptimizedEngineKernel;
import static com.bodosql.calcite.application.JoinCondVisitor.getStreamingJoinKeyIndices;
import static com.bodosql.calcite.application.JoinCondVisitor.visitJoinCond;
import static com.bodosql.calcite.application.JoinCondVisitor.visitNonEquiConditions;
import static com.bodosql.calcite.application.Utils.AggHelpers.aggContainsFilter;
import static com.bodosql.calcite.application.Utils.Utils.getBodoIndent;
import static com.bodosql.calcite.application.Utils.Utils.integerLiteralArange;
import static com.bodosql.calcite.application.Utils.Utils.isSnowflakeCatalogTable;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;
import static com.bodosql.calcite.application.Utils.Utils.sqlTypenameToPandasTypename;
import static com.bodosql.calcite.application.Utils.Utils.stringsToStringLiterals;

import com.bodosql.calcite.adapter.pandas.PandasAggregate;
import com.bodosql.calcite.adapter.pandas.PandasFilter;
import com.bodosql.calcite.adapter.pandas.PandasIntersect;
import com.bodosql.calcite.adapter.pandas.PandasJoin;
import com.bodosql.calcite.adapter.pandas.PandasMinus;
import com.bodosql.calcite.adapter.pandas.PandasProject;
import com.bodosql.calcite.adapter.pandas.PandasRel;
import com.bodosql.calcite.adapter.pandas.PandasRowSample;
import com.bodosql.calcite.adapter.pandas.PandasSample;
import com.bodosql.calcite.adapter.pandas.PandasSort;
import com.bodosql.calcite.adapter.pandas.PandasTableCreate;
import com.bodosql.calcite.adapter.pandas.PandasTableModify;
import com.bodosql.calcite.adapter.pandas.PandasTableScan;
import com.bodosql.calcite.adapter.pandas.PandasTargetTableScan;
import com.bodosql.calcite.adapter.pandas.PandasUnion;
import com.bodosql.calcite.adapter.pandas.PandasValues;
import com.bodosql.calcite.adapter.pandas.RexToPandasTranslator;
import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan;
import com.bodosql.calcite.adapter.snowflake.SnowflakeToPandasConverter;
import com.bodosql.calcite.application.BodoSQLCodeGen.WindowAggCodeGen;
import com.bodosql.calcite.application.BodoSQLCodeGen.WindowedAggregationArgument;
import com.bodosql.calcite.application.Utils.BodoCtx;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.ir.Dataframe;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.Call;
import com.bodosql.calcite.ir.Expr.IntegerLiteral;
import com.bodosql.calcite.ir.Expr.StringLiteral;
import com.bodosql.calcite.ir.Module;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.Op.Assign;
import com.bodosql.calcite.ir.StreamingPipelineFrame;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.schema.CatalogSchemaImpl;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.LocalTableImpl;
import com.bodosql.calcite.traits.BatchingProperty;
import com.bodosql.calcite.traits.CombineStreamsExchange;
import com.bodosql.calcite.traits.SeparateStreamExchange;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Stack;
import java.util.TreeMap;
import kotlin.jvm.functions.Function0;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.prepare.RelOptTableImpl;
import org.apache.calcite.rel.RelFieldCollation;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelVisitor;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Correlate;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Intersect;
import org.apache.calcite.rel.core.JoinInfo;
import org.apache.calcite.rel.core.Minus;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.Sort;
import org.apache.calcite.rel.core.TableModify;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.core.Union;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexOver;
import org.apache.calcite.rex.RexWindow;
import org.apache.calcite.rex.RexWindowBound;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Pair;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Visitor class for parsed SQL nodes to generate Pandas code from SQL code. */
public class PandasCodeGenVisitor extends RelVisitor {

  /** Stack of generated variables df1, df2 , etc. */
  private final Stack<Variable> varGenStack = new Stack<>();

  /* Reserved column name for generating dummy columns. */
  // TODO: Add this to the docs as banned
  private final Module.Builder generatedCode;

  // Note that a given query can only have one MERGE INTO statement. Therefore,
  // we can statically define the variable names we'll use for the iceberg file list and snapshot
  // id,
  // since we'll only be using these variables once per query
  private static final String icebergFileListVarName = "__bodo_Iceberg_file_list";
  private static final String icebergSnapshotIDName = "__bodo_Iceberg_snapshot_id";

  public static final Expr BATCH_SIZE = new Expr.IntegerLiteral(4000);

  private static final String ROW_ID_COL_NAME = "_bodo_row_id";
  private static final String MERGE_ACTION_ENUM_COL_NAME = "_merge_into_change";

  // Mapping from a unique key per node to exprTypes
  public final HashMap<String, BodoSQLExprType.ExprType> exprTypesMap;

  // Mapping from String Key of Search Nodes to the RexNodes expanded
  // TODO: Replace this code with something more with an actual
  // update to the plan.
  // Ideally we can use RexRules when they are available
  // https://issues.apache.org/jira/browse/CALCITE-4559
  public final HashMap<String, RexNode> searchMap;

  // Map of RelNode ID -> <DataFrame variable name>
  // Because the logical plan is a tree, Nodes that are at the bottom of
  // the tree must be repeated, even if they are identical. However, when
  // calcite produces identical nodes, it gives them the same node ID. As a
  // result, when finding nodes we wish to cache, we log variable names in this
  // map and load them inplace of segments of generated code.
  // This is currently only implemented for a subset of nodes.

  private final HashMap<Integer, Variable> varCache;

  /*
  Hashmap containing globals that need to be lowered into the output func_text. Used for lowering
  metadata types to improve compilation speed.
  hashmap maps String variable names to their String value.
  For example loweredGlobals = {"x": "ColumnMetaDataType(('A', 'B', 'C'))"} would lead to the
  a func_text generation/execution that is equivalent to the following:

  x = ColumnMetaDataType(('A', 'B', 'C'))
  def impl(...):
   ...
   init_dataframe( _, _, x)
   ...

  (Note, we do not actually generate the above func text, we pass the values as globals when calling exec in python. See
   context.py and context_ext.py for more info)
  */
  private final HashMap<String, String> loweredGlobals;

  // The original SQL query. This is used for any operations that must be entirely
  // pushed into a remote db (e.g. Snowflake)
  private final String originalSQLQuery;

  // Debug flag set for certain tests in our test suite. Causes the codegen to return simply return
  // the delta table
  // when encountering a merge into operation
  private final boolean debuggingDeltaTable;

  // The typesystem, used to access timezone info during codegen
  private final RelDataTypeSystem typeSystem;

  // Bodo verbose level. This is used to generate code/compiler information
  // with extra debugging or logging. 0 is the default verbose level which
  // means no action should be taken. As verboseLevel increases more detailed
  // information can be shown.
  private final int verboseLevel;

  // These Variables track the target table for merge into
  private @Nullable Variable targetTableDf;
  // Extra arguments to pass to the write code for the fileList and Snapshot
  // id in the form of "argName1=varName1, argName2=varName2"
  private @Nullable String fileListAndSnapshotIdArgs;

  private static int RelNodeTimingVerboseLevel = 2;

  public PandasCodeGenVisitor(
      HashMap<String, BodoSQLExprType.ExprType> exprTypesMap,
      HashMap<String, RexNode> searchMap,
      HashMap<String, String> loweredGlobalVariablesMap,
      String originalSQLQuery,
      RelDataTypeSystem typeSystem,
      boolean debuggingDeltaTable,
      int verboseLevel) {
    super();
    this.exprTypesMap = exprTypesMap;
    this.searchMap = searchMap;
    this.varCache = new HashMap<>();
    this.loweredGlobals = loweredGlobalVariablesMap;
    this.originalSQLQuery = originalSQLQuery;
    this.typeSystem = typeSystem;
    this.debuggingDeltaTable = debuggingDeltaTable;
    this.targetTableDf = null;
    this.fileListAndSnapshotIdArgs = null;
    this.verboseLevel = verboseLevel;
    this.generatedCode = new Module.Builder();
  }

  /**
   * Generate the new dataframe variable name for step by step pandas codegen
   *
   * @return variable
   */
  public Variable genDfVar() {
    return generatedCode.getSymbolTable().genDfVar();
  }

  /**
   * Generate the new table variable for step by step pandas codegen
   *
   * @return variable
   */
  public Variable genTableVar() {
    return generatedCode.getSymbolTable().genTableVar();
  }

  /**
   * Generate the new Series variable for step by step pandas codegen
   *
   * @return variable
   */
  public Variable genSeriesVar() {
    return generatedCode.getSymbolTable().genSeriesVar();
  }

  /**
   * Generate a new index variable for step by step pandas codegen
   *
   * @return variable
   */
  public Variable genIndexVar() {
    return generatedCode.getSymbolTable().genIndexVar();
  }

  /**
   * Generate the new temporary variable for step by step pandas codegen.
   *
   * @return variable
   */
  public Variable genGenericTempVar() {
    return generatedCode.getSymbolTable().genGenericTempVar();
  }

  /**
   * Generate the new temporary variable for step by step pandas codegen.
   *
   * @return variable
   */
  public Variable genReaderVar() {
    return generatedCode.getSymbolTable().genReaderVar();
  }

  /**
   * Generate the new variable for a precomputed windowed aggregation column.
   *
   * @return variable
   */
  private Variable genTempColumnVar() {
    return generatedCode.getSymbolTable().genTempColumnVar();
  }

  /**
   * Generate the new variable for a precomputed windowed aggregation dataframe.
   *
   * @return variable
   */
  private Variable genWindowedAggDf() {
    return generatedCode.getSymbolTable().genWindowedAggDf();
  }

  /**
   * generate a new variable used for nested aggregation functions.
   *
   * @return variable
   */
  private Variable genWindowedAggFnVar() {
    return generatedCode.getSymbolTable().genWindowedAggFnName();
  }

  /**
   * generate a new variable for groupby apply functions in agg.
   *
   * @return variable
   */
  private Variable genGroupbyApplyAggFnVar() {
    return generatedCode.getSymbolTable().genGroupbyApplyAggFnName();
  }

  /**
   * Generate the new variable for a streaming accumulator variable
   *
   * @return variable
   */
  private Variable genBatchAccumulatorVar() {
    return generatedCode.getSymbolTable().genBatchAccumulatorVar();
  }

  /**
   * Generate a variable name for the flags keeping track of if a given streaming operation has been
   * exhausted, and no longer has outputs.
   *
   * @return variable name
   */
  public Variable genFinishedStreamingFlag() {
    return generatedCode.getSymbolTable().genFinishedStreamingFlag();
  }

  /**
   * Modifies the codegen such that the specified expression will be lowered into the func_text as a
   * global. This is currently only used for lowering metaDataType's and array types.
   *
   * @return Variable for the global.
   */
  public Variable lowerAsGlobal(Expr expression) {
    Variable globalVar = generatedCode.getSymbolTable().genGlobalVar();
    this.loweredGlobals.put(globalVar.getName(), expression.emit());
    return globalVar;
  }

  /**
   * pass expression as a MetaType global to the generated output function
   *
   * @param expression to pass, e.g. (2, 3, 1)
   * @return variable for the global
   */
  public Variable lowerAsMetaType(Expr expression) {
    Expr typeCall = new Expr.Call("MetaType", List.of(expression));
    return lowerAsGlobal(typeCall);
  }

  /**
   * Modifies the codegen such that the specified expression will be lowered into the func_text as a
   * ColNameMetaType global.
   *
   * @return variable for the global
   */
  public Variable lowerAsColNamesMetaType(Expr expression) {
    Expr typeCall = new Expr.Call("ColNamesMetaType", List.of(expression));
    return lowerAsGlobal(typeCall);
  }

  /**
   * return the final code after step by step pandas codegen
   *
   * @return generated code
   */
  public String getGeneratedCode() {
    // If the stack is size 0 we don't return a DataFrame (e.g. to_sql)
    if (this.varGenStack.size() == 1) {
      this.generatedCode.add(new Op.ReturnStatement(this.varGenStack.pop()));
    }
    assert this.varGenStack.size() == 0
        : "Internal error: varGenStack should contain 1 or 0 values";

    Module m = this.generatedCode.build();
    return m.emit(1);
  }

  /**
   * Basic Visitor Method for all Relational Algebra Nodes
   *
   * @param node RelNode to be visited
   * @param ordinal Ordinal of node within its parent
   * @param parent Parent node calling this visitor
   */
  @Override
  public void visit(RelNode node, int ordinal, RelNode parent) {

    // If we have a SnowflakeToPandasConverter that directly
    // wraps a TableScan, let's extract it so we can use
    // visitTableScan.
    //
    // Ideally, we would not require this. But, because there might
    // be some special logic in the generation of the filter pushdown
    // that we don't want to interfere with (yet), let's intercept
    // it so we aren't changing this code pathway quite yet.
    //
    // In the future, we really want to move filter pushdowns into
    // Calcite itself as that will be an easier and more reliable
    // way to do code generation.
    if (node instanceof SnowflakeToPandasConverter && node.getInput(0) instanceof TableScan) {
      node = node.getInput(0);
    }

    if (node instanceof TableScan) {
      this.visitTableScan((TableScan) node, !(parent instanceof Filter));
    } else if (node instanceof PandasJoin) {
      this.visitPandasJoin((PandasJoin) node);
    } else if (node instanceof PandasSort) {
      this.visitLogicalSort((Sort) node);
    } else if (node instanceof PandasProject) {
      this.visitLogicalProject((Project) node);
    } else if (node instanceof PandasAggregate) {
      this.visitLogicalAggregate((Aggregate) node);
    } else if (node instanceof PandasUnion) {
      this.visitLogicalUnion((Union) node);
    } else if (node instanceof PandasIntersect) {
      this.visitLogicalIntersect((Intersect) node);
    } else if (node instanceof PandasMinus) {
      this.visitLogicalMinus((Minus) node);
    } else if (node instanceof PandasFilter) {
      this.visitLogicalFilter((Filter) node);
    } else if (node instanceof PandasValues) {
      this.visitLogicalValues((Values) node);
    } else if (node instanceof PandasTableModify) {
      this.visitLogicalTableModify((TableModify) node);
    } else if (node instanceof PandasTableCreate) {
      this.visitLogicalTableCreate((PandasTableCreate) node);
    } else if (node instanceof PandasRowSample) {
      this.visitRowSample((PandasRowSample) node);
    } else if (node instanceof PandasSample) {
      this.visitSample((PandasSample) node);
    } else if (node instanceof Correlate) {
      throw new BodoSQLCodegenException(
          "Internal Error: BodoSQL does not support Correlated Queries");
    } else if (node instanceof CombineStreamsExchange) {
      this.visitCombineStreamsExchange((CombineStreamsExchange) node);
    } else if (node instanceof SeparateStreamExchange) {
      this.visitSeparateStreamExchange((SeparateStreamExchange) node);
    } else if (node instanceof PandasRel) {
      this.visitPandasRel((PandasRel) node);
    } else {
      throw new BodoSQLCodegenException(
          "Internal Error: Encountered Unsupported Calcite Node " + node.getClass().toString());
    }
  }

  private void visitCombineStreamsExchange(CombineStreamsExchange node) {
    // For information on how this node handles codegen, please see:
    // https://bodo.atlassian.net/wiki/spaces/B/pages/1337524225/Code+Generation+Design+WIP

    // If we're in a distributed situation, we expect our child to return a distributed dataframe,
    // and a flag that indicates if it's run out of output.
    this.visit(node.getInput(0), 0, node);

    Variable inputDFVar = varGenStack.pop();
    // Generate the list we are accumulating into.
    Variable batchAccumulatorVariable = this.genBatchAccumulatorVar();
    StreamingPipelineFrame activePipeline = this.generatedCode.getCurrentStreamingPipeline();
    activePipeline.addInitialization(
        new Op.Assign(batchAccumulatorVariable, new Expr.List(List.of())));

    // Append to the list at the end of the loop.
    List<Expr> args = new ArrayList<>();
    args.add(inputDFVar);
    Op appendStatement =
        new Op.Stmt(new Expr.Call.Method(batchAccumulatorVariable, "append", args, List.of()));
    generatedCode.add(appendStatement);

    // Pop the pipeline
    StreamingPipelineFrame finishedPipeline = generatedCode.endCurrentStreamingPipeline();
    // Append the pipeline
    generatedCode.add(new Op.StreamingPipeline(finishedPipeline));
    // Finally, concatenate the batches in the accumulator into one dataframe, so that the following
    // operations that
    // expect a single-batch dataframe can operate as needed.
    Variable accumulatedDfVar = this.genDfVar();
    kotlin.Pair<String, Expr.BooleanLiteral> ignoreIndex =
        new kotlin.Pair<>("ignore_index", new Expr.BooleanLiteral(true));
    Expr concatenatedDataFrame =
        new Expr.Call("pd.concat", List.of(batchAccumulatorVariable), List.of(ignoreIndex));
    generatedCode.add(new Op.Assign(accumulatedDfVar, concatenatedDataFrame));
    this.varGenStack.push(accumulatedDfVar);
  }

  private void visitSeparateStreamExchange(SeparateStreamExchange node) {
    // For information on how this node handles codegen, please see:
    // https://bodo.atlassian.net/wiki/spaces/B/pages/1337524225/Code+Generation+Design+WIP

    this.visit(node.getInput(0), 0, node);
    // Since input is single-batch, we know the RHS of the pair must be null
    Variable nonStreamingInput = this.varGenStack.pop();

    // Create the variable that "drives" the loop
    Variable exitCond = genFinishedStreamingFlag();
    generatedCode.startStreamingPipelineFrame(exitCond);
    StreamingPipelineFrame streamingInfo = generatedCode.getCurrentStreamingPipeline();
    // Initialize the iteration.
    Variable iteratorNumber = genGenericTempVar();
    Op.Assign assn = new Op.Assign(iteratorNumber, new Expr.IntegerLiteral(0));
    streamingInfo.addInitialization(assn);

    Variable outputDfVar = genDfVar();
    Expr sliceStart = new Expr.Binary("*", iteratorNumber, BATCH_SIZE);
    Expr sliceEnd =
        new Expr.Binary(
            "*", new Expr.Binary("+", iteratorNumber, new Expr.IntegerLiteral(1)), BATCH_SIZE);

    Expr slicedDfExpr = new Expr.DataFrameSlice(nonStreamingInput, sliceStart, sliceEnd);

    Expr dfLen = new Expr.Call("len", List.of(nonStreamingInput), List.of());

    Expr done = new Expr.Binary(">=", sliceStart, dfLen);

    // Take the current slice of the dataframe
    generatedCode.add(new Op.Assign(outputDfVar, slicedDfExpr));

    // Check if we're at the end
    generatedCode.add(new Op.Assign(exitCond, done));
    // Increment the iterator variable
    generatedCode.add(
        new Op.Assign(
            iteratorNumber, new Expr.Binary("+", iteratorNumber, new Expr.IntegerLiteral(1))));

    this.varGenStack.push(outputDfVar);
  }

  /**
   * Generic visitor method for any RelNode that implements PandasRel.
   *
   * <p>This method handles node caching, visiting inputs, passing those inputs to the node itself
   * to emit code into the Module.Builder, and generating timers when requested.
   *
   * <p>The resulting variable that is generated for this RelNode is placed on the varGenStack.
   *
   * @param node the node to emit code for.
   */
  private void visitPandasRel(PandasRel node) {
    // Determine if this node has already been cached.
    // If it has, just return that immediately.

    // SnowflakeToPandas is a pandasRel,
    // which does not support caching. For now, we've disabled
    // caching for all PandasRels for this reason
    if (isNodeCached(node) && !node.getTraitSet().contains(BatchingProperty.STREAMING)) {
      varGenStack.push(varCache.get(node.getId()));
      return;
    }

    // Currently wrapping this functionality in a closure.
    // We pass this to the emit function so it can decide whether
    // to run this code.
    //
    // The reason for this is because we haven't correctly set a convention
    // and implemented a convention converter. In the calcite code, see the cassandra
    // module, CassandraRel, and CassandraToEnumerableConverter. The last one
    // implements EnumerableRel which allows the enumerable engine built into calcite
    // to invoke the Cassandra relation. The converter then constructs the underlying
    // Cassandra query and executes it. We need a common top-level converter that implements
    // PandaRel. Once that's properly in place, this could be converted to just emitting
    // the inputs eagerly.
    Function0<? extends List<Dataframe>> getInputs =
        () -> {
          // Visit the inputs for this node.
          node.childrenAccept(this);

          // Construct the input variables by popping off the stack.
          // We use reverse on this afterwards since building the list is in reverse order.
          int capacity = node.getInputs().size();
          ImmutableList.Builder<Dataframe> inputs = ImmutableList.builder();
          for (int i = capacity - 1; i >= 0; i--) {
            // TODO(jsternberg): When varGenStack is refactored to natively hold dataframes,
            // the creation of a dataframe here should be removed.
            Dataframe df = new Dataframe(varGenStack.pop().emit(), node.getInput(i));
            inputs.add(df);
          }
          return inputs.build().reverse();
        };

    // Construct the inputs and then invoke the emit function.

    // Can't properly handle timers since some pandas Rels are streaming
    Dataframe out = node.emit(this, generatedCode, getInputs);

    Variable outVal = out.getVariable();
    // Can't handle caching in streaming case (see above)
    if (!node.getTraitSet().contains(BatchingProperty.STREAMING)) {
      varCache.put(node.getId(), outVal);
    }
    varGenStack.push(outVal);
  }

  /**
   * Visitor method for logicalValue Nodes
   *
   * @param node RelNode to be visited
   */
  private void visitLogicalValues(Values node) {
    Variable outVar = this.genDfVar();
    List<String> exprCodes = new ArrayList<>();
    int id = node.getId();
    if (node.getTuples().size() != 0) {
      for (RexLiteral colLiteral : node.getTuples().get(0)) {
        // We cannot be within a case statement since LogicalValues is a RelNode and
        // cannot be inside a case statement (which is a RexNode)
        RexNodeVisitorInfo literalExpr = this.visitLiteralScan(colLiteral, false);
        assert exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(colLiteral, id))
            == BodoSQLExprType.ExprType.SCALAR;
        exprCodes.add(literalExpr.getExprCode());
      }
    }
    this.genRelnodeTimerStart(node);
    Expr logicalValuesExpr = generateLogicalValuesCode(exprCodes, node.getRowType(), this);
    this.generatedCode.add(new Op.Assign(outVar, logicalValuesExpr));
    this.varGenStack.push(outVar);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor method for logicalFilter nodes to support HAVING clause
   *
   * @param node RelNode to be visited
   */
  private void visitLogicalFilter(Filter node) {

    if (node.getTraitSet().contains(BatchingProperty.STREAMING)) {
      this.visitStreamingLogicalFilter(node);
    } else {
      this.visitSingleBatchLogicalFilter(node);
    }
  }

  /**
   * Visitor method for logicalFilter nodes that are NOT streaming.
   *
   * @param node RelNode to be visited
   */
  private void visitSingleBatchLogicalFilter(Filter node) {
    if (this.isNodeCached(node)) {
      Variable outVar = genDfVar();
      int nodeId = node.getId();
      // If the node is in the cache load
      Variable cacheKey = this.varCache.get(nodeId);
      this.generatedCode.add(new Assign(outVar, cacheKey));
      varGenStack.push(outVar);
    } else {
      visitLogicalFilterCommon(node);
    }
  }

  /**
   * Visitor method for logicalFilter nodes that are streaming.
   *
   * @param node RelNode to be visited
   */
  private void visitStreamingLogicalFilter(Filter node) {
    assert node.getTraitSet().contains(BatchingProperty.STREAMING)
        : "Internal error in visitStreamingLogicalFilter: input node not streaming";
    // TODO: determine how to handle caching for a streaming system
    visitLogicalFilterCommon(node);
  }

  /**
   * Code common to both the streaming and non-streaming logicalFilter nodes that are streaming.
   * Handles visiting the input, appending the generated code, and pushing the output dataframe (and
   * flag, if applicable) to the varGenStack.
   *
   * @param node RelNode to be visited
   */
  private void visitLogicalFilterCommon(Filter node) {

    Variable outVar = genDfVar();
    RelNode input = node.getInput();
    this.visit(input, 0, node);
    this.genRelnodeTimerStart(node);
    Variable inVar = varGenStack.pop();
    // The hashset is unused because the result should always be empty.
    // We assume filter is never within an apply.
    RexToPandasTranslator translator = getRexTranslator(node, inVar.emit(), input);
    Expr expr = node.getCondition().accept(translator);
    Expr filterExpr =
        generateFilterCode(
            inVar,
            expr,
            exprTypesMap.get(
                ExprTypeVisitor.generateRexNodeKey(node.getCondition(), node.getId())));
    this.generatedCode.add(new Op.Assign(outVar, filterExpr));
    this.genRelnodeTimerStop(node);

    varGenStack.push(outVar);
  }

  /**
   * Visitor for Logical Union node. Code generation for UNION [ALL/DISTINCT] in SQL
   *
   * @param node LogicalUnion node to be visited
   */
  private void visitLogicalUnion(Union node) {
    List<Variable> childExprs = new ArrayList<>();
    List<List<String>> childExprsColumns = new ArrayList<>();
    // Visit all of the inputs
    for (int i = 0; i < node.getInputs().size(); i++) {
      RelNode input = node.getInput(i);
      this.visit(input, i, node);
      childExprs.add(varGenStack.pop());
      childExprsColumns.add(input.getRowType().getFieldNames());
    }
    this.genRelnodeTimerStart(node);
    Variable outVar = this.genDfVar();
    List<String> columnNames = node.getRowType().getFieldNames();
    Expr unionExpr = generateUnionCode(columnNames, childExprs, node.all, this);
    this.generatedCode.add(new Op.Assign(outVar, unionExpr));
    varGenStack.push(outVar);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor for Logical Intersect node. Code generation for INTERSECT [ALL/DISTINCT] in SQL
   *
   * @param node LogicalIntersect node to be visited
   */
  private void visitLogicalIntersect(Intersect node) {
    // We always assume intersect is between exactly two inputs
    if (node.getInputs().size() != 2) {
      throw new BodoSQLCodegenException(
          "Internal Error: Intersect should be between exactly two inputs");
    }

    // Visit the two inputs
    RelNode lhs = node.getInput(0);
    this.visit(lhs, 0, node);
    Variable lhsExpr = varGenStack.pop();
    List<String> lhsColNames = lhs.getRowType().getFieldNames();

    RelNode rhs = node.getInput(1);
    this.visit(rhs, 1, node);
    Variable rhsExpr = varGenStack.pop();
    List<String> rhsColNames = rhs.getRowType().getFieldNames();

    Variable outVar = this.genDfVar();
    List<String> colNames = node.getRowType().getFieldNames();
    this.genRelnodeTimerStart(node);

    this.generatedCode.addAll(
        generateIntersectCode(
            outVar, lhsExpr, lhsColNames, rhsExpr, rhsColNames, colNames, node.all, this));

    varGenStack.push(outVar);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor for Logical Minus node. Code generation for EXCEPT/MINUS in SQL
   *
   * @param node LogicalMinus node to be visited
   */
  private void visitLogicalMinus(Minus node) {
    // We always assume minus is between exactly two inputs
    if (node.getInputs().size() != 2) {
      throw new BodoSQLCodegenException(
          "Internal Error: Except should be between exactly two inputs");
    }

    // Visit the two inputs
    RelNode lhs = node.getInput(0);
    this.visit(lhs, 0, node);
    Variable lhsVar = varGenStack.pop();
    List<String> lhsColNames = lhs.getRowType().getFieldNames();

    RelNode rhs = node.getInput(1);
    this.visit(rhs, 1, node);
    Variable rhsVar = varGenStack.pop();
    List<String> rhsColNames = rhs.getRowType().getFieldNames();

    assert lhsColNames.size() == rhsColNames.size();
    assert lhsColNames.size() > 0 && rhsColNames.size() > 0;

    Variable outVar = this.genDfVar();
    List<String> colNames = node.getRowType().getFieldNames();
    this.genRelnodeTimerStart(node);

    this.generatedCode.addAll(
        generateExceptCode(
            outVar, lhsVar, lhsColNames, rhsVar, rhsColNames, colNames, node.all, this));

    varGenStack.push(outVar);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor for Logical Sort node. Code generation for ORDER BY clauses in SQL
   *
   * @param node Logical Sort node to be visited
   */
  public void visitLogicalSort(Sort node) {
    RelNode input = node.getInput();
    this.visit(input, 0, node);
    this.genRelnodeTimerStart(node);
    Variable inVar = varGenStack.pop();
    List<String> colNames = input.getRowType().getFieldNames();
    /* handle case for queries with "ORDER BY" clause */
    List<RelFieldCollation> sortOrders = node.getCollation().getFieldCollations();
    Variable outVar = this.genDfVar();
    String limitStr = "";
    String offsetStr = "";
    /* handle case for queries with "LIMIT" clause */
    RexNode fetch = node.fetch; // for a select query including a clause LIMIT N, fetch returns N.
    if (fetch != null) {
      // Check type for fetch. If its not an integer it shouldn't be a legal limit.
      // This is handled by the parser for all situations except namedParams
      // TODO: Determine how to move this into Calcite
      SqlTypeName typeName = fetch.getType().getSqlTypeName();
      if ((typeName != SqlTypeName.TINYINT)
          && (typeName != SqlTypeName.SMALLINT)
          && (typeName != SqlTypeName.INTEGER)
          && (typeName != SqlTypeName.BIGINT)) {
        throw new BodoSQLCodegenException(
            "Limit value must be an integer, value is of type: "
                + sqlTypenameToPandasTypename(typeName, true));
      }

      // fetch is either a named Parameter or a literal from parsing.
      // We visit the node to resolve the name.
      RexToPandasTranslator translator = getRexTranslator(node, inVar.emit(), input);
      limitStr = fetch.accept(translator).emit();
    }
    RexNode offset = node.offset;
    if (offset != null) {
      // Check type for fetch. If its not an integer it shouldn't be a legal offset.
      // This is handled by the parser for all situations except namedParams
      // TODO: Determine how to move this into Calcite
      SqlTypeName typeName = offset.getType().getSqlTypeName();
      if ((typeName != SqlTypeName.TINYINT)
          && (typeName != SqlTypeName.SMALLINT)
          && (typeName != SqlTypeName.INTEGER)
          && (typeName != SqlTypeName.BIGINT)) {
        throw new BodoSQLCodegenException(
            "Offset value must be an integer, value is of type: "
                + sqlTypenameToPandasTypename(typeName, true));
      }

      // Offset is either a named Parameter or a literal from parsing.
      // We visit the node to resolve the name.
      RexToPandasTranslator translator = getRexTranslator(node, inVar.emit(), input);
      offsetStr = offset.accept(translator).emit();
    }

    Expr sortExpr = generateSortCode(inVar, colNames, sortOrders, limitStr, offsetStr);
    this.generatedCode.add(new Op.Assign(outVar, sortExpr));
    varGenStack.push(outVar);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor for LogicalProject node.
   *
   * @param node LogicalProject node to be visited
   */
  public void visitLogicalProject(Project node) {

    if (node.getTraitSet().contains(BatchingProperty.STREAMING)) {
      this.visitStreamingLogicalProject(node);
    } else {
      this.visitSingleBatchLogicalProject(node);
    }
  }

  /**
   * Visitor for a non-streaming LogicalProject node.
   *
   * @param node LogicalProject node to be visited
   */
  public void visitSingleBatchLogicalProject(Project node) {

    if (this.isNodeCached(node)) {
      int nodeId = node.getId();
      Variable projectOutVar = this.genDfVar();
      Variable cacheKey = this.varCache.get(nodeId);
      Op outputCode = new Assign(projectOutVar, cacheKey);
      this.generatedCode.add(outputCode);
      varGenStack.push(projectOutVar);
      return;
    }

    this.genRelnodeTimerStart(node);
    this.visitLogicalProjectCommon(node);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor for a streaming LogicalProject node.
   *
   * @param node LogicalProject node to be visited
   */
  public void visitStreamingLogicalProject(Project node) {
    assert node.getTraitSet().contains(BatchingProperty.STREAMING)
        : "Internal error in visitLogicalProjectStreaming: Input node not streaming";
    this.genRelnodeTimerStart(node);
    this.visitLogicalProjectCommon(node);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Helper function for the code that is common between a streaming and non-streaming
   * LogicalProject node.
   *
   * @param node LogicalProject node to be visited
   */
  public void visitLogicalProjectCommon(Project node) {
    Variable projectOutVar = this.genDfVar();
    List<String> colNames = node.getInput().getRowType().getFieldNames();
    // Output column names selected.
    List<String> outputColumns = new ArrayList();
    int nodeId = node.getId();
    this.visit(node.getInput(), 0, node);
    this.genRelnodeTimerStart(node);
    Variable projectInVar = varGenStack.pop();
    // ChildExprs operations produced.
    List<RexNodeVisitorInfo> childExprs = new ArrayList<>();
    // ExprTypes used for code generation
    List<BodoSQLExprType.ExprType> exprTypes = new ArrayList<>();
    // Check for all InputRefs. If so then we optimize to output df.loc
    boolean useLoc = true;
    // Use a tree map so the ordering is consistent across ranks.
    TreeMap<String, String> renameMap = new TreeMap<>();
    HashSet<Integer> seenIdxs = new HashSet<>();
    for (Pair<RexNode, String> named_r : node.getNamedProjects()) {
      RexNode r = named_r.getKey();
      String alias = named_r.getValue();
      if (!(r instanceof RexInputRef)) {
        // If we have a non input ref we can't use the loc path
        useLoc = false;
        break;
      }
      RexInputRef inputRef = (RexInputRef) r;
      if (seenIdxs.contains(inputRef.getIndex())) {
        /**
         * When we have a situation with common subexpressions like "sum(A) as alias2, sum(A) as
         * alias from table1 groupby D" Calcite generates a plan like: LogicalProject(alias2=[$1],
         * alias=[$1]) LogicalAggregate(group=[{0}], alias=[SUM($1)]) In this case, we can't use
         * loc, as it would lead to duplicate column names in the output dataframe See
         * test_repeat_columns in BodoSQL/bodosql/tests/test_agg_groupby.py
         */
        useLoc = false;
        break;
      }
      seenIdxs.add(inputRef.getIndex());
      String colName = colNames.get(inputRef.getIndex());
      // If we have a rename and useLoc=True
      // we do a rename after df.loc.
      if (!colName.equals(alias)) {
        renameMap.put(colName, alias);
      }
    }

    // for windowed aggregations, we put them all into the list, and pass the list to
    // visitAggOverOp,
    // which handles fusion of windowed aggregations.
    List<RexOver> windowAggList = new ArrayList<>();
    // We also need to keep track of the original indexes, so that we can properly update
    // outputColumns and childExprs after processing the windowed aggregation
    List<Integer> idx_list = new ArrayList<>();

    if (useLoc) {
      // If we are just a doing a loc we may want to load from cache. Needed for caching/reusing
      // joins.
      Expr locExpr = generateLocCode(projectInVar, node.getProjects(), colNames, outputColumns);
      if (!renameMap.isEmpty()) {

        locExpr = generateRenameCode(locExpr, renameMap);
        // Update the output columns with the mapping.
        List<String> oldOutputColumns = outputColumns;
        outputColumns = new ArrayList<>();
        for (String col : oldOutputColumns) {
          if (renameMap.containsKey(col)) {
            outputColumns.add(renameMap.get(col));
          } else {
            outputColumns.add(col);
          }
        }
      }
      this.generatedCode.add(new Op.Assign(projectOutVar, locExpr));
      // Currently we only cache projections if it has a loc. This is because the
      // projection can cause functions that trigger data generation (e.g random
      // values). In the future we should optimize this code to directly check for
      // random data generation and cache in all other cases.
      // NOTE: This varCache is currently unused in the streaming case.
      this.varCache.put(nodeId, projectOutVar);
    } else {
      List<RelDataTypeField> fields = node.getRowType().getFieldList();
      List<RelDataType> sqlTypes = new ArrayList<>();
      for (int i = 0; i < node.getNamedProjects().size(); i++) {
        Pair<RexNode, String> named_r = node.getNamedProjects().get(i);
        RexNode r = named_r.getKey();
        outputColumns.add(named_r.getValue());
        exprTypes.add(exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(r, node.getId())));
        sqlTypes.add(fields.get(i).getType());

        if (r instanceof RexOver) {
          windowAggList.add((RexOver) r);
          idx_list.add(i);
          // append dummy values to childExprs.
          // This value will be overwritten, once we are done processing the windowed aggregations
          childExprs.add(new RexNodeVisitorInfo("DUMMY_VAL"));
          continue;
        }

        /* Handles:
            - "SELECT <COLUMNNAMES> FROM TABLE",
            - "SELECT SUM(COLUMN), COUNT(COLUMN), MAX(COLUMN), MIN(COLUMN) FROM TABLE"
            - "SELECT COLUMN +/-/* CONST FROM TABLE" (includes nesting expressions)
        */
        // The BodoCtx is unused because the result should always be empty.
        // We assume project is never within an apply.
        RexNodeVisitorInfo childExpr;
        // Pass input column index for RexInputRef to generateProjectCode() directly instead of
        // df["A"] codegen
        // Allows generateProjectCode() to keep data in table format as much as possible
        if (r instanceof RexInputRef) {
          childExpr = new RexNodeVisitorInfo("", ((RexInputRef) r).getIndex());
        } else {
          childExpr =
              new RexNodeVisitorInfo(
                  r.accept(getRexTranslator(node, projectInVar.emit(), node.getInput())).emit());
        }
        childExprs.add(childExpr);
      }
      // handle windowed Aggregations
      Dataframe dfInput = new Dataframe(projectInVar.emit(), node.getInput());
      List<Expr> windowAggInfo =
          visitAggOverOp(windowAggList, colNames, node.getId(), dfInput, false, new BodoCtx());
      // check that these all have the same len
      assert windowAggInfo.size() == windowAggList.size()
          && windowAggInfo.size() == idx_list.size();
      for (int i = 0; i < windowAggList.size(); i++) {
        int origIdx = idx_list.get(i);
        // grab the original alias from the value stored value in childExprs
        RexNodeVisitorInfo realChildExpr = new RexNodeVisitorInfo(windowAggInfo.get(i).emit());
        // overwrite the dummy value that we stored earlier in childExpr
        childExprs.set(origIdx, realChildExpr);
      }

      List<Op.Assign> projectAssigns =
          generateProjectCode(
              projectInVar,
              projectOutVar,
              outputColumns,
              childExprs,
              exprTypes,
              sqlTypes,
              this,
              colNames.size());
      this.generatedCode.addAll(projectAssigns);
    }

    varGenStack.push(projectOutVar);
  }

  public void visitLogicalTableCreate(PandasTableCreate node) {
    this.visit(node.getInput(0), 0, node);
    // Not going to do a relNodeTimer on this

    Schema outputSchema = node.getSchema();

    // Fow now, we only support CREATE TABLE AS with CatalogSchema
    if (!(outputSchema instanceof CatalogSchemaImpl)) {
      throw new BodoSQLCodegenException(
          "CREATE TABLE is only supported for Snowflake Catalog Schemas");
    }

    CatalogSchemaImpl outputSchemaAsCatalog = (CatalogSchemaImpl) outputSchema;

    BodoSQLCatalog.ifExistsBehavior ifExists;
    if (node.isReplace()) {
      ifExists = BodoSQLCatalog.ifExistsBehavior.REPLACE;
    } else {
      ifExists = BodoSQLCatalog.ifExistsBehavior.FAIL;
    }

    SqlCreateTable.CreateTableType createTableType = node.getCreateTableType();

    this.generatedCode.add(
        new Op.Stmt(
            outputSchemaAsCatalog.generateWriteCode(
                this.varGenStack.pop(), node.getTableName(), ifExists, createTableType)));
  }

  /**
   * Visitor for LogicalTableModify, which is used to support certain SQL write operations.
   *
   * @param node LogicalTableModify node to be visited
   */
  public void visitLogicalTableModify(TableModify node) {
    switch (node.getOperation()) {
      case INSERT:
        this.visitInsertInto(node);
        break;
      case MERGE:
        this.visitMergeInto(node);
        break;
      case DELETE:
        this.visitDelete(node);
        break;
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Encountered Unsupported Calcite Modify operation "
                + node.getOperation().toString());
    }
  }

  /**
   * Visitor for MERGE INTO operation for SQL write. Currently, it just returns the delta table, for
   * testing purposes.
   *
   * @param node
   */
  public void visitMergeInto(TableModify node) {
    assert node.getOperation() == TableModify.Operation.MERGE;

    RelNode input = node.getInput();
    this.visit(input, 0, node);
    this.genRelnodeTimerStart(node);
    Variable deltaDfVar = this.varGenStack.pop();
    List<String> currentDeltaDfColNames = input.getRowType().getFieldNames();

    if (this.debuggingDeltaTable) {
      // If this environment variable is set, we're only testing the generation of the delta table.
      // Just return the delta table.
      // We drop no-ops from the delta table, as a few Calcite Optimizations can result in their
      // being removed from the table, and their presence/lack thereof shouldn't impact anything in
      // the
      // final implementation, but it can cause issues when testing the delta table
      Variable outputVar = this.genDfVar();
      this.generatedCode.add(
          new Op.Assign(
              outputVar,
              new Expr.Raw(
                  new StringBuilder()
                      .append(getBodoIndent())
                      .append(outputVar.emit())
                      .append(" = ")
                      .append(deltaDfVar.emit())
                      .append(".dropna(subset=[")
                      .append(
                          makeQuoted(currentDeltaDfColNames.get(currentDeltaDfColNames.size() - 1)))
                      .append("])")
                      .toString())));

      this.generatedCode.add(new Op.ReturnStatement(outputVar));
    } else {
      // Assert that we've encountered a PandasTargetTableScan in the codegen, and
      // set the appropriate variables
      assert targetTableDf != null;
      assert fileListAndSnapshotIdArgs != null;

      RelOptTableImpl relOptTable = (RelOptTableImpl) node.getTable();
      BodoSqlTable bodoSqlTable = (BodoSqlTable) relOptTable.table();
      if (!(bodoSqlTable.isWriteable() && bodoSqlTable.getDBType().equals("ICEBERG"))) {
        throw new BodoSQLCodegenException(
            "MERGE INTO is only supported with Iceberg table destinations provided via the the"
                + " SQL TablePath API");
      }

      // note table.getColumnNames does NOT include ROW_ID or MERGE_ACTION_ENUM_COL_NAME column
      // names,
      // because of the way they are added plan in calcite (extension fields)
      // We know that the row ID and merge columns exist in the input table due to our code
      // invariants
      List<String> targetTableFinalColumnNames = bodoSqlTable.getColumnNames();
      List<String> deltaTableExpectedColumnNames;
      deltaTableExpectedColumnNames = new ArrayList<>(targetTableFinalColumnNames);
      deltaTableExpectedColumnNames.add(ROW_ID_COL_NAME);
      deltaTableExpectedColumnNames.add(MERGE_ACTION_ENUM_COL_NAME);

      Variable writebackDf = genDfVar();

      Expr renamedDeltaDfVar =
          handleRename(deltaDfVar, currentDeltaDfColNames, deltaTableExpectedColumnNames);
      this.generatedCode.add(
          new Op.Assign(
              writebackDf,
              new Expr.Raw(
                  new StringBuilder()
                      .append("bodosql.libs.iceberg_merge_into.do_delta_merge_with_target(")
                      .append(this.targetTableDf.emit())
                      .append(", ")
                      .append(renamedDeltaDfVar.emit())
                      .append(")")
                      .toString())));

      // TODO: this can just be cast, since we handled rename
      Expr castedAndRenamedWriteBackDfExpr =
          handleCastAndRenameBeforeWrite(writebackDf, targetTableFinalColumnNames, bodoSqlTable);
      Variable castedAndRenamedWriteBackDfVar = this.genDfVar();
      this.generatedCode.add(
          new Op.Assign(castedAndRenamedWriteBackDfVar, castedAndRenamedWriteBackDfExpr));
      this.generatedCode.add(
          new Op.Stmt(
              bodoSqlTable.generateWriteCode(
                  castedAndRenamedWriteBackDfVar, this.fileListAndSnapshotIdArgs)));
    }
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor for Insert INTO operation for SQL write.
   *
   * @param node LogicalTableModify node to be visited
   */
  public void visitInsertInto(TableModify node) {
    // Insert into consists of two steps:
    // 1. Create a projection to write
    // 2. Generate the "write" code.
    //
    // We currently only support insertInto for SQL operations that
    // write directly to a DataBase. As a result we make the following
    // assumption:
    //
    // We can always write with the `to_sql` API. This should be checked
    // and enforced at a prior stage that prevents in memory updates.
    //
    // At this time we don't make any additional optimizations, such as omitting
    // NULL columns.

    // Generate code for a projection.
    this.visit(node.getInput(), 0, node);
    this.genRelnodeTimerStart(node);
    // Generate the to_sql code
    List<String> colNames = node.getInput().getRowType().getFieldNames();
    Variable inVar = this.varGenStack.pop();
    RelOptTableImpl relOptTable = (RelOptTableImpl) node.getTable();
    BodoSqlTable bodoSqlTable = (BodoSqlTable) relOptTable.table();
    if (!bodoSqlTable.isWriteable()) {
      throw new BodoSQLCodegenException(
          "Insert Into is only supported with table destinations provided via the Snowflake"
              + "catalog or the SQL TablePath API");
    }

    Expr castedAndRenamedDfExpr = handleCastAndRenameBeforeWrite(inVar, colNames, bodoSqlTable);
    Variable castedAndRenamedDfVar = this.genDfVar();
    this.generatedCode.add(new Op.Assign(castedAndRenamedDfVar, castedAndRenamedDfExpr));
    this.generatedCode.add(new Op.Stmt(bodoSqlTable.generateWriteCode(castedAndRenamedDfVar)));
    this.genRelnodeTimerStop(node);
  }

  /**
   * Returns an expression which is the input dataframe with the appropriate casting/column renaming
   * required for writing to the specified table
   *
   * <p>May append intermediate variables to generated code.
   *
   * @param inVar The input dataframe to be casted/renamed
   * @param colNames Expected colNames of the output expression
   * @param bodoSqlTable The table to be written to.
   * @return
   */
  public Expr handleCastAndRenameBeforeWrite(
      Variable inVar, List<String> colNames, BodoSqlTable bodoSqlTable) {
    Expr outputExpr = inVar;

    // Do the cast and update outputExpr if needed
    Expr castExpr = bodoSqlTable.generateWriteCastCode(inVar);
    if (!castExpr.emit().equals("")) {
      outputExpr = castExpr;
    }

    Variable intermediateDf = this.genDfVar();
    this.generatedCode.add(new Op.Assign(intermediateDf, outputExpr));

    // Update column names to the write names.
    outputExpr = handleRename(intermediateDf, colNames, bodoSqlTable.getWriteColumnNames());
    return outputExpr;
  }

  /**
   * Creates an expression that renames the columns of the input dataFrame variable. Simply returns
   * the dataframe variable if no renaming is needed.
   *
   * @param inVar The input dataframe variable
   * @param oldColNames The old column names
   * @param newColNames The new column names
   * @return
   */
  public Expr handleRename(Variable inVar, List<String> oldColNames, List<String> newColNames) {
    assert oldColNames.size() == newColNames.size();
    StringBuilder outputCode = new StringBuilder(inVar.emit());
    boolean hasRename = false;
    for (int i = 0; i < newColNames.size(); i++) {
      if (!oldColNames.get(i).equals(newColNames.get(i))) {
        if (!hasRename) {
          // Only generate the rename if at least 1 column needs renaming to avoid any empty
          // dictionary issues.
          outputCode.append(".rename(columns={");
          hasRename = true;
        }
        outputCode.append(makeQuoted(oldColNames.get(i)));
        outputCode.append(" : ");
        outputCode.append(makeQuoted(newColNames.get(i)));
        outputCode.append(", ");
      }
    }
    if (hasRename) {
      outputCode.append("}, copy=False)");
    }
    return new Expr.Raw(outputCode.toString());
  }

  /**
   * Visitor for SQL Delete Operation with a remote database. We currently only support delete via
   * our Snowflake Catalog.
   *
   * <p>Note: This operation DOES NOT support caching as it has side effects.
   */
  public void visitDelete(TableModify node) {
    RelOptTableImpl relOptTable = (RelOptTableImpl) node.getTable();
    BodoSqlTable bodoSqlTable = (BodoSqlTable) relOptTable.table();
    Variable outputVar = this.genDfVar();
    List<String> outputColumns = node.getRowType().getFieldNames();
    if (isSnowflakeCatalogTable(bodoSqlTable)) {
      this.genRelnodeTimerStart(node);
      // If we are updating Snowflake we push the query into Snowflake.
      // We require the Snowflake Catalog to ensure we don't need to remap
      // any names.
      try {
        this.generatedCode.add(
            new Op.Assign(outputVar, bodoSqlTable.generateRemoteQuery(this.originalSQLQuery)));

      } catch (RuntimeException e) {
        // If we encounter an exception we cannot push the query into Snowflake.
        String errorMsg =
            "BodoSQL implements Delete for Snowflake tables by pushing the entire query into"
                + " Snowflake.\n"
                + "Please verify that all of your Delete query syntax is supported inside of"
                + " Snowflake and doesn't contain any BodoSQL Specific Features.\n"
                + "Detailed Error:\n"
                + e.getMessage();
        throw new RuntimeException(errorMsg);
      }
      // Update the column names to ensure they match as we don't know the
      // Snowflake names right now
      // TODO: FIX the LHS here in the IR
      Variable columnsVar = new Variable(outputVar.getName() + ".columns");
      List<Expr.StringLiteral> colNames = new ArrayList<>();
      for (String colName : outputColumns) {
        colNames.add(new StringLiteral(colName));
      }
      Expr.List colNamesExpr = new Expr.List(colNames);
      this.generatedCode.add(new Op.Assign(columnsVar, colNamesExpr));
      this.genRelnodeTimerStop(node);
    } else {
      throw new BodoSQLCodegenException(
          "Delete only supported when all source tables are found within a user's Snowflake"
              + " account and are provided via the Snowflake catalog.");
    }
    this.varGenStack.push(outputVar);
  }

  /**
   * Visitor for Logical Aggregate, support for Aggregations in SQL such as SUM, COUNT, MIN, MAX.
   *
   * @param node LogicalAggregate node to be visited
   */
  public void visitLogicalAggregate(Aggregate node) {
    final List<Integer> groupingVariables = node.getGroupSet().asList();
    final List<ImmutableBitSet> groups = node.getGroupSets();

    // Based on the calcite code that we've seen generated, we assume that every Logical Aggregation
    // node has
    // at least one grouping set.
    assert groups.size() > 0;

    List<String> expectedOutputCols = node.getRowType().getFieldNames();

    int nodeId = node.getId();
    Variable finalOutVar = this.genDfVar();
    if (isNodeCached(node)) {
      Variable cacheKey = this.varCache.get(nodeId);
      this.generatedCode.add(new Op.Assign(finalOutVar, cacheKey));
    } else {
      final List<AggregateCall> aggCallList = node.getAggCallList();

      // Expected output column names according to the calcite plan, contains any/all of the
      // expected
      // aliases

      List<String> aggCallNames = new ArrayList<>();
      for (int i = 0; i < aggCallList.size(); i++) {
        AggregateCall aggregateCall = aggCallList.get(i);

        // Check our assumptions about the aggregateCall names
        // I have never seen this be true in a finalized calcite plan, but ensure that we throw a
        // reasonable error
        // just in case
        if (aggregateCall.getName() == null) {
          aggCallNames.add(expectedOutputCols.get(groupingVariables.size() + i));
        } else if (!aggregateCall
            .getName()
            .equals(expectedOutputCols.get(groupingVariables.size() + i))) {
          // In the case that the aggregateCall is named, it always has the same name as the
          // expected
          // output column.
          throw new BodoSQLCodegenException(
              "Unexpected Calcite plan generated: Unable to find aggregateCall name not present in"
                  + " the expected location in the output column names.");
        } else {
          aggCallNames.add(aggregateCall.getName());
        }
      }

      RelNode inputNode = node.getInput();
      this.visit(inputNode, 0, node);
      this.genRelnodeTimerStart(node);
      List<String> inputColumnNames = inputNode.getRowType().getFieldNames();
      Variable inVar = varGenStack.pop();
      List<String> outputDfNames = new ArrayList<>();

      // If any group is missing a column we may need to do a concat.
      boolean hasMissingColsGroup = false;

      boolean distIfNoGroup = groups.size() > 1;

      // Naive implementation for handling multiple aggregation groups, where we repeatedly call
      // group
      // by, and append the dataframes together
      for (int i = 0; i < groups.size(); i++) {
        List<Integer> curGroup = groups.get(i).toList();

        hasMissingColsGroup = hasMissingColsGroup || curGroup.size() < groupingVariables.size();
        Expr curGroupAggExpr;
        /* First rename any input keys to the output. */

        /* group without aggregation : e.g. select B from table1 groupby A */
        if (aggCallList.isEmpty()) {
          curGroupAggExpr = generateAggCodeNoAgg(inVar, inputColumnNames, curGroup);
        }
        /* aggregate without group : e.g. select sum(A) from table1 */
        else if (curGroup.isEmpty()) {
          curGroupAggExpr =
              generateAggCodeNoGroupBy(
                  inVar, inputColumnNames, aggCallList, aggCallNames, distIfNoGroup);
        }
        /* group with aggregation : e.g. select sum(B) from table1 groupby A */
        else {
          Pair<Expr, @Nullable Op> curGroupAggExprAndAdditionalGeneratedCode =
              handleLogicalAggregateWithGroups(
                  inVar, inputColumnNames, aggCallList, aggCallNames, curGroup);

          curGroupAggExpr = curGroupAggExprAndAdditionalGeneratedCode.getKey();
          @Nullable Op prependOp = curGroupAggExprAndAdditionalGeneratedCode.getValue();
          if (prependOp != null) {
            this.generatedCode.add(prependOp);
          }
        }
        // assign each of the generated dataframes their own variable, for greater clarity in the
        // generated code
        Variable outDf = this.genDfVar();
        outputDfNames.add(outDf.getName());
        this.generatedCode.add(new Op.Assign(outDf, curGroupAggExpr));
      }
      // If we have multiple groups, append the dataframes together
      if (groups.size() > 1 || hasMissingColsGroup) {
        // It is not guaranteed that a particular input column exists in any of the output
        // dataframes,
        // but Calcite expects
        // All input dataframes to be carried into the output. It is also not
        // guaranteed that the output dataframes contain the columns in the order expected by
        // calcite.
        // In order to ensure that we have all the input columns in the output,
        // we create a dummy dataframe that has all the columns with
        // a length of 0. The ordering is handled by a loc after the concat

        // We initialize the dummy column like this, as Bodo will default these columns to string
        // type
        // if we
        // initialize empty columns.
        List<String> concatDfs = new ArrayList<>();
        if (hasMissingColsGroup) {
          Variable dummyDfVar = genDfVar();
          // TODO: Switch to proper IR
          Expr dummyDfExpr = new Expr.Raw(inVar.getName() + ".iloc[:0, :]");
          // Assign the dummy df to a variable name,
          this.generatedCode.add(new Op.Assign(dummyDfVar, dummyDfExpr));
          concatDfs.add(dummyDfVar.emit());
        }
        concatDfs.addAll(outputDfNames);

        // Generate the concatenation expression
        StringBuilder concatExprRaw = new StringBuilder(concatDataFrames(concatDfs).emit());

        // Sort the output dataframe, so that they are in the ordering expected by Calcite
        // Needed in the case that the topmost dataframe in the concat does not contain all the
        // columns in the correct ordering
        concatExprRaw.append(".loc[:, [");

        for (int i = 0; i < expectedOutputCols.size(); i++) {
          concatExprRaw.append(makeQuoted(expectedOutputCols.get(i))).append(", ");
        }
        concatExprRaw.append("]]");

        // Generate the concatenation
        this.generatedCode.add(new Op.Assign(finalOutVar, new Expr.Raw(concatExprRaw.toString())));

      } else {
        finalOutVar = new Variable(outputDfNames.get(0));
      }
      this.varCache.put(nodeId, finalOutVar);
      this.genRelnodeTimerStop(node);
    }
    varGenStack.push(finalOutVar);
  }

  /**
   * Generates an expression for a Logical Aggregation with grouped variables. May return code to be
   * appended to the generated code (this is needed if the aggregation list contains a filter).
   *
   * @param inVar The input variable.
   * @param inputColumnNames The names of the columns of the input var.
   * @param aggCallList The list of aggregations to be performed.
   * @param aggCallNames The list of column names to be used for the output of the aggregations
   * @param group List of integer column indices by which to group
   * @return A pair of strings, the key is the expression that evaluates to the output of the
   *     aggregation, and the value is the code that needs to be appended to the generated code.
   */
  public Pair<Expr, @Nullable Op> handleLogicalAggregateWithGroups(
      Variable inVar,
      List<String> inputColumnNames,
      List<AggregateCall> aggCallList,
      List<String> aggCallNames,
      List<Integer> group) {

    Pair<Expr, Op> exprAndAdditionalGeneratedCode;
    if (aggContainsFilter(aggCallList)) {
      // If we have a Filter we need to generate a groupby apply
      exprAndAdditionalGeneratedCode =
          generateApplyCodeWithGroupBy(
              inVar,
              inputColumnNames,
              group,
              aggCallList,
              aggCallNames,
              this.genGroupbyApplyAggFnVar());

    } else {
      // Otherwise generate groupby.agg
      Expr groupbyExpr =
          generateAggCodeWithGroupBy(inVar, inputColumnNames, group, aggCallList, aggCallNames);
      exprAndAdditionalGeneratedCode = new Pair<>(groupbyExpr, null);
    }

    return exprAndAdditionalGeneratedCode;
  }

  /**
   * Return a pandas expression that replicates a SQL windowed aggregation function.
   *
   * @param aggOperations List of internal operator RexCall being visited
   * @param colNames List of colNames used in the relational expression
   * @param id The RelNode id used to uniquely identify the table.
   * @param inputVar Name of dataframe from which InputRefs select Columns
   * @param isSingleRow Boolean for if table references refer to a single row or the whole table.
   *     Operations that operate per row (i.e. Case switch this to True). This is used for
   *     determining if an expr returns a scalar or a column.
   * @param ctx A ctx object containing the Hashset of columns used that need null handling, the
   *     List of precomputed column variables that need to be added to the dataframe before an
   *     apply, and the list of named parameters that need to be passed to an apply function as
   *     arguments.
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public List<Expr> visitAggOverOp(
      List<RexOver> aggOperations,
      List<String> colNames,
      int id,
      Dataframe inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {

    final String indent = getBodoIndent();
    if (aggOperations.size() == 0) {
      return new ArrayList<>();
    }

    RexToPandasTranslator translator = getRexTranslator(id, inputVar);

    // Check if we can use the optimized groupby.window C++ kernel. If so
    // we directly to that codegen path.
    if (usesOptimizedEngineKernel(aggOperations)) {
      // usesOptimizedEngineKernel enforces that we have exactly 1 aggOperation
      // and exactly 1 order by column.
      RexOver windowFunc = aggOperations.get(0);
      RexWindow window = windowFunc.getWindow();
      // We need to visit all partition keys before generating
      // code for the kernel
      List<Expr> childExprs = new ArrayList<>();
      List<BodoSQLExprType.ExprType> childExprTypes = new ArrayList<>();
      for (int i = 0; i < window.partitionKeys.size(); i++) {
        RexNode node = window.partitionKeys.get(i);
        Expr curInfo = node.accept(translator);
        childExprs.add(curInfo);
        childExprTypes.add(exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(node, id)));
      }
      // Visit the order by key. This is required to be a single column.
      // The RHS of order keys is a set of SqlKind Values. These are used to stored information
      // about ascending and nulls first/last
      for (int i = 0; i < window.orderKeys.size(); i++) {
        RexNode node = window.orderKeys.get(i).left;
        Expr curRexInfo = node.accept(translator);
        childExprs.add(curRexInfo);
        childExprTypes.add(exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(node, id)));
      }
      Variable outputVar = this.genWindowedAggDf();
      Expr outputCode =
          generateOptimizedEngineKernelCode(
              inputVar.getVariable().getName(),
              outputVar,
              aggOperations,
              childExprs,
              childExprTypes,
              this.generatedCode);

      return Collections.singletonList(outputCode);
    }

    // The overall goal of this section of code is to separate the input list
    // of agg operations into groups that can be done in the same group-by
    // function.
    // These are the variables used to store each distinct window and the
    // list of corresponding window function calls for each group.
    List<List<RexWindow>> windows = new ArrayList<>();
    List<List<SqlKind>> fnKinds = new ArrayList<>();
    List<List<String>> fnNames = new ArrayList<>();
    List<List<RexOver>> aggSets = new ArrayList<>();
    List<List<Boolean>> respectNullsLists = new ArrayList<>();

    // Used to store how each output within each groupby-apply corresponds
    // to one of the original column outputs in the SELECT statement
    // indices.get(x).get(y) = z means that "group by apply function" number x's
    // y'th output is the equivalent expression for aggOperations.get(z)
    List<List<Integer>> indices = new ArrayList<>();

    // Eventually used to store the names of each output
    List<String> outputColExprs = new ArrayList<>();

    // Loop over each aggregation and identify if it can be fused with one
    // of the aggregations already added to the aggSets list
    for (Integer aggOperationIndex = 0;
        aggOperationIndex < aggOperations.size();
        aggOperationIndex++) {
      outputColExprs.add(null);

      RexOver agg = aggOperations.get(aggOperationIndex);
      Boolean canFuse = false;
      RexWindow window = agg.getWindow();
      SqlKind fnKind = agg.getAggOperator().getKind();
      String fnName = agg.getAggOperator().getName();
      Boolean respectNulls = !agg.ignoreNulls();
      for (Integer groupbyApplyIdx = 0; groupbyApplyIdx < windows.size(); groupbyApplyIdx++) {

        // Check if the window function can be fused with one of the earlier
        // closures. This is true if it has the same partition & order as the
        // first window in that closure.
        if (windows.get(groupbyApplyIdx).get(0).partitionKeys.equals(window.partitionKeys)
            && windows.get(groupbyApplyIdx).get(0).orderKeys.equals(window.orderKeys)) {
          canFuse = true;
          windows.get(groupbyApplyIdx).add(window);
          fnKinds.get(groupbyApplyIdx).add(fnKind);
          fnNames.get(groupbyApplyIdx).add(fnName);
          aggSets.get(groupbyApplyIdx).add(agg);
          respectNullsLists.get(groupbyApplyIdx).add(respectNulls);
          indices.get(groupbyApplyIdx).add(aggOperationIndex);
          break;
        }
      }

      // If the window function call was not added to one of the existing
      // closure entries, create a new entry for this window function call
      if (!canFuse) {
        windows.add(new ArrayList<>(Arrays.asList(window)));
        fnKinds.add(new ArrayList<>(Arrays.asList(fnKind)));
        fnNames.add(new ArrayList<>(Arrays.asList(fnName)));
        respectNullsLists.add(new ArrayList<>(Arrays.asList(respectNulls)));
        aggSets.add(new ArrayList<>(Arrays.asList(agg)));
        indices.add(new ArrayList<>(Arrays.asList(aggOperationIndex)));
      }
    }

    // For each distinct window/function combination, create a new
    // closure and groupby-apply call
    for (int groupbyApplyIdx = 0; groupbyApplyIdx < windows.size(); groupbyApplyIdx++) {
      List<RexWindow> windowList = windows.get(groupbyApplyIdx);
      List<SqlKind> fnKindList = fnKinds.get(groupbyApplyIdx);
      List<String> fnNameList = fnNames.get(groupbyApplyIdx);
      List<Boolean> respectNullsList = respectNullsLists.get(groupbyApplyIdx);
      List<RexOver> aggSet = aggSets.get(groupbyApplyIdx);
      Pair<String, List<String>> out =
          visitAggOverHelper(
              aggSet,
              colNames,
              windowList,
              fnKindList,
              fnNameList,
              respectNullsList,
              id,
              inputVar.getVariable().getName(),
              translator);

      // Extract the dataframe whose columns contain the output(s) of the
      // window aggregation
      String dfExpr = out.getKey();
      List<String> outputDfColnameList = out.getValue();
      Variable generatedDf = this.genWindowedAggDf();
      Expr rhs = new Expr.Raw(dfExpr);
      this.generatedCode.addToMainFunction(new Op.Assign(generatedDf, rhs));

      // For each aggregation that was fused into this window, extract the
      // corresponding column
      // Reminder that "indices.get(x).get(y) = z" Means, from the dataframe
      // returned by group-by apply function number x, the y'th column
      // is the equivalent expression for aggOperations.get(z)
      List<Integer> innerIndices = indices.get(groupbyApplyIdx);
      for (int j = 0; j < aggSet.size(); j++) {
        String outputDfColName = outputDfColnameList.get(j);
        String outputCode =
            new StringBuilder("bodo.hiframes.pd_series_ext.get_series_data(")
                .append(generatedDf.getName())
                .append("[" + makeQuoted(outputDfColName) + "])")
                .toString();

        // Map the output value back to the correct column location
        Integer index = innerIndices.get(j);
        outputColExprs.set(index, outputCode);
      }
    }

    // Verify that all of the output columns were set
    assert !outputColExprs.contains(null);

    List<Expr> outputRexInfoList = new ArrayList<>();

    for (int outputColExprsIdx = 0;
        outputColExprsIdx < outputColExprs.size();
        outputColExprsIdx++) {
      if (isSingleRow) {
        // In the case that we're inside an apply, we precalculate the column
        // and add the column to colsToAddList. This will result in
        // the precomputed column being added to the dataframe before the apply.
        Variable colVar = genTempColumnVar();
        int colInd = colNames.size() + ctx.getColsToAddList().size();
        ctx.getColsToAddList().add(colVar.getName());
        // TODO: Move to proper IR
        Expr colVarRhs = new Expr.Raw(outputColExprs.get(outputColExprsIdx));
        this.generatedCode.addToMainFunction(new Op.Assign(colVar, colVarRhs));

        // Since we're adding the column to the dataframe before the apply, we return
        // an expression that references the added column.
        ctx.getUsedColumns().add(colInd);
        // NOTE: Codegen for bodosql_case_placeholder() expects column value
        // accesses
        // (e.g. bodo.utils.indexing.scalar_optional_getitem(T1_1, i))
        String returnExpr =
            "bodo.utils.indexing.scalar_optional_getitem("
                + inputVar.emit()
                + "_"
                + colInd
                + ", i)";
        outputRexInfoList.add(new Expr.Raw(returnExpr));
      } else {
        outputRexInfoList.add(new Expr.Raw(outputColExprs.get(outputColExprsIdx)));
      }
    }

    return outputRexInfoList;
  }

  /**
   * Helper function for visitAggOverOp. Return a pandas expression that replicates a SQL windowed
   * aggregation function, given a list of aggregation functions and the window over which they
   * occur. Currently, in all cases but first_value, we do not support fusing multiple window
   * functions into one groupby apply, and the length of aggOperations is 1.
   *
   * @param aggOperations A list
   * @param colNames List of colNames used in the relational expression
   * @param windows the RexWindow over which the aggregation occurs
   * @param aggFns the SQL kinds of the window functions.
   * @param names the names of the window functions.
   * @param id The RelNode id used to uniquely identify the table.
   * @param inputVar Name of dataframe from which InputRefs select Columns
   * @param translator A ctx object containing the Hashset of columns used that need null handling,
   *     the List of precomputed column variables that need to be added to the dataframe before an
   *     apply, and the list of named parameters that need to be passed to an apply function as
   *     arguments.
   * @return A pair of arguments. The first is the string expression of the manipulated dataframe,
   *     the second is a list of output columns where the overall output of the windowed
   *     aggregations are stored.
   */
  private Pair<String, List<String>> visitAggOverHelper(
      List<RexOver> aggOperations,
      List<String> colNames,
      List<RexWindow> windows,
      List<SqlKind> aggFns,
      List<String> names,
      List<Boolean> isRespectNulls,
      int id,
      String inputVar,
      RexToPandasTranslator translator) {

    /*
    BASIC STEPS:
      First, generate a projection, something like

      PROJECTION = {"AGGCOL": (agg_expr0), "SORTCOL_ASC_1": (sort_col_expr),
      "GRPBY_COL": (whatever column we're partitioning by) (+whatever other cols we need to copy over for the partition/grpby)
      "ORIG_COL_POSITION": np.arange(len(df)),
      }


      Then, perform a groupby apply on the projection, using a custom function to correctly sort/window the aggregation function.
      (Ideally, this should use the rolling api in the cases that we can. For now, we need to use a for loop to handle the windowing)
      The custom function is written out to the codegen.

      (Written to generated code)
      def sql_windowed_apply_fn(df, upperbound, lowerbound):
        ...

      (returned)
      PROJECTION.groupby("GRPBY_COL").apply(lambda x: sql_windowed_apply_fn(x, upperbound, lowerbound)["AGG_OUTPUT"]
     */

    // GENERATE THE PROJECTION
    // this has many sub steps, as we need to identify all the columns that need to be present in
    // the dataframe on which
    // we call groupby. This includes the columns by which we are Ordering/sorting by, the columns
    // by which we are
    // partitioning/grouping, and the columns which we are using for computing the output.
    // All of these columns will be added to childExprs, and their expression types will be added to
    // exprTypes
    List<String> childExprNames = new ArrayList<>();
    List<Expr> childExprs = new ArrayList<>();
    List<BodoSQLExprType.ExprType> exprTypes = new ArrayList<>();
    List<String> groupbyCols = new ArrayList<>();

    // simple incremented variable, used for making sure we don't have duplicate column names
    int col_id_var = 0;

    RexWindow window = windows.get(0);

    if (window.partitionKeys.size() == 0) {
      for (String name : names) {
        if (!name.equals("ROW_NUMBER")) {
          throw new BodoSQLCodegenException(
              "BODOSQL currently requires a partition column when handling windowed aggregation"
                  + " functions, except for ROW_NUMBER()");
        }
      }
    }

    for (int i = 1; i < windows.size(); i++) {
      if (!windows.get(i).partitionKeys.equals(window.partitionKeys)) {
        throw new BodoSQLCodegenException(
            "BODOSQL currently requires all window functions in the same closure to have the same"
                + " PARTITION BY");
      }
      if (!windows.get(i).orderKeys.equals(window.orderKeys)) {
        throw new BodoSQLCodegenException(
            "BODOSQL currently requires all window functions in the same closure to have the same"
                + " ORDER BY");
      }
    }

    // Ensure that all the columns needed to correctly do the groupby are added, and record their
    // names for later, so we know what columns by which to group when generating the groupby
    // function text
    for (int i = 0; i < window.partitionKeys.size(); i++) {
      RexNode node = window.partitionKeys.get(i);
      Expr curInfo = node.accept(translator);
      String colName = "GRPBY_COL_" + col_id_var++;
      childExprNames.add(colName);
      childExprs.add(curInfo);
      groupbyCols.add(makeQuoted(colName));
      exprTypes.add(exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(node, id)));
    }

    List<String> orderKeys = new ArrayList<>();
    List<String> orderAscending = new ArrayList<>();
    List<String> orderNAPosition = new ArrayList<>();

    col_id_var = 0;
    // Add all the columns that need to be sorted by into the projection, and record their names so
    // that we can properly
    // generate the func text for sorting the dataframe before calling apply
    // (sorting not needed for all window functions)

    for (int i = 0; i < window.orderKeys.size(); i++) {
      // The RHS of order keys is a set of SqlKind Values. I'm uncertain of what they are used for
      RexNode node = window.orderKeys.get(i).left;
      Expr curRexInfo = node.accept(translator);
      RelFieldCollation.Direction dir = window.orderKeys.get(i).getDirection();
      RelFieldCollation.NullDirection nullDir = window.orderKeys.get(i).getNullDirection();
      if (dir == RelFieldCollation.Direction.ASCENDING) {
        String colName = "ASC_COL_" + col_id_var++;
        childExprNames.add(colName);
        childExprs.add(curRexInfo);
        orderKeys.add(makeQuoted(colName));
      } else {
        assert dir == RelFieldCollation.Direction.DESCENDING;
        String colName = "DEC_COL_" + col_id_var++;
        childExprNames.add(colName);
        childExprs.add(curRexInfo);
        orderKeys.add(makeQuoted(colName));
      }
      orderAscending.add(getAscendingExpr(dir).emit());
      orderNAPosition.add(getNAPositionStringLiteral(nullDir).emit());
      exprTypes.add(exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(node, id)));
    }

    // Add the columns that need to be aggregated into the projection, and record their names.
    // Note that we can have several aggregations that all operate over the same window, IE:
    // SELECT MAX(A), MAX(B) over *window*.

    // For all window functions that we currently support, we have one column upon which we are
    // performing the
    // aggregation, plus one or more scalar arguments. In order to capture this, we create two
    // lists, aggColNames
    // which keeps track of the column we are aggregating, and argsListList that keeps track of the
    // scalar arguments
    // For example: for a call like LEAD(A, 1), LEAD(B, 2) over window,
    // aggColNames would equal [A, B] and argsListList would equal [[1], [2]]

    List<List<WindowedAggregationArgument>> argsListList = new ArrayList();

    // right now, we expect exactly 2, 1 or 0 operands (count *). This will need to be extended in
    // the future
    int cols_used = 0;
    for (int j = 0; j < aggOperations.size(); j++) {
      // Column name for the aggregation column, if needed (argument may be scalar literal)

      List<WindowedAggregationArgument> curArgslist = new ArrayList<>();
      RexOver curAggOperation = aggOperations.get(j);

      if (curAggOperation.getOperands().size() == 0) {
        // count* and NTILE case. In both of these cases, we can just
        // take the length of the dataframe
      } else {
        assert curAggOperation.getOperands().size() != 0;
        for (int i = 0; i < curAggOperation.getOperands().size(); i++) {
          Expr curInfo = curAggOperation.getOperands().get(i).accept(translator);
          if (i == 0 || WindowAggCodeGen.twoArgWindowOptimizedKernels.contains(names.get(j))) {
            // For the majority of aggregation functions, the first argument is the column on which
            // we perform the aggregation. Therefore, we add into the projection, so it will be a
            // part
            // of the table. We also do this for the other columns of certain functions like
            // COVAR_SAMP, which has two column inputs
            if (aggFns.get(j) != SqlKind.NTILE) {
              String curAggColName = "AGG_OP_" + cols_used;
              cols_used += 1;
              childExprNames.add(curAggColName);
              childExprs.add(curInfo);
              BodoSQLExprType.ExprType curExprType =
                  exprTypesMap.get(
                      ExprTypeVisitor.generateRexNodeKey(curAggOperation.getOperands().get(i), id));
              exprTypes.add(curExprType);
              curArgslist.add(WindowedAggregationArgument.fromColumnName(curAggColName));
            } else {
              // In the case of NTILE, it's always a scalar literal, so we just add it to argsList.
              curArgslist.add(WindowedAggregationArgument.fromLiteralExpr(curInfo.emit()));
            }

          } else {
            // For all other currently supported window functions, all arguments after the 0-th
            // are always scalar literals
            assert exprTypesMap.get(
                    ExprTypeVisitor.generateRexNodeKey(curAggOperation.getOperands().get(i), id))
                == BodoSQLExprType.ExprType.SCALAR;
            curArgslist.add(WindowedAggregationArgument.fromLiteralExpr(curInfo.emit()));
          }
        }
      }
      argsListList.add(curArgslist);
    }

    // String for how we represent an offset of 0. This contains any
    // typing info that might be needed.
    String zeroExpr = "np.int64(0)";

    List<Boolean> lowerBoundedList = new ArrayList<Boolean>();
    List<Boolean> upperBoundedList = new ArrayList<Boolean>();
    List<String> lowerBoundsList = new ArrayList<String>();
    List<String> upperBoundsList = new ArrayList<String>();

    for (int i = 0; i < windows.size(); i++) {
      window = windows.get(i);

      // Determine the upper and lower bound of the windowed aggregation.
      // For window function which do not support bounds (NTILE, RANK, LEAD, LAG, etc.) we expect
      // that Calcite will throw an error in the case that they have invalid bounds.

      boolean lowerUnBound = window.getLowerBound().isUnbounded();
      boolean upperUnBound = window.getUpperBound().isUnbounded();
      String lowerBound =
          extractWindowBound(
              window.getLowerBound(),
              "lower",
              WindowAggCodeGen.unboundedLowerBound,
              colNames,
              id,
              inputVar,
              translator,
              zeroExpr);
      String upperBound =
          extractWindowBound(
              window.getUpperBound(),
              "upper",
              WindowAggCodeGen.unboundedUpperBound,
              colNames,
              id,
              inputVar,
              translator,
              zeroExpr);
      lowerBoundedList.add(!lowerUnBound);
      upperBoundedList.add(!upperUnBound);
      lowerBoundsList.add(lowerBound);
      upperBoundsList.add(upperBound);
    }

    // Add a column to the dataframe that tracks the original positions of each of the values.
    // This is used to sort each of the return dataframes, so the output for each row gets mapped
    // back to the correct row
    // generateWindowedAggFn expects this column exists, and has the specified name.
    childExprNames.add(revertSortColumnName);
    childExprs.add(new Expr.Raw("np.arange(len(" + inputVar + "))"));
    exprTypes.add(BodoSQLExprType.ExprType.COLUMN);

    // Create the projection of the input dataframe, which contains only the values which we require
    // in order to
    // perform the window function(s)
    String projection = generateProjectedDataframe(inputVar, childExprNames, childExprs, exprTypes);

    StringBuilder groupedColExpr = new StringBuilder(projection);
    // Projection is done, now we need to produce something like this:
    // projection_df.groupby(whatever cols).apply(my_fn(args))["AGGCOL"]

    // Sort the dataframe by the ORDER BY columns of the window

    // Ascending/Descending/NAPosition have an effect on the output)?

    StringBuilder ascendingString = new StringBuilder();
    StringBuilder NAPositionString = new StringBuilder();
    StringBuilder sortString = new StringBuilder();

    if (!window.orderKeys.isEmpty()) {
      // TODO: much of this is copied from SortCodeGen. Might be decent to refactor to a common
      // function
      ascendingString.append("[");
      sortString.append("[");
      NAPositionString.append("[");
      for (int i = 0; i < orderKeys.size(); i++) {
        sortString.append(orderKeys.get(i) + ", ");
        ascendingString.append(orderAscending.get(i) + ", ");
        NAPositionString.append(orderNAPosition.get(i) + ", ");
      }
      ascendingString.append("]");
      sortString.append("]");
      NAPositionString.append("]");
    }

    // generate the groupby (if needed)
    // currently, we require there to always be a partition clause
    if (!window.partitionKeys.isEmpty()) {
      StringBuilder grpbyExpr = new StringBuilder(".groupby(").append(groupbyCols.toString());
      grpbyExpr.append(", as_index=False, dropna=False, _is_bodosql=True)");
      groupedColExpr.append(grpbyExpr);
    }

    List<RelDataType> typs = new ArrayList<>();
    for (int i = 0; i < aggOperations.size(); i++) {
      typs.add(aggOperations.get(i).getType());
    }

    List<String> outputColList;
    StringBuilder outputExpr = new StringBuilder();
    if (!window.partitionKeys.isEmpty()) {
      // We have a GroupBy object due to performing a partition,

      // Generate the function definition to use within the groupby apply. This
      // returns two items, the generated function text of the function definition,
      // and a map of input aggregation column name to output aggregation column name.
      // For example:
      //
      // LEAD(A, 1), LEAD(B, 2) ==> aggColNames = [A, B]
      //
      // (simplified example func_text, this is the 0th return value in the tuple)
      // def impl(df):
      //     ...
      //     out_df["AGG_OUTPUT_1"] = RESULT_OF_LEAD_AGG_ON_A
      //     out_df["AGG_OUTPUT_2"] = RESULT_OF_LEAD_AGG_ON_B
      //     return out_df
      // (this is the 1st return value in the tuple)
      // outputColMap = {A: AGG_OUTPUT_1, B: AGG_OUTPUT_2}
      //
      Variable fnVar = this.genWindowedAggFnVar();

      Pair<String, List<String>> out =
          generateWindowedAggFn(
              fnVar,
              sortString.toString(),
              ascendingString.toString(),
              NAPositionString.toString(),
              orderKeys,
              aggFns,
              names,
              typs,
              lowerBoundedList,
              upperBoundedList,
              lowerBoundsList,
              upperBoundsList,
              zeroExpr,
              argsListList,
              isRespectNulls);
      String funcText = out.getKey();
      outputColList = out.getValue();

      // The length of the output column list should be the same length as argsListList
      assert argsListList.size() == outputColList.size();
      // TODO: Fix with proper IR.
      this.generatedCode.appendToMainFunction(funcText);

      // perform the groupby apply, using the generated function call
      outputExpr.append(groupedColExpr).append(".apply(").append(fnVar.getName()).append(")");

    } else {
      // Currently the only supported function without a partition is row_number,
      // so the only thing returned will be a single Series containing the
      // output, then each column that needs to access it do so in the form "[:]"
      outputExpr
          .append("bodo.libs.bodosql_array_kernels.row_number(")
          .append(groupedColExpr)
          .append(", ")
          .append(sortString.toString())
          .append(", ")
          .append(ascendingString.toString())
          .append(", ")
          .append(NAPositionString.toString())
          .append(")");
      outputColList = new ArrayList<String>();
      for (int i = 0; i < aggOperations.size(); i++) {
        outputColList.add("ROW_NUMBER");
      }
    }

    return new Pair<>(outputExpr.toString(), outputColList);
  }

  /**
   * Return a string representation of a window function frame bound as will be used by the window
   * function code generation. I.e.: - CURRENT ROW => 0 - 4 PRECEDING => -4 - etc.
   *
   * @param bound The object storing informaiton about the window function bound
   * @param name Either "upper" or "lower"
   * @param defaultString what to learn if there is no ound
   * @param colNames List of colNames used in the relational expression
   * @param id The RelNode id used to uniquely identify the table.
   * @param inputVar Name of dataframe from which InputRefs select Columns
   * @param translator A ctx object containing the Hashset of columns used that need null handling,
   *     the List of precomputed column variables that need to be added to the dataframe before an
   *     apply, and the list of named parameters that need to be passed to an apply function as
   *     arguments.
   * @param zeroExpr String representation of 0
   * @return The string representation of the bound
   */
  private String extractWindowBound(
      RexWindowBound bound,
      String name,
      String defaultString,
      List<String> colNames,
      int id,
      String inputVar,
      RexToPandasTranslator translator,
      String zeroExpr) {

    if (bound.isUnbounded()) {
      return defaultString;
    }

    String result;
    RexNode boundNode = bound.getOffset();
    BodoSQLExprType.ExprType boundExprType;
    if (bound.isPreceding()) {
      // choosing to represent preceding values as negative
      // doesn't require null checking, as value is either a literal or a column
      result = "-(" + boundNode.accept(translator).emit() + ")";
      boundExprType = exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(boundNode, id));
    } else if (bound.isFollowing()) {
      result = boundNode.accept(translator).emit();
      boundExprType = exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(boundNode, id));
    } else if (bound.isCurrentRow()) {
      result = zeroExpr;
      boundExprType = BodoSQLExprType.ExprType.SCALAR;
    } else {
      throw new BodoSQLCodegenException(
          "Error, " + name + " bound of windowed operation not supported:" + bound.toString());
    }

    // We currently require scalar bounds
    assert boundExprType == BodoSQLExprType.ExprType.SCALAR;
    return result;
  }

  /**
   * Visitor for Rex Literals.
   *
   * @param node RexLiteral being visited
   * @param node isSingleRow flag for if table references refer to a single row or the whole table.
   *     This is used for determining if an expr returns a scalar or a column. Only CASE statements
   *     set this to True currently.
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public RexNodeVisitorInfo visitLiteralScan(RexLiteral node, boolean isSingleRow) {
    Expr literal = generateLiteralCode(node, isSingleRow, this);
    return new RexNodeVisitorInfo(literal.emit());
  }

  /**
   * Visitor for Table Scan.
   *
   * @param node TableScan node being visited
   * @param canLoadFromCache Can we load the variable from cache? This is set to False if we have a
   *     filter that wasn't previously cached to enable filter pushdown.
   */
  public void visitTableScan(TableScan node, boolean canLoadFromCache) {

    boolean supportedTableScan =
        node instanceof PandasTableScan
            || node instanceof PandasTargetTableScan
            || node instanceof SnowflakeTableScan;
    if (!supportedTableScan) {
      throw new BodoSQLCodegenException(
          "Internal error: unsupported tableScan node generated:" + node.toString());
    }
    boolean isTargetTableScan = node instanceof PandasTargetTableScan;

    // Streaming snowflake read is handled in a separate function, because of the need to handle
    // codegen in a
    // more precise way due to streaming.
    if (node instanceof SnowflakeTableScan
        && node.getTraitSet().contains(BatchingProperty.STREAMING)) {
      visitStreamingSnowFlakeTableScan((SnowflakeTableScan) node);
      return;
    }

    Variable outVar;
    int nodeId = node.getId();
    if (canLoadFromCache && this.isNodeCached(node)) {
      outVar = this.genDfVar();
      Variable cacheKey = this.varCache.get(nodeId);
      this.generatedCode.add(new Op.Assign(outVar, cacheKey));
    } else {
      BodoSqlTable table;

      // TODO(jsternberg): The proper way to do this is to have the individual nodes
      // handle the code generation. Due to the way the code generation is constructed,
      // we can't really do that so we're just going to hack around it for now to avoid
      // a large refactor
      if (node instanceof SnowflakeTableScan) {
        table = ((SnowflakeTableScan) node).getCatalogTable();
      } else {
        RelOptTableImpl relTable = (RelOptTableImpl) node.getTable();
        table = (BodoSqlTable) relTable.table();
      }

      outVar = visitSingleBatchTableScanCommon(table, isTargetTableScan, nodeId);
    }

    varGenStack.push(outVar);
  }

  /**
   * Helper function that contains the code needed to perform a read of the specified
   *
   * @param table The BodoSqlTable to read
   * @param isTargetTableScan Is the read a TargetTableScan (used in MERGE INTO)
   * @param nodeId node id
   * @return outVar The returned dataframe variable
   */
  public Variable visitSingleBatchTableScanCommon(
      BodoSqlTable table, boolean isTargetTableScan, int nodeId) {
    Expr readCode;
    Op readAssign;

    Variable readVar = this.genDfVar();
    // Add the table to cached values
    if (isTargetTableScan) {
      // TODO: Properly restrict to Iceberg.
      if (!(table instanceof LocalTableImpl) || !table.getDBType().equals("ICEBERG")) {
        throw new BodoSQLCodegenException(
            "Insert Into is only supported with Iceberg table destinations provided via"
                + " the the SQL TablePath API");
      }
      readCode = table.generateReadCode("_bodo_merge_into=True,");
      this.fileListAndSnapshotIdArgs =
          String.format(
              "snapshot_id=%s, old_fnames=%s,", icebergSnapshotIDName, icebergFileListVarName);

      // TODO: replace this with tuple assign instead of raw code
      readAssign =
          new Op.Code(
              String.format(
                  "%s, %s, %s = %s\n",
                  readVar.emit(), icebergFileListVarName, icebergSnapshotIDName, readCode.emit()));
      targetTableDf = readVar;
    } else {
      readCode = table.generateReadCode(false);
      readAssign = new Op.Assign(readVar, readCode);
    }

    boolean readUsesIO = table.readRequiresIO();
    if (this.verboseLevel >= 1 && readUsesIO) {
      // If the user has set verbose level >= 1 and there is IO, we generate
      // timers and print the read time.
      Variable timerVar = this.genGenericTempVar();
      String tableName = table.getName();
      this.generatedCode.add(new Op.Assign(timerVar, new Expr.Raw("time.time()")));
      this.generatedCode.add(readAssign);
      // Generate the Debug message.
      List<Expr> logMsgArgs =
          List.of(
              new Expr.StringLiteral("IO TIMING"),
              new Expr.Raw(
                  String.format(
                      "f'Finished reading table %s in {time.time() - %s} seconds'",
                      tableName, timerVar.emit())));
      Expr.Call logMsgCall = new Expr.Call("bodo.user_logging.log_message", logMsgArgs, List.of());
      this.generatedCode.add(new Op.Stmt(logMsgCall));
    } else {
      this.generatedCode.add(readAssign);
    }

    Variable outVar;

    Expr castExpr = table.generateReadCastCode(readVar);
    if (!castExpr.equals(readVar)) {
      // Generate a new outVar to avoid shadowing
      outVar = this.genDfVar();
      this.generatedCode.add(new Op.Assign(outVar, castExpr));
    } else {
      outVar = readVar;
    }

    if (!isTargetTableScan) {
      // Add the table to cached values. We only support this for regular
      // tables and not the target table in merge into.
      this.varCache.put(nodeId, outVar);
    }

    return outVar;
  }

  public void visitStreamingSnowFlakeTableScan(SnowflakeTableScan node) {
    Variable outVar = this.genDfVar();

    BodoSqlTable table = node.getCatalogTable();
    Expr readCode = table.generateReadCode(true);

    List<String> columnNames = node.getRowType().getFieldNames();

    varGenStack.push(initStreamingIoLoop(readCode, columnNames));
  }

  /**
   * Helper function that handles the setup for an IO loop that draws from a streaming iterator.
   *
   * <p>Handles: Creating the streaming Pipeline Calling read_arrow_next on the iterator object
   * Converting the output table from read_arrow_next into a dataframe
   *
   * @param readCode The expression that evaluates to the streaming iterator object.
   * @param columnNames The expected output column names.
   * @return Dataframe variable for a fully initialized dataframe, containing a batch of data.
   */
  public Variable initStreamingIoLoop(Expr readCode, List<String> columnNames) {

    Variable flagVar = genFinishedStreamingFlag();
    // start the streaming pipeline
    this.generatedCode.startStreamingPipelineFrame(flagVar);

    StreamingPipelineFrame currentPipeline = this.generatedCode.getCurrentStreamingPipeline();
    // Create the reader before the loop
    Variable readerVar = this.genReaderVar();
    currentPipeline.addInitialization(new Op.Assign(readerVar, readCode));

    Variable dfChunkVar = this.genTableVar();

    Call read_arrow_next_call =
        new Call("bodo.io.arrow_reader.read_arrow_next", List.of(readerVar), List.of());
    this.generatedCode.add(new Op.TupleAssign(List.of(dfChunkVar, flagVar), read_arrow_next_call));

    Expr df_len = new Call("len", dfChunkVar);

    Variable idx_var = genGenericTempVar();

    List<Expr> argsList =
        List.of(new Expr.IntegerLiteral(0), df_len, new Expr.IntegerLiteral(1), Expr.None.INSTANCE);

    Call init_range_index_call =
        new Call("bodo.hiframes.pd_index_ext.init_range_index", argsList, List.of());

    this.generatedCode.add(new Op.Assign(idx_var, init_range_index_call));

    Expr tableTuple = new Expr.Tuple(List.of(dfChunkVar));

    List<Expr> columnNamesExpr = new ArrayList<>();
    for (int i = 0; i < columnNames.size(); i++) {
      columnNamesExpr.add(new Expr.StringLiteral(columnNames.get(i)));
    }
    Expr.Tuple TupleExpr = new Expr.Tuple(columnNamesExpr);

    Expr colNamesMetaExpr = this.lowerAsColNamesMetaType(TupleExpr);

    Call init_dataframe_call =
        new Call(
            "bodo.hiframes.pd_dataframe_ext.init_dataframe",
            List.of(tableTuple, idx_var, colNamesMetaExpr),
            List.of());

    Variable outVar = this.genDfVar();
    this.generatedCode.add(new Op.Assign(outVar, init_dataframe_call));

    return outVar;
  }

  /**
   * Visitor for Join: Supports JOIN clause in SQL.
   *
   * @param node join node being visited
   */
  public void visitPandasJoin(PandasJoin node) {
    if (node.getTraitSet().contains(BatchingProperty.STREAMING)) {
      visitStreamingPandasJoin(node);
    } else {
      visitBatchedPandasJoin(node);
    }
  }

  /**
   * Visitor for Join without streaming.
   *
   * @param node join node being visited
   */
  public void visitBatchedPandasJoin(PandasJoin node) {
    /* get left/right tables */
    Variable outVar = this.genDfVar();
    int nodeId = node.getId();
    List<String> outputColNames = node.getRowType().getFieldNames();
    if (this.isNodeCached(node)) {
      Variable cacheKey = this.varCache.get(nodeId);
      this.generatedCode.add(new Op.Assign(outVar, cacheKey));
    } else {
      this.visit(node.getLeft(), 0, node);
      List<String> leftColNames = node.getLeft().getRowType().getFieldNames();
      Variable leftTable = varGenStack.pop();
      this.visit(node.getRight(), 1, node);
      List<String> rightColNames = node.getRight().getRowType().getFieldNames();
      Variable rightTable = varGenStack.pop();
      this.genRelnodeTimerStart(node);

      RexNode cond = node.getCondition();

      /** Generate the expression for the join condition in a format Bodo supports. */
      HashSet<String> mergeCols = new HashSet<>();
      Expr joinCond = visitJoinCond(cond, leftColNames, rightColNames, mergeCols);

      /* extract join type */
      String joinType = node.getJoinType().lowerName;

      // a join without any conditions is a cross join (how="cross" in pd.merge)
      if (joinCond.emit().equals("True")) {
        joinType = "cross";
      }

      boolean tryRebalanceOutput = node.getRebalanceOutput();

      Op.Assign joinCode =
          generateJoinCode(
              outVar,
              joinType,
              rightTable,
              leftTable,
              rightColNames,
              leftColNames,
              outputColNames,
              joinCond,
              mergeCols,
              tryRebalanceOutput);
      this.generatedCode.add(joinCode);
      this.varCache.put(nodeId, outVar);
      this.genRelnodeTimerStop(node);
    }
    varGenStack.push(outVar);
  }

  /**
   * Visitor for PandasJoin when it is a streaming join. Both nested loop and hash join use the same
   * API calls and the decision is made based on the values of the arguments.
   *
   * <p>Note: Streaming doesn't support caching yet.
   */
  void visitStreamingPandasJoin(PandasJoin node) {
    // Extract the Hash Join information
    JoinInfo joinInfo = node.analyzeCondition();
    Pair<Variable, Variable> keyIndices = getStreamingJoinKeyIndices(joinInfo, this);
    List<String> buildNodeNames = node.getLeft().getRowType().getFieldNames();
    List<String> probeNodeNames = node.getRight().getRowType().getFieldNames();
    // Fetch the names for each child.
    List<Expr.StringLiteral> buildColNames = stringsToStringLiterals(buildNodeNames);
    List<Expr.StringLiteral> probeColNames = stringsToStringLiterals(probeNodeNames);
    Variable buildNamesGlobal = lowerAsColNamesMetaType(new Expr.Tuple(buildColNames));
    Variable probeNamesGlobal = lowerAsColNamesMetaType(new Expr.Tuple(probeColNames));
    // Get the non equi-join info
    Expr nonEquiCond =
        visitNonEquiConditions(joinInfo.nonEquiConditions, buildNodeNames, probeNodeNames);
    // Right now we must process nonEquiCond as a string.
    String condString = nonEquiCond.emit();
    final List<kotlin.Pair<String, Expr>> namedArgs;
    if (condString.equals("")) {
      // There is no join condition
      namedArgs = List.of();
    } else {
      namedArgs =
          List.of(new kotlin.Pair<>("non_equijoin_cond", new Expr.StringLiteral(condString)));
    }

    // Visit the batch side
    this.visit(node.getLeft(), 0, node);
    Variable buildDf = varGenStack.pop();
    // Fetch the batch state
    StreamingPipelineFrame batchPipeline = generatedCode.getCurrentStreamingPipeline();
    // Create the state var.
    Variable joinStateVar = genGenericTempVar();
    Expr.BooleanLiteral isLeftOuter =
        new Expr.BooleanLiteral(node.getJoinType().generatesNullsOnRight());
    Expr.BooleanLiteral isRightOuter =
        new Expr.BooleanLiteral(node.getJoinType().generatesNullsOnLeft());
    Expr.Call stateCall =
        new Expr.Call(
            "bodo.libs.stream_join.init_join_state",
            List.of(
                keyIndices.left,
                keyIndices.right,
                buildNamesGlobal,
                probeNamesGlobal,
                isLeftOuter,
                isRightOuter),
            namedArgs);
    Op.Assign joinInit = new Op.Assign(joinStateVar, stateCall);
    batchPipeline.addInitialization(joinInit);
    // Join needs the isLast to be global before calling the build side change.
    batchPipeline.ensureExitCondSynchronized();
    Variable batchExitCond = batchPipeline.getExitCond();
    // Fetch the underlying table for the join.
    Variable buildTable = genTableVar();
    Expr.Call buildDfData =
        new Expr.Call("bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data", List.of(buildDf));
    int numBuildCols = node.getLeft().getRowType().getFieldCount();
    List<Expr.IntegerLiteral> buildIndices = integerLiteralArange(numBuildCols);
    Variable buildColNums = lowerAsMetaType(new Expr.Tuple(buildIndices));
    Expr.Call buildTableCall =
        new Expr.Call(
            "bodo.hiframes.table.logical_table_to_table",
            buildDfData,
            new Expr.Tuple(List.of()),
            buildColNums,
            new Expr.IntegerLiteral(numBuildCols));
    generatedCode.add(new Assign(buildTable, buildTableCall));
    Expr.Call batchCall =
        new Expr.Call(
            "bodo.libs.stream_join.join_build_consume_batch",
            List.of(joinStateVar, buildTable, batchExitCond));
    generatedCode.add(new Op.Stmt(batchCall));
    // Finalize and add the batch pipeline.
    generatedCode.add(new Op.StreamingPipeline(generatedCode.endCurrentStreamingPipeline()));
    // Visit the probe side
    this.visit(node.getRight(), 1, node);
    Variable probeDf = varGenStack.pop();
    StreamingPipelineFrame probePipeline = generatedCode.getCurrentStreamingPipeline();
    Variable oldFlag = probePipeline.getExitCond();
    // Change the probe condition
    Variable newFlag = genGenericTempVar();
    probePipeline.endSection(newFlag);
    // Fetch the underlying table for the join.
    Variable probeTable = genTableVar();
    Expr.Call probeDfData =
        new Expr.Call("bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data", List.of(probeDf));
    int numProbeCols = node.getRight().getRowType().getFieldCount();
    List<Expr.IntegerLiteral> probeIndices = integerLiteralArange(numProbeCols);
    Variable leftColNums = lowerAsMetaType(new Expr.Tuple(probeIndices));
    Expr.Call probeTableCall =
        new Expr.Call(
            "bodo.hiframes.table.logical_table_to_table",
            probeDfData,
            new Expr.Tuple(List.of()),
            leftColNums,
            new Expr.IntegerLiteral(numProbeCols));
    generatedCode.add(new Assign(probeTable, probeTableCall));
    // Add the probe side
    Variable outTable = genTableVar();
    Expr.Call probeCall =
        new Expr.Call(
            "bodo.libs.stream_join.join_probe_consume_batch",
            List.of(joinStateVar, probeTable, oldFlag));
    generatedCode.add(new Op.TupleAssign(List.of(outTable, newFlag), probeCall));
    // Generate an index.
    Variable indexVar = genIndexVar();
    Expr.Call lenCall = new Expr.Call("len", List.of(outTable));
    Expr.IntegerLiteral zero = new IntegerLiteral(0);
    Expr.IntegerLiteral one = new IntegerLiteral(1);
    Expr.Call indexCall =
        new Expr.Call(
            "bodo.hiframes.pd_index_ext.init_range_index",
            List.of(zero, lenCall, one, Expr.None.INSTANCE));
    generatedCode.add(new Op.Assign(indexVar, indexCall));
    // Generate a DataFrame
    Variable outDf = genDfVar();
    Expr.Tuple tableTuple = new Expr.Tuple(List.of(outTable));
    // Generate the column names global
    List<Expr.StringLiteral> colNamesLiteral =
        stringsToStringLiterals(node.getRowType().getFieldNames());
    Expr.Tuple colNamesTuple = new Expr.Tuple(colNamesLiteral);
    Variable colNamesMeta = lowerAsColNamesMetaType(colNamesTuple);
    Expr.Call initDfCall =
        new Expr.Call(
            "bodo.hiframes.pd_dataframe_ext.init_dataframe",
            List.of(tableTuple, indexVar, colNamesMeta));
    generatedCode.add(new Op.Assign(outDf, initDfCall));
    // Append the code to delete the state
    Op.Stmt deleteState =
        new Op.Stmt(
            new Expr.Call("bodo.libs.stream_join.delete_join_state", List.of(joinStateVar)));
    probePipeline.addTermination(deleteState);
    // Add the DF to the stack
    this.varGenStack.push(outDf);
  }

  /**
   * Visitor for RowSample: Supports SAMPLE clause in SQL with a fixed number of rows.
   *
   * @param node rowSample node being visited
   */
  public void visitRowSample(PandasRowSample node) {
    // We always assume row sample has exactly one input
    assert node.getInputs().size() == 1;

    // Visit the input
    RelNode inp = node.getInput(0);
    this.visit(inp, 0, node);
    Variable inpExpr = varGenStack.pop();

    Variable outVar = this.genDfVar();
    this.genRelnodeTimerStart(node);

    this.generatedCode.add(new Op.Assign(outVar, generateRowSampleCode(inpExpr, node.getParams())));

    varGenStack.push(outVar);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor for Sample: Supports SAMPLE clause in SQL with a fraction of the input.
   *
   * @param node sample node being visited
   */
  public void visitSample(PandasSample node) {
    // We always assume sample has exactly one input
    assert node.getInputs().size() == 1;

    // Visit the input
    RelNode inp = node.getInput(0);
    this.visit(inp, 0, node);
    Variable inpExpr = varGenStack.pop();

    Variable outVar = this.genDfVar();
    this.genRelnodeTimerStart(node);

    this.generatedCode.add(new Op.Assign(outVar, generateSampleCode(inpExpr, node.getParams())));
    varGenStack.push(outVar);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Determine if the given node is cached. We check if a node is cached by checking if its id is
   * stored in the cache. In case a particular node does not support caching (or only supports
   * caching in certain cases), we check if all children are cached as well.
   *
   * @param node the node that may be cached
   * @return If the node and all children are cached
   */
  private boolean isNodeCached(RelNode node) {
    // Perform BFS to search all children
    Stack<RelNode> nodeStack = new Stack<>();
    nodeStack.add(node);
    while (!nodeStack.isEmpty()) {
      RelNode n = nodeStack.pop();

      // We skip CombineStreamsExchange and SeparateStreamExchange for now because they
      // don't generate code yet.
      while (n instanceof CombineStreamsExchange || n instanceof SeparateStreamExchange) {
        // TODO: Actually implement
        // nothing change to allow me to open the pr
        n = n.getInput(0);
      }

      // SnowflakeToPandasConverter is weird and this should be removed
      // once we refactor the code generation logic for direct table reads.
      // Since this node typically represents an SQL call, we never actually
      // visit the children and so the children couldn't possibly be cached.
      // On the other hand, we have a special code generation for direct table
      // reads that we would need to check.
      // If we see the input to SnowflakeToPandasConverter is a TableScan,
      // skip this node and go directly to the TableScan. Otherwise, check it as
      // normal.
      if (n instanceof SnowflakeToPandasConverter && n.getInput(0) instanceof TableScan) {
        n = n.getInput(0);
      }
      if (!this.varCache.containsKey(n.getId())) {
        return false;
      }
      if (n instanceof SnowflakeToPandasConverter) {
        continue;
      }
      nodeStack.addAll(n.getInputs());
      for (RelNode child : n.getInputs()) {
        nodeStack.add(child);
      }
    }

    return true;
  }

  /**
   * Adds code to the generated python that logs a simple message that tells the user the node being
   * executed, and starts a timer. Only works if verboseLevel is set to 1 or greater.
   *
   * <p>This is currently very rough, and should only be used for internal debugging
   *
   * @param node The node being timed
   */
  private void genRelnodeTimerStart(RelNode node) {
    if (this.verboseLevel >= this.RelNodeTimingVerboseLevel) {
      int node_id = node.getId();
      Variable var = new Variable("t_" + node_id);
      Expr.Call timeCall = new Call("time.time");
      this.generatedCode.add(new Assign(var, timeCall));
    }
  }

  private static Variable NODE_STRING = new Variable("node_string");

  /**
   * Adds code to the generated python that logs a simple message that tells the user the node being
   * executed has finished, and prints the execution time. Only works if verboseLevel is set to 1 or
   * greater.
   *
   * <p>This is currently very rough, and should only be used for internal debugging
   *
   * @param node The node being timed
   */
  private void genRelnodeTimerStop(RelNode node) {
    if (this.verboseLevel >= this.RelNodeTimingVerboseLevel) {
      int node_id = node.getId();
      // Some logic to get the first line of the full relTree (the string of the node that we're
      // interested in)
      String nodeStr = Arrays.stream(RelOptUtil.toString(node).split("\n")).findFirst().get();

      Op nodeString = new Op.Assign(NODE_STRING, new Expr.TripleQuotedString(nodeStr));
      this.generatedCode.add(nodeString);
      Op logMessageCall =
          new Op.Stmt(
              new Expr.Call(
                  "bodo.user_logging.log_message",
                  new Expr.StringLiteral("RELNODE_TIMING"),
                  new Expr.Raw(
                      "f'''Execution time for RelNode {node_string}: {time.time() - t_"
                          + node_id
                          + "}'''")));
      this.generatedCode.add(logMessageCall);
    }
  }

  private RexToPandasTranslator getRexTranslator(RelNode self, String inVar, RelNode input) {
    return getRexTranslator(self, new Dataframe(inVar, input));
  }

  private RexToPandasTranslator getRexTranslator(RelNode self, Dataframe input) {
    return getRexTranslator(self.getId(), input);
  }

  private RexToPandasTranslator getRexTranslator(int nodeId, Dataframe input) {
    return new RexToPandasTranslator(this, this.generatedCode, this.typeSystem, nodeId, input);
  }
}
