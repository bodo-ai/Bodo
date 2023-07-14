package com.bodosql.calcite.application;

import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.concatDataFrames;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.generateAggCodeNoAgg;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.generateAggCodeNoGroupBy;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.generateAggCodeWithGroupBy;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.generateApplyCodeWithGroupBy;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.getStreamingGroupbyFtypes;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.getStreamingGroupbyKeyIndices;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.getStreamingGroupbyOffsetAndCols;
import static com.bodosql.calcite.application.BodoSQLCodeGen.JoinCodeGen.generateJoinCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.LiteralCodeGen.generateLiteralCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.LogicalValuesCodeGen.generateLogicalValuesCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SampleCodeGen.generateRowSampleCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SampleCodeGen.generateSampleCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SetOpCodeGen.generateExceptCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SetOpCodeGen.generateIntersectCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SetOpCodeGen.generateUnionCode;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SortCodeGen.generateSortCode;
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
import com.bodosql.calcite.adapter.pandas.PandasIntersect;
import com.bodosql.calcite.adapter.pandas.PandasJoin;
import com.bodosql.calcite.adapter.pandas.PandasMinus;
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
import com.bodosql.calcite.adapter.pandas.StreamingOptions;
import com.bodosql.calcite.adapter.snowflake.SnowflakeToPandasConverter;
import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer;
import com.bodosql.calcite.application.timers.StreamingRelNodeTimer;
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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Stack;
import java.util.function.Supplier;
import kotlin.jvm.functions.Function1;
import org.apache.calcite.prepare.RelOptTableImpl;
import org.apache.calcite.rel.RelFieldCollation;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelVisitor;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Correlate;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.JoinInfo;
import org.apache.calcite.rel.core.TableModify;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Pair;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.jetbrains.annotations.NotNull;

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

  private final StreamingOptions streamingOptions;

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

  public PandasCodeGenVisitor(
      HashMap<String, BodoSQLExprType.ExprType> exprTypesMap,
      HashMap<String, RexNode> searchMap,
      HashMap<String, String> loweredGlobalVariablesMap,
      String originalSQLQuery,
      RelDataTypeSystem typeSystem,
      boolean debuggingDeltaTable,
      int verboseLevel,
      int batchSize) {
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
    this.streamingOptions = new StreamingOptions(batchSize);
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
   * Generate a new iter variable for step by step pandas codegen
   *
   * @return variable
   */
  public Variable genIterVar() {
    return generatedCode.getSymbolTable().genIterVar();
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
   * Generate the new temporary variable for step by step pandas codegen.
   *
   * @return variable
   */
  public Variable genWriterVar() {
    return generatedCode.getSymbolTable().genWriterVar();
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
    if (node instanceof TableScan) {
      this.visitTableScan((TableScan) node, !(parent instanceof Filter));
    } else if (node instanceof PandasJoin) {
      this.visitPandasJoin((PandasJoin) node);
    } else if (node instanceof PandasSort) {
      this.visitPandasSort((PandasSort) node);
    } else if (node instanceof PandasAggregate) {
      this.visitLogicalAggregate((PandasAggregate) node);
    } else if (node instanceof PandasUnion) {
      this.visitLogicalUnion((PandasUnion) node);
    } else if (node instanceof PandasIntersect) {
      this.visitLogicalIntersect((PandasIntersect) node);
    } else if (node instanceof PandasMinus) {
      this.visitLogicalMinus((PandasMinus) node);
    } else if (node instanceof PandasValues) {
      this.visitLogicalValues((PandasValues) node);
    } else if (node instanceof PandasTableModify) {
      this.visitLogicalTableModify((PandasTableModify) node);
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

    // Fetch the underlying table
    Variable inputTable = genTableVar();
    Expr.Call dfData =
        new Expr.Call("bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data", List.of(inputDFVar));
    int numInputCols = node.getRowType().getFieldCount();
    List<Expr.IntegerLiteral> inputIndices = integerLiteralArange(numInputCols);
    Variable colNums = lowerAsMetaType(new Expr.Tuple(inputIndices));
    Expr.Call inputTableCall =
        new Expr.Call(
            "bodo.hiframes.table.logical_table_to_table",
            dfData,
            new Expr.Tuple(List.of()),
            colNums,
            new Expr.IntegerLiteral(numInputCols));
    generatedCode.add(new Assign(inputTable, inputTableCall));

    // Append to the list at the end of the loop.
    List<Expr> args = new ArrayList<>();
    args.add(inputTable);
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
    Variable accumulatedTable = genTableVar();
    Expr concatenatedTable =
        new Expr.Call("bodo.utils.table_utils.concat_tables", List.of(batchAccumulatorVariable));
    generatedCode.add(new Op.Assign(accumulatedTable, concatenatedTable));

    Variable accumulatedDfVar = this.genDfVar();
    // Generate an index.
    Variable indexVar = genIndexVar();
    Expr.Call lenCall = new Expr.Call("len", List.of(accumulatedTable));
    Expr.IntegerLiteral zero = new IntegerLiteral(0);
    Expr.IntegerLiteral one = new IntegerLiteral(1);
    Expr.Call indexCall =
        new Expr.Call(
            "bodo.hiframes.pd_index_ext.init_range_index",
            List.of(zero, lenCall, one, Expr.None.INSTANCE));
    generatedCode.add(new Op.Assign(indexVar, indexCall));
    // Generate a DataFrame
    Expr.Tuple tableTuple = new Expr.Tuple(List.of(accumulatedTable));
    // Generate the column names global
    List<Expr.StringLiteral> colNamesLiteral =
        stringsToStringLiterals(node.getRowType().getFieldNames());
    Expr.Tuple colNamesTuple = new Expr.Tuple(colNamesLiteral);
    Variable colNamesMeta = lowerAsColNamesMetaType(colNamesTuple);
    Expr.Call initDfCall =
        new Expr.Call(
            "bodo.hiframes.pd_dataframe_ext.init_dataframe",
            List.of(tableTuple, indexVar, colNamesMeta));
    generatedCode.add(new Op.Assign(accumulatedDfVar, initDfCall));
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
    Variable iterVar = genIterVar();
    generatedCode.startStreamingPipelineFrame(exitCond, iterVar);
    StreamingPipelineFrame streamingInfo = generatedCode.getCurrentStreamingPipeline();
    // Initialize the iteration.
    Variable iteratorNumber = genGenericTempVar();
    Op.Assign assn = new Op.Assign(iteratorNumber, new Expr.IntegerLiteral(0));
    streamingInfo.addInitialization(assn);

    Variable outputDfVar = genDfVar();
    Expr sliceStart =
        new Expr.Binary(
            "*", iteratorNumber, new Expr.IntegerLiteral(streamingOptions.getChunkSize()));
    Expr sliceEnd =
        new Expr.Binary(
            "*",
            new Expr.Binary("+", iteratorNumber, new Expr.IntegerLiteral(1)),
            new Expr.IntegerLiteral(streamingOptions.getChunkSize()));

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
    if (node.canUseNodeCache() && isNodeCached(node)) {
      varGenStack.push(varCache.get(node.getId()));
      return;
    }

    // Note: All timer handling is done in emit
    Dataframe out = node.emit(new Implementor(node));

    // Place the output variable in the varCache and varGenStack.
    varCache.put(node.getId(), out);
    varGenStack.push(out);
  }

  /**
   * Visitor method for logicalValue Nodes
   *
   * @param node RelNode to be visited
   */
  private void visitLogicalValues(PandasValues node) {
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
    singleBatchTimer(
        node,
        () -> {
          Expr logicalValuesExpr = generateLogicalValuesCode(exprCodes, node.getRowType(), this);
          this.generatedCode.add(new Op.Assign(outVar, logicalValuesExpr));
          this.varGenStack.push(outVar);
        });
  }

  /**
   * Visitor for Logical Union node. Code generation for UNION [ALL/DISTINCT] in SQL
   *
   * @param node LogicalUnion node to be visited
   */
  private void visitLogicalUnion(PandasUnion node) {
    List<Variable> childExprs = new ArrayList<>();
    List<List<String>> childExprsColumns = new ArrayList<>();
    // Visit all of the inputs
    for (int i = 0; i < node.getInputs().size(); i++) {
      RelNode input = node.getInput(i);
      this.visit(input, i, node);
      childExprs.add(varGenStack.pop());
      childExprsColumns.add(input.getRowType().getFieldNames());
    }
    singleBatchTimer(
        node,
        () -> {
          Variable outVar = this.genDfVar();
          List<String> columnNames = node.getRowType().getFieldNames();
          Expr unionExpr = generateUnionCode(columnNames, childExprs, node.all, this);
          this.generatedCode.add(new Op.Assign(outVar, unionExpr));
          varGenStack.push(outVar);
        });
  }

  /**
   * Visitor for Logical Intersect node. Code generation for INTERSECT [ALL/DISTINCT] in SQL
   *
   * @param node LogicalIntersect node to be visited
   */
  private void visitLogicalIntersect(PandasIntersect node) {
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
    singleBatchTimer(
        node,
        () -> {
          this.generatedCode.addAll(
              generateIntersectCode(
                  outVar, lhsExpr, lhsColNames, rhsExpr, rhsColNames, colNames, node.all, this));
          varGenStack.push(outVar);
        });
  }

  /**
   * Visitor for Logical Minus node. Code generation for EXCEPT/MINUS in SQL
   *
   * @param node LogicalMinus node to be visited
   */
  private void visitLogicalMinus(PandasMinus node) {
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
    singleBatchTimer(
        node,
        () -> {
          this.generatedCode.addAll(
              generateExceptCode(
                  outVar, lhsVar, lhsColNames, rhsVar, rhsColNames, colNames, node.all, this));
          varGenStack.push(outVar);
        });
  }

  /**
   * Visitor for Logical Sort node. Code generation for ORDER BY clauses in SQL
   *
   * @param node PandasSort node to be visited
   */
  public void visitPandasSort(PandasSort node) {
    RelNode input = node.getInput();
    this.visit(input, 0, node);
    singleBatchTimer(
        node,
        () -> {
          Variable inVar = varGenStack.pop();
          List<String> colNames = input.getRowType().getFieldNames();
          /* handle case for queries with "ORDER BY" clause */
          List<RelFieldCollation> sortOrders = node.getCollation().getFieldCollations();
          Variable outVar = this.genDfVar();
          String limitStr = "";
          String offsetStr = "";
          /* handle case for queries with "LIMIT" clause */
          RexNode fetch =
              node.fetch; // for a select query including a clause LIMIT N, fetch returns N.
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
        });
  }

  /**
   * Generate Code for Non-Streaming / Single Batch CREATE TABLE
   *
   * @param node Create Table Node that Code is Generated For
   * @param outputSchemaAsCatalog Catalog of Output Table
   * @param ifExists Action if Table Already Exists
   * @param createTableType Type of Table to Create
   */
  public void genSingleBatchTableCreate(
      PandasTableCreate node,
      CatalogSchemaImpl outputSchemaAsCatalog,
      BodoSQLCatalog.ifExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType) {
    singleBatchTimer(
        node,
        () -> {
          this.generatedCode.add(
              new Op.Stmt(
                  outputSchemaAsCatalog.generateWriteCode(
                      this.varGenStack.pop(), node.getTableName(), ifExists, createTableType)));
        });
  }

  /**
   * Generate Code for Streaming CREATE TABLE
   *
   * @param node Create Table Node that Code is Generated For
   * @param outputSchemaAsCatalog Catalog of Output Table
   * @param ifExists Action if Table Already Exists
   * @param createTableType Type of Table to Create
   */
  public void genStreamingTableCreate(
      PandasTableCreate node,
      CatalogSchemaImpl outputSchemaAsCatalog,
      BodoSQLCatalog.ifExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType) {
    // Generate Streaming Code in this case
    // Get or create current streaming pipeline
    StreamingPipelineFrame currentPipeline = this.generatedCode.getCurrentStreamingPipeline();
    // TODO: Move to a wrapper function to avoid the timerInfo calls.
    // This requires more information about the high level design of the streaming
    // operators since there are several parts (e.g. state, multiple loop sections, etc.)
    // At this time it seems like it would be too much work to have a clean interface.
    // There may be a need to pass in several lambdas, so other changes may be needed to avoid
    // constant rewriting.
    StreamingRelNodeTimer timerInfo =
        StreamingRelNodeTimer.createStreamingTimer(
            this.generatedCode,
            this.verboseLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());
    timerInfo.initializeTimer();

    // First, create the writer state before the loop
    timerInfo.insertStateStartTimer();
    Expr writeInitCode =
        outputSchemaAsCatalog.generateStreamingWriteInitCode(
            node.getTableName(), ifExists, createTableType);
    Variable writerVar = this.genWriterVar();
    currentPipeline.addInitialization((new Op.Assign(writerVar, writeInitCode)));
    timerInfo.insertStateEndTimer();

    // Second, append the Dataframe to the writer
    timerInfo.insertLoopOperationStartTimer();
    Expr writerAppendCall =
        outputSchemaAsCatalog.generateStreamingWriteAppendCode(
            writerVar, this.varGenStack.pop(), currentPipeline.getExitCond());
    this.generatedCode.add(new Op.Stmt(writerAppendCall));
    timerInfo.insertLoopOperationEndTimer();

    // Lastly, end the loop
    timerInfo.terminateTimer();
    StreamingPipelineFrame finishedPipeline = this.generatedCode.endCurrentStreamingPipeline();
    this.generatedCode.add(new Op.StreamingPipeline(finishedPipeline));
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

    // No streaming or single batch case
    if (node.getInput().getTraitSet().containsIfApplicable(BatchingProperty.SINGLE_BATCH)) {
      genSingleBatchTableCreate(node, outputSchemaAsCatalog, ifExists, createTableType);
    } else {
      genStreamingTableCreate(node, outputSchemaAsCatalog, ifExists, createTableType);
    }
  }

  /**
   * Visitor for LogicalTableModify, which is used to support certain SQL write operations.
   *
   * @param node LogicalTableModify node to be visited
   */
  public void visitLogicalTableModify(PandasTableModify node) {
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
  public void visitMergeInto(PandasTableModify node) {
    assert node.getOperation() == TableModify.Operation.MERGE;

    RelNode input = node.getInput();
    this.visit(input, 0, node);
    singleBatchTimer(
        node,
        () -> {
          Variable deltaDfVar = this.varGenStack.pop();
          List<String> currentDeltaDfColNames = input.getRowType().getFieldNames();

          if (this.debuggingDeltaTable) {
            // If this environment variable is set, we're only testing the generation of the delta
            // table.
            // Just return the delta table.
            // We drop no-ops from the delta table, as a few Calcite Optimizations can result in
            // their
            // being removed from the table, and their presence/lack thereof shouldn't impact
            // anything in
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
                                makeQuoted(
                                    currentDeltaDfColNames.get(currentDeltaDfColNames.size() - 1)))
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
                  "MERGE INTO is only supported with Iceberg table destinations provided via the"
                      + " the SQL TablePath API");
            }

            // note table.getColumnNames does NOT include ROW_ID or MERGE_ACTION_ENUM_COL_NAME
            // column
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
                handleCastAndRenameBeforeWrite(
                    writebackDf, targetTableFinalColumnNames, bodoSqlTable);
            Variable castedAndRenamedWriteBackDfVar = this.genDfVar();
            this.generatedCode.add(
                new Op.Assign(castedAndRenamedWriteBackDfVar, castedAndRenamedWriteBackDfExpr));
            this.generatedCode.add(
                new Op.Stmt(
                    bodoSqlTable.generateWriteCode(
                        castedAndRenamedWriteBackDfVar, this.fileListAndSnapshotIdArgs)));
          }
        });
  }

  /**
   * Generate Code for Single Batch, Non-Streaming INSERT INTO
   *
   * @param node LogicalTableModify node to be visited
   * @param inVar Input Var containing table to write
   * @param colNames List of column names
   * @param bodoSqlTable Reference to Table to Write to
   */
  void genSingleBatchInsertInto(
      PandasTableModify node, Variable inVar, List<String> colNames, BodoSqlTable bodoSqlTable) {
    singleBatchTimer(
        node,
        () -> {
          Expr castedAndRenamedDfExpr =
              handleCastAndRenameBeforeWrite(inVar, colNames, bodoSqlTable);
          Variable castedAndRenamedDfVar = this.genDfVar();
          this.generatedCode.add(new Op.Assign(castedAndRenamedDfVar, castedAndRenamedDfExpr));
          this.generatedCode.add(
              new Op.Stmt(bodoSqlTable.generateWriteCode(castedAndRenamedDfVar)));
        });
  }

  /**
   * Generate Code for Streaming INSERT INTO
   *
   * @param node LogicalTableModify node to be visited
   * @param inVar Input Var containing table to write
   * @param colNames List of column names
   * @param bodoSqlTable Reference to Table to Write to
   */
  void genStreamingInsertInto(
      PandasTableModify node, Variable inVar, List<String> colNames, BodoSqlTable bodoSqlTable) {
    Expr castedAndRenamedDfExpr = handleCastAndRenameBeforeWrite(inVar, colNames, bodoSqlTable);
    Variable castedAndRenamedDfVar = this.genDfVar();
    this.generatedCode.add(new Op.Assign(castedAndRenamedDfVar, castedAndRenamedDfExpr));

    // Generate Streaming Code in this case
    // Get or create current streaming pipeline
    StreamingPipelineFrame currentPipeline = this.generatedCode.getCurrentStreamingPipeline();

    // TODO: Move to a wrapper function to avoid the timerInfo calls.
    // This requires more information about the high level design of the streaming
    // operators since there are several parts (e.g. state, multiple loop sections, etc.)
    // At this time it seems like it would be too much work to have a clean interface.
    // There may be a need to pass in several lambdas, so other changes may be needed to avoid
    // constant rewriting.
    StreamingRelNodeTimer timerInfo =
        StreamingRelNodeTimer.createStreamingTimer(
            this.generatedCode,
            this.verboseLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());
    timerInfo.initializeTimer();

    // First, create the writer state before the loop
    timerInfo.insertStateStartTimer();
    Expr writeInitCode = bodoSqlTable.generateStreamingWriteInitCode();
    Variable writerVar = this.genWriterVar();
    currentPipeline.addInitialization((new Op.Assign(writerVar, writeInitCode)));
    timerInfo.insertStateEndTimer();

    // Second, append the Dataframe to the writer
    timerInfo.insertLoopOperationStartTimer();
    Expr writerAppendCall =
        bodoSqlTable.generateStreamingWriteAppendCode(
            writerVar, castedAndRenamedDfVar, currentPipeline.getExitCond());
    this.generatedCode.add(new Op.Stmt(writerAppendCall));
    timerInfo.insertLoopOperationEndTimer();

    // Lastly, end the loop
    timerInfo.terminateTimer();
    StreamingPipelineFrame finishedPipeline = this.generatedCode.endCurrentStreamingPipeline();
    this.generatedCode.add(new Op.StreamingPipeline(finishedPipeline));
  }

  /**
   * Visitor for Insert INTO operation for SQL write.
   *
   * @param node LogicalTableModify node to be visited
   */
  public void visitInsertInto(PandasTableModify node) {
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

    if (node.getInput().getTraitSet().containsIfApplicable(BatchingProperty.SINGLE_BATCH)) {
      genSingleBatchInsertInto(node, inVar, colNames, bodoSqlTable);
    } else {
      genStreamingInsertInto(node, inVar, colNames, bodoSqlTable);
    }
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
  public void visitDelete(PandasTableModify node) {
    RelOptTableImpl relOptTable = (RelOptTableImpl) node.getTable();
    BodoSqlTable bodoSqlTable = (BodoSqlTable) relOptTable.table();
    Variable outputVar = this.genDfVar();
    List<String> outputColumns = node.getRowType().getFieldNames();
    singleBatchTimer(
        node,
        () -> {
          if (isSnowflakeCatalogTable(bodoSqlTable)) {
            // Note: Using the generic timer since we don't do the actual delete.
            // In the future we should move this to the IO timer.
            // If we are updating Snowflake we push the query into Snowflake.
            // We require the Snowflake Catalog to ensure we don't need to remap
            // any names.
            try {
              this.generatedCode.add(
                  new Op.Assign(
                      outputVar, bodoSqlTable.generateRemoteQuery(this.originalSQLQuery)));

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
          } else {
            throw new BodoSQLCodegenException(
                "Delete only supported when all source tables are found within a user's Snowflake"
                    + " account and are provided via the Snowflake catalog.");
          }
          this.varGenStack.push(outputVar);
        });
  }

  /**
   * Visitor for Logical Aggregate, support for Aggregations in SQL such as SUM, COUNT, MIN, MAX.
   *
   * @param node LogicalAggregate node to be visited
   */
  public void visitLogicalAggregate(PandasAggregate node) {
    if (node.getTraitSet().contains(BatchingProperty.STREAMING)) {
      visitStreamingLogicalAggregate(node);
    } else {
      visitSingleBatchedLogicalAggregate(node);
    }
  }

  /**
   * Visitor for Aggregate without streaming.
   *
   * @param node aggregate node being visited
   */
  private void visitSingleBatchedLogicalAggregate(PandasAggregate node) {
    final List<Integer> groupingVariables = node.getGroupSet().asList();
    final List<ImmutableBitSet> groups = node.getGroupSets();

    // Based on the calcite code that we've seen generated, we assume that every Logical Aggregation
    // node has
    // at least one grouping set.
    assert groups.size() > 0;

    List<String> expectedOutputCols = node.getRowType().getFieldNames();

    int nodeId = node.getId();
    Variable outVar = this.genDfVar();
    if (isNodeCached(node)) {
      Variable cacheKey = this.varCache.get(nodeId);
      this.generatedCode.add(new Op.Assign(outVar, cacheKey));
      varGenStack.push(outVar);
    } else {
      final List<AggregateCall> aggCallList = node.getAggCallList();

      // Expected output column names according to the calcite plan, contains any/all of the
      // expected aliases

      List<String> aggCallNames = new ArrayList<>();
      for (int i = 0; i < aggCallList.size(); i++) {
        AggregateCall aggregateCall = aggCallList.get(i);

        if (aggregateCall.getName() == null) {
          aggCallNames.add(expectedOutputCols.get(groupingVariables.size() + i));
        } else {
          aggCallNames.add(aggregateCall.getName());
        }
      }

      RelNode inputNode = node.getInput();
      this.visit(inputNode, 0, node);

      singleBatchTimer(
          node,
          () -> {
            Variable finalOutVar = outVar;
            List<String> inputColumnNames = inputNode.getRowType().getFieldNames();
            Variable inVar = varGenStack.pop();
            List<String> outputDfNames = new ArrayList<>();

            // If any group is missing a column we may need to do a concat.
            boolean hasMissingColsGroup = false;

            boolean distIfNoGroup = groups.size() > 1;

            // Naive implementation for handling multiple aggregation groups, where we repeatedly
            // call
            // group
            // by, and append the dataframes together
            for (int i = 0; i < groups.size(); i++) {
              List<Integer> curGroup = groups.get(i).toList();

              hasMissingColsGroup =
                  hasMissingColsGroup || curGroup.size() < groupingVariables.size();
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
              // assign each of the generated dataframes their own variable, for greater clarity in
              // the
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

              // We initialize the dummy column like this, as Bodo will default these columns to
              // string
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
              // Needed in the case that the topmost dataframe in the concat does not contain all
              // the
              // columns in the correct ordering
              concatExprRaw.append(".loc[:, [");

              for (int i = 0; i < expectedOutputCols.size(); i++) {
                concatExprRaw.append(makeQuoted(expectedOutputCols.get(i))).append(", ");
              }
              concatExprRaw.append("]]");

              // Generate the concatenation
              this.generatedCode.add(
                  new Op.Assign(finalOutVar, new Expr.Raw(concatExprRaw.toString())));

            } else {
              finalOutVar = new Variable(outputDfNames.get(0));
            }
            this.varCache.put(nodeId, finalOutVar);
            varGenStack.push(finalOutVar);
          });
    }
  }

  /**
   * Visitor for Aggregate with streaming.
   *
   * @param node aggregate node being visited
   */
  private void visitStreamingLogicalAggregate(PandasAggregate node) {
    // Visit the input node
    this.visit(node.getInput(), 0, node);
    Variable buildDf = varGenStack.pop();
    // Create the state var.
    // TODO: Add streaming timer support
    Variable groupbyStateVar = genGenericTempVar();
    Variable keyIndices = getStreamingGroupbyKeyIndices(node.getGroupSet(), this);
    Pair<Variable, Variable> offsetAndCols =
        getStreamingGroupbyOffsetAndCols(node.getAggCallList(), this);
    Variable offset = offsetAndCols.left;
    Variable cols = offsetAndCols.right;
    Variable ftypes = getStreamingGroupbyFtypes(node.getAggCallList(), this);
    Expr.Call stateCall =
        new Expr.Call(
            "bodo.libs.stream_groupby.init_groupby_state",
            List.of(keyIndices, ftypes, offset, cols),
            List.of());
    Op.Assign groupbyInit = new Op.Assign(groupbyStateVar, stateCall);
    // Fetch the streaming pipeline
    StreamingPipelineFrame inputPipeline = generatedCode.getCurrentStreamingPipeline();
    inputPipeline.addInitialization(groupbyInit);
    // Groupby needs the isLast to be global before calling the build side change.
    inputPipeline.ensureExitCondSynchronized();
    Variable batchExitCond = inputPipeline.getExitCond();
    Variable buildTable = genTableVar();
    Expr.Call buildDfData = getDfData(buildDf);
    int numBuildCols = node.getRowType().getFieldCount();
    List<Expr.IntegerLiteral> buildIndices = integerLiteralArange(numBuildCols);
    generateStreamingTableCode(buildIndices, buildDfData, numBuildCols, buildTable);
    Expr.Call batchCall =
        new Expr.Call(
            "bodo.libs.stream_groupby.groupby_build_consume_batch",
            List.of(groupbyStateVar, buildTable, batchExitCond));
    generatedCode.add(new Op.Stmt(batchCall));
    // Finalize and add the batch pipeline.
    generatedCode.add(new Op.StreamingPipeline(generatedCode.endCurrentStreamingPipeline()));

    // Create a new pipeline
    Variable newFlag = genGenericTempVar();
    Variable iterVar = genIterVar();
    generatedCode.startStreamingPipelineFrame(newFlag, iterVar);
    StreamingPipelineFrame outputPipeline = generatedCode.getCurrentStreamingPipeline();
    // Add the output side
    Variable outTable = genTableVar();
    Expr.Call outputCall =
        new Expr.Call(
            "bodo.libs.stream_groupby.groupby_produce_output_batch", List.of(groupbyStateVar));
    generatedCode.add(new Op.TupleAssign(List.of(outTable, newFlag), outputCall));
    // Generate an index.
    Variable indexVar = genIndexVar();
    Expr.Call lenCall = new Expr.Call("len", List.of(outTable));
    generateRangeIndexCode(lenCall, indexVar);
    // Generate a DataFrame
    Variable outDf = genDfVar();
    // Generate the column names global
    List<Expr.StringLiteral> colNamesLiteral =
        stringsToStringLiterals(node.getRowType().getFieldNames());
    generateInitOutputDfCode(colNamesLiteral, outTable, indexVar, outDf);

    // Append the code to delete the state
    Op.Stmt deleteState =
        new Op.Stmt(
            new Expr.Call(
                "bodo.libs.stream_groupby.delete_groupby_state", List.of(groupbyStateVar)));
    outputPipeline.addTermination(deleteState);
    // Add the DF to the stack
    this.varGenStack.push(outDf);
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
      Expr output =
          generateAggCodeWithGroupBy(inVar, inputColumnNames, group, aggCallList, aggCallNames);

      exprAndAdditionalGeneratedCode = new Pair<>(output, null);
    }

    return exprAndAdditionalGeneratedCode;
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
        node instanceof PandasTableScan || node instanceof PandasTargetTableScan;
    if (!supportedTableScan) {
      throw new BodoSQLCodegenException(
          "Internal error: unsupported tableScan node generated:" + node.toString());
    }
    boolean isTargetTableScan = node instanceof PandasTargetTableScan;

    singleBatchTimer(
        (PandasRel) node,
        () -> {
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
            RelOptTableImpl relTable = (RelOptTableImpl) node.getTable();
            table = (BodoSqlTable) relTable.table();

            outVar = visitSingleBatchTableScanCommon(table, isTargetTableScan, nodeId);
          }

          varGenStack.push(outVar);
        });
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
      readCode = table.generateReadCode(false, streamingOptions);
      readAssign = new Op.Assign(readVar, readCode);
    }
    this.generatedCode.add(readAssign);

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
  public Variable initStreamingIoLoop(PandasRel node, Expr readCode, List<String> columnNames) {

    Variable flagVar = genFinishedStreamingFlag();
    Variable iterVar = genIterVar();
    // start the streaming pipeline
    this.generatedCode.startStreamingPipelineFrame(flagVar, iterVar);

    // TODO: Move to a wrapper function to avoid the timerInfo calls.
    // This requires more information about the high level design of the streaming
    // operators since there are several parts (e.g. state, multiple loop sections, etc.)
    // At this time it seems like it would be too much work to have a clean interface.
    // There may be a need to pass in several lambdas, so other changes may be needed to avoid
    // constant rewriting.
    StreamingRelNodeTimer timerInfo =
        StreamingRelNodeTimer.createStreamingTimer(
            this.generatedCode,
            this.verboseLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());
    timerInfo.initializeTimer();

    StreamingPipelineFrame currentPipeline = this.generatedCode.getCurrentStreamingPipeline();
    // Create the reader before the loop
    timerInfo.insertStateStartTimer();
    Variable readerVar = this.genReaderVar();
    currentPipeline.addInitialization(new Op.Assign(readerVar, readCode));
    timerInfo.insertStateEndTimer();

    timerInfo.insertLoopOperationStartTimer();
    Variable dfChunkVar = this.genTableVar();

    Call read_arrow_next_call =
        new Call("bodo.io.arrow_reader.read_arrow_next", List.of(readerVar), List.of());
    this.generatedCode.add(new Op.TupleAssign(List.of(dfChunkVar, flagVar), read_arrow_next_call));

    Expr.Call df_len = new Call("len", dfChunkVar);

    Variable idx_var = genIndexVar();

    generateRangeIndexCode(df_len, idx_var);

    List<Expr.StringLiteral> colNamesLiteral = new ArrayList<>();
    for (int i = 0; i < columnNames.size(); i++) {
      colNamesLiteral.add(new Expr.StringLiteral(columnNames.get(i)));
    }
    Variable outVar = this.genDfVar();
    generateInitOutputDfCode(colNamesLiteral, dfChunkVar, idx_var, outVar);
    timerInfo.insertLoopOperationEndTimer();
    timerInfo.terminateTimer();

    // Delete the reader state at end of loop
    Op.Stmt deleteState =
        new Op.Stmt(new Expr.Call("bodo.io.arrow_reader.arrow_reader_del", List.of(readerVar)));
    currentPipeline.addTermination(deleteState);

    return outVar;
  }

  /**
   * Visitor for Join: Supports JOIN clause in SQL.
   *
   * @param node join node being visited
   */
  private void visitPandasJoin(PandasJoin node) {
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
  private void visitBatchedPandasJoin(PandasJoin node) {
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
      singleBatchTimer(
          node,
          () -> {
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
          });
    }
    varGenStack.push(outVar);
  }

  /**
   * Visitor for PandasJoin when it is a streaming join. Both nested loop and hash join use the same
   * API calls and the decision is made based on the values of the arguments.
   *
   * <p>Note: Streaming doesn't support caching yet.
   */
  private void visitStreamingPandasJoin(PandasJoin node) {
    // TODO: Move to a wrapper function to avoid the timerInfo calls.
    // This requires more information about the high level design of the streaming
    // operators since there are several parts (e.g. state, multiple loop sections, etc.)
    // At this time it seems like it would be too much work to have a clean interface.
    // There may be a need to pass in several lambdas, so other changes may be needed to avoid
    // constant rewriting.
    StreamingRelNodeTimer timerInfo =
        StreamingRelNodeTimer.createStreamingTimer(
            this.generatedCode,
            this.verboseLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());
    Variable joinStateVar = visitStreamingPandasJoinBatch(node, timerInfo);
    visitStreamingPandasJoinProbe(node, joinStateVar, timerInfo);
    timerInfo.terminateTimer();
  }

  private Variable visitStreamingPandasJoinState(PandasJoin node, StreamingRelNodeTimer timerInfo) {
    // Extract the Hash Join information
    timerInfo.initializeTimer();
    timerInfo.insertStateStartTimer();
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
          List.of(new kotlin.Pair<>("non_equi_condition", new Expr.StringLiteral(condString)));
    }
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
    timerInfo.insertStateEndTimer();
    return joinStateVar;
  }

  private Variable visitStreamingPandasJoinBatch(PandasJoin node, StreamingRelNodeTimer timerInfo) {
    // Visit the batch side
    this.visit(node.getLeft(), 0, node);
    Variable buildDf = varGenStack.pop();
    Variable joinStateVar = visitStreamingPandasJoinState(node, timerInfo);
    timerInfo.insertLoopOperationStartTimer();
    // Fetch the batch state
    StreamingPipelineFrame batchPipeline = generatedCode.getCurrentStreamingPipeline();
    // Join needs the isLast to be global before calling the build side change.
    batchPipeline.ensureExitCondSynchronized();
    Variable batchExitCond = batchPipeline.getExitCond();
    // Fetch the underlying table for the join.
    Variable buildTable = genTableVar();
    Expr.Call buildDfData = getDfData(buildDf);
    int numBuildCols = node.getLeft().getRowType().getFieldCount();
    List<Expr.IntegerLiteral> buildIndices = integerLiteralArange(numBuildCols);
    generateStreamingTableCode(buildIndices, buildDfData, numBuildCols, buildTable);
    Expr.Call batchCall =
        new Expr.Call(
            "bodo.libs.stream_join.join_build_consume_batch",
            List.of(joinStateVar, buildTable, batchExitCond));
    generatedCode.add(new Op.Stmt(batchCall));
    timerInfo.insertLoopOperationEndTimer();
    // Finalize and add the batch pipeline.
    generatedCode.add(new Op.StreamingPipeline(generatedCode.endCurrentStreamingPipeline()));
    return joinStateVar;
  }

  private void visitStreamingPandasJoinProbe(
      PandasJoin node, Variable joinStateVar, StreamingRelNodeTimer timerInfo) {
    // Visit the probe side
    this.visit(node.getRight(), 1, node);
    timerInfo.insertLoopOperationStartTimer();
    Variable probeDf = varGenStack.pop();
    StreamingPipelineFrame probePipeline = generatedCode.getCurrentStreamingPipeline();

    Variable oldFlag = probePipeline.getExitCond();
    // Change the probe condition
    Variable newFlag = genGenericTempVar();
    probePipeline.endSection(newFlag);
    // Fetch the underlying table for the join.
    Variable probeTable = genTableVar();
    Expr.Call probeDfData = getDfData(probeDf);
    int numProbeCols = node.getRight().getRowType().getFieldCount();
    List<Expr.IntegerLiteral> probeIndices = integerLiteralArange(numProbeCols);
    generateStreamingTableCode(probeIndices, probeDfData, numProbeCols, probeTable);
    // Add the probe side
    Variable outTable = genTableVar();
    Expr.Call probeCall =
        new Expr.Call(
            "bodo.libs.stream_join.join_probe_consume_batch",
            List.of(joinStateVar, probeTable, oldFlag));
    generatedCode.add(new Op.TupleAssign(List.of(outTable, newFlag), probeCall));
    // Generate an index.
    Expr.Call lenCall = new Expr.Call("len", List.of(outTable));
    Variable indexVar = genIndexVar();
    generateRangeIndexCode(lenCall, indexVar);
    // Generate a DataFrame
    Variable outDf = genDfVar();
    // Generate the column names global
    List<Expr.StringLiteral> colNamesLiteral =
        stringsToStringLiterals(node.getRowType().getFieldNames());
    generateInitOutputDfCode(colNamesLiteral, outTable, indexVar, outDf);
    timerInfo.insertLoopOperationEndTimer();
    // Append the code to delete the state
    Op.Stmt deleteState =
        new Op.Stmt(
            new Expr.Call("bodo.libs.stream_join.delete_join_state", List.of(joinStateVar)));
    probePipeline.addTermination(deleteState);
    // Add the DF to the stack
    this.varGenStack.push(outDf);
  }

  /**
   * Helper function for getting data from a dataframe
   *
   * @param dataframe the dataframe to retrieve data
   */
  private Expr.Call getDfData(Variable dataframe) {
    return new Expr.Call(
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data", List.of(dataframe));
  }

  /**
   * Helper function for generating table code for streaming
   *
   * @param indices indices of table
   * @param dfData dataframe data
   * @param tableCols number of logical input columns in input table
   * @param table variable of the table
   */
  private void generateStreamingTableCode(
      List<Expr.IntegerLiteral> indices, Expr.Call dfData, int tableCols, Variable table) {
    Variable colNums = lowerAsMetaType(new Expr.Tuple(indices));
    Expr.Call probeTableCall =
        new Expr.Call(
            "bodo.hiframes.table.logical_table_to_table",
            dfData,
            new Expr.Tuple(List.of()),
            colNums,
            new Expr.IntegerLiteral(tableCols));
    generatedCode.add(new Assign(table, probeTableCall));
  }

  /**
   * Helper function for generating range index code for streaming
   *
   * @param lenCall length of the range
   * @param indexVar index variable
   */
  private void generateRangeIndexCode(Expr.Call lenCall, Variable indexVar) {
    Expr.IntegerLiteral zero = new IntegerLiteral(0);
    Expr.IntegerLiteral one = new IntegerLiteral(1);
    Expr.Call indexCall =
        new Expr.Call(
            "bodo.hiframes.pd_index_ext.init_range_index",
            List.of(zero, lenCall, one, Expr.None.INSTANCE));
    generatedCode.add(new Op.Assign(indexVar, indexCall));
  }

  /**
   * Helper function for generating output dataframe code for streaming
   *
   * @param colNamesLiteral A list of column name strings
   * @param outTable output table
   * @param indexVar index variable
   * @param outDf output dataframe
   */
  private void generateInitOutputDfCode(
      List<Expr.StringLiteral> colNamesLiteral,
      Variable outTable,
      Variable indexVar,
      Variable outDf) {
    Expr.Tuple tableTuple = new Expr.Tuple(List.of(outTable));
    Expr.Tuple colNamesTuple = new Expr.Tuple(colNamesLiteral);
    Variable colNamesMeta = lowerAsColNamesMetaType(colNamesTuple);
    Expr.Call initDfCall =
        new Expr.Call(
            "bodo.hiframes.pd_dataframe_ext.init_dataframe",
            List.of(tableTuple, indexVar, colNamesMeta));
    generatedCode.add(new Op.Assign(outDf, initDfCall));
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
    singleBatchTimer(
        node,
        () -> {
          this.generatedCode.add(
              new Op.Assign(outVar, generateRowSampleCode(inpExpr, node.getParams())));
          varGenStack.push(outVar);
        });
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
    singleBatchTimer(
        node,
        () -> {
          this.generatedCode.add(
              new Op.Assign(outVar, generateSampleCode(inpExpr, node.getParams())));
          varGenStack.push(outVar);
        });
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
    // TODO[BSE-534] : Avoid visiting all the children. This was done as a precaution and
    // shouldn't be necessary.
    while (!nodeStack.isEmpty()) {
      RelNode n = nodeStack.pop();
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

  private void singleBatchTimer(PandasRel node, Runnable fn) {
    SingleBatchRelNodeTimer timerInfo =
        SingleBatchRelNodeTimer.createSingleBatchTimer(
            this.generatedCode,
            this.verboseLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());
    timerInfo.insertStartTimer();
    fn.run();
    timerInfo.insertEndTimer();
  }

  // TODO: Fuse with above so everything returns a DataFrame
  private Dataframe singleBatchTimer(PandasRel node, Supplier<Dataframe> fn) {
    SingleBatchRelNodeTimer timerInfo =
        SingleBatchRelNodeTimer.createSingleBatchTimer(
            this.generatedCode,
            this.verboseLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());
    timerInfo.insertStartTimer();
    Dataframe res = fn.get();
    timerInfo.insertEndTimer();
    return res;
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

  private RexToPandasTranslator getRexTranslator(
      int nodeId, Dataframe input, List<? extends Expr> localRefs) {
    return new RexToPandasTranslator(
        this, this.generatedCode, this.typeSystem, nodeId, input, localRefs);
  }

  private class Implementor implements PandasRel.Implementor {
    private final @NotNull PandasRel node;

    public Implementor(@NotNull PandasRel node) {
      this.node = node;
    }

    @NotNull
    @Override
    public Dataframe visitChild(@NotNull final RelNode input, final int ordinal) {
      visit(input, ordinal, node);
      Variable dfVar = varGenStack.pop();
      if (dfVar instanceof Dataframe) {
        return (Dataframe) dfVar;
      } else {
        // Ideally, all parts of code generation return dataframes
        // and this isn't needed. That is not true so fabricate the proper
        // dataframe value.
        return new Dataframe(dfVar.getName(), input.getRowType());
      }
    }

    @NotNull
    @Override
    public List<Dataframe> visitChildren(@NotNull final List<? extends RelNode> inputs) {
      return DefaultImpls.visitChildren(this, inputs);
    }

    @NotNull
    @Override
    public Dataframe build(@NotNull final Function1<? super PandasRel.BuildContext, Dataframe> fn) {
      return singleBatchTimer(node, () -> fn.invoke(new PandasCodeGenVisitor.BuildContext(node)));
    }

    @org.jetbrains.annotations.Nullable
    @Override
    public Dataframe buildStreaming(
        @NotNull Function1<? super PandasRel.BuildContext, Dataframe> fn) {
      // TODO: Move to a wrapper function to avoid the timerInfo calls.
      // This requires more information about the high level design of the streaming
      // operators since there are several parts (e.g. state, multiple loop sections, etc.)
      // At this time it seems like it would be too much work to have a clean interface.
      // There may be a need to pass in several lambdas, so other changes may be needed to avoid
      // constant rewriting.
      StreamingRelNodeTimer timerInfo =
          StreamingRelNodeTimer.createStreamingTimer(
              generatedCode,
              verboseLevel,
              node.operationDescriptor(),
              node.loggingTitle(),
              node.nodeDetails(),
              node.getTimerType());
      timerInfo.initializeTimer();
      timerInfo.insertLoopOperationStartTimer();
      Dataframe res = fn.invoke(new PandasCodeGenVisitor.BuildContext(node));
      timerInfo.insertLoopOperationEndTimer();
      timerInfo.terminateTimer();
      return res;
    }

    @org.jetbrains.annotations.Nullable
    @Override
    public Dataframe buildStreamingNoTimer(
        @NotNull Function1<? super PandasRel.BuildContext, Dataframe> fn) {
      // This is a temporary function call for sections that require manual timer calls.
      return fn.invoke(new PandasCodeGenVisitor.BuildContext(node));
    }
  }

  private class BuildContext implements PandasRel.BuildContext {
    private final @NotNull PandasRel node;

    public BuildContext(@NotNull PandasRel node) {
      this.node = node;
    }

    @NotNull
    @Override
    public Module.Builder builder() {
      return generatedCode;
    }

    @NotNull
    @Override
    public Variable lowerAsGlobal(@NotNull final Expr expression) {
      return PandasCodeGenVisitor.this.lowerAsGlobal(expression);
    }

    @NotNull
    @Override
    public RexToPandasTranslator rexTranslator(@NotNull final Dataframe input) {
      return getRexTranslator(node.getId(), input);
    }

    @NotNull
    @Override
    public RexToPandasTranslator rexTranslator(
        @NotNull final Dataframe input, @NotNull final List<? extends Expr> localRefs) {
      return getRexTranslator(node.getId(), input, localRefs);
    }

    @NotNull
    @Override
    public Dataframe returns(@NotNull final Expr result) {
      Dataframe destination = generatedCode.genDataframe(node);
      generatedCode.add(new Op.Assign(destination, result));
      return destination;
    }

    @NotNull
    @Override
    public StreamingOptions streamingOptions() {
      return streamingOptions;
    }

    @NotNull
    @Override
    public Dataframe initStreamingIoLoop(
        @NotNull final Expr expr, @NotNull final RelDataType rowType) {
      List<String> columnNames = rowType.getFieldNames();
      Variable out = PandasCodeGenVisitor.this.initStreamingIoLoop(node, expr, columnNames);
      return new Dataframe(out.getName(), node);
    }
  }
}
