package com.bodosql.calcite.application;

import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.concatDataFrames;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.generateAggCodeNoAgg;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.generateAggCodeNoGroupBy;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.generateAggCodeWithGroupBy;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.generateApplyCodeWithGroupBy;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.getStreamingGroupByKeyIndices;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.getStreamingGroupByOffsetAndCols;
import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.getStreamingGroupbyFnames;
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
import static com.bodosql.calcite.application.utils.AggHelpers.aggContainsFilter;
import static com.bodosql.calcite.application.utils.Utils.concatenateLiteralAggValue;
import static com.bodosql.calcite.application.utils.Utils.getBodoIndent;
import static com.bodosql.calcite.application.utils.Utils.integerLiteralArange;
import static com.bodosql.calcite.application.utils.Utils.isSnowflakeCatalogTable;
import static com.bodosql.calcite.application.utils.Utils.literalAggPrunedAggList;
import static com.bodosql.calcite.application.utils.Utils.makeQuoted;
import static com.bodosql.calcite.application.utils.Utils.sqlTypenameToPandasTypename;
import static com.bodosql.calcite.application.utils.Utils.stringsToStringLiterals;

import com.bodosql.calcite.adapter.pandas.ArrayRexToPandasTranslator;
import com.bodosql.calcite.adapter.pandas.PandasAggregate;
import com.bodosql.calcite.adapter.pandas.PandasIntersect;
import com.bodosql.calcite.adapter.pandas.PandasJoin;
import com.bodosql.calcite.adapter.pandas.PandasMinRowNumberFilter;
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
import com.bodosql.calcite.adapter.pandas.StreamingRexToPandasTranslator;
import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer;
import com.bodosql.calcite.application.timers.StreamingRelNodeTimer;
import com.bodosql.calcite.application.utils.RelationalOperatorCache;
import com.bodosql.calcite.application.write.WriteTarget;
import com.bodosql.calcite.application.write.WriteTarget.IfExistsBehavior;
import com.bodosql.calcite.ir.BodoEngineTable;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Module;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.OperatorType;
import com.bodosql.calcite.ir.StateVariable;
import com.bodosql.calcite.ir.StreamingPipelineFrame;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.schema.CatalogSchema;
import com.bodosql.calcite.sql.ddl.SnowflakeCreateTableMetadata;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.LocalTable;
import com.bodosql.calcite.traits.BatchingProperty;
import com.bodosql.calcite.traits.CombineStreamsExchange;
import com.bodosql.calcite.traits.SeparateStreamExchange;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Stack;
import java.util.function.Supplier;
import kotlin.Unit;
import kotlin.jvm.functions.Function1;
import kotlin.jvm.functions.Function2;
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
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Pair;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.jetbrains.annotations.NotNull;

/** Visitor class for parsed SQL nodes to generate Pandas code from SQL code. */
public class PandasCodeGenVisitor extends RelVisitor {

  /** Stack of generated tables T1, T2, etc. */
  private final Stack<BodoEngineTable> tableGenStack = new Stack<>();

  // TODO: Add this to the docs as banned
  private final Module.Builder generatedCode;

  // Note that a given query can only have one MERGE INTO statement. Therefore,
  // we can statically define the variable names we'll use for the iceberg file
  // list and snapshot
  // id,
  // since we'll only be using these variables once per query
  private static final String icebergFileListVarName = "__bodo_Iceberg_file_list";
  private static final String icebergSnapshotIDName = "__bodo_Iceberg_snapshot_id";

  private final StreamingOptions streamingOptions;

  private static final String ROW_ID_COL_NAME = "_BODO_ROW_ID";
  private static final String MERGE_ACTION_ENUM_COL_NAME = "_MERGE_INTO_CHANGE";

  /*
   * Hashmap containing globals that need to be lowered into the output func_text.
   * Used for lowering
   * metadata types to improve compilation speed.
   * hashmap maps String variable names to their String value.
   * For example loweredGlobals = {"x": "ColumnMetaDataType(('A', 'B', 'C'))"}
   * would lead to the
   * a func_text generation/execution that is equivalent to the following:
   *
   * x = ColumnMetaDataType(('A', 'B', 'C'))
   * def impl(...):
   * ...
   * init_dataframe( _, _, x)
   * ...
   *
   * (Note, we do not actually generate the above func text, we pass the values as
   * globals when calling exec in python. See
   * context.py and context_ext.py for more info)
   */
  private final HashMap<String, String> loweredGlobals;

  // The original SQL query. This is used for any operations that must be entirely
  // pushed into a remote db (e.g. Snowflake)
  private final String originalSQLQuery;

  // Debug flag set for certain tests in our test suite. Causes the codegen to
  // return simply return
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
  private @Nullable Variable mergeIntoTargetTable;
  private @Nullable TableScan mergeIntoTargetNode;
  // Extra arguments to pass to the write code for the fileList and Snapshot
  // id in the form of "argName1=varName1, argName2=varName2"
  private @Nullable String fileListAndSnapshotIdArgs;

  // TODO(aneesh) consider moving this to the C++ code, or derive the
  // chunksize for writing from the allocated budget.
  // Writes only use a constant amount of memory of 256MB. We multiply
  // that by 1.5 to allow for some wiggle room.
  private final int SNOWFLAKE_WRITE_MEMORY_ESTIMATE = ((int) (1.5 * 256 * 1024 * 1024));

  public PandasCodeGenVisitor(
      HashMap<String, String> loweredGlobalVariablesMap,
      String originalSQLQuery,
      RelDataTypeSystem typeSystem,
      boolean debuggingDeltaTable,
      int verboseLevel,
      int batchSize) {
    super();
    this.loweredGlobals = loweredGlobalVariablesMap;
    this.originalSQLQuery = originalSQLQuery;
    this.typeSystem = typeSystem;
    this.debuggingDeltaTable = debuggingDeltaTable;
    this.mergeIntoTargetTable = null;
    this.mergeIntoTargetNode = null;
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
   * Generate the new Series variable for step by step pandas codegen
   *
   * @return variable
   */
  public Variable genArrayVar() {
    return generatedCode.getSymbolTable().genArrayVar();
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
   * Generate a new output control variable for step by step pandas codegen
   *
   * @return variable
   */
  public Variable genOutputControlVar() {
    return generatedCode.getSymbolTable().genOutputControlVar();
  }

  /**
   * Generate a new input request variable for step by step pandas codegen
   *
   * @return variable
   */
  public Variable genInputRequestVar() {
    return generatedCode.getSymbolTable().genInputRequestVar();
  }

  /**
   * Generate the new temporary variable for step by step pandas codegen.
   *
   * @return variable
   */
  public Variable genGenericTempVar() {
    return generatedCode.getSymbolTable().genGenericTempVar();
  }

  public StateVariable genStateVar() {
    return generatedCode.getSymbolTable().genStateVar();
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

  public HashMap<String, Variable> globalVarCache = new HashMap<String, Variable>();

  /**
   * Modifies the codegen such that the specified expression will be lowered into the func_text as a
   * global. This is currently only used for lowering metaDataType's and array types.
   *
   * @return Variable for the global.
   */
  public Variable lowerAsGlobal(Expr expression) {
    String exprString = expression.emit();
    if (globalVarCache.containsKey(exprString)) {
      return globalVarCache.get(exprString);
    }
    Variable globalVar = generatedCode.getSymbolTable().genGlobalVar();
    this.loweredGlobals.put(globalVar.getName(), exprString);
    globalVarCache.put(exprString, globalVar);
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
   * Modifies the codegen so that an expression representing an array is stored in a new variable
   * that can be used later, to help avoid excessively long lines in the codegen.
   *
   * @param expression the expression that is to be stored in a local variable
   * @return the newly created variable that the expression was stored in
   */
  public Variable storeAsArrayVariable(Expr expression) {
    Variable var = genArrayVar();
    this.generatedCode.add(new Op.Assign(var, expression));
    return var;
  }

  /**
   * Return the final code after step by step pandas codegen. Coerces the final answer to a
   * DataFrame if it is a Table, lowering a new global in the process.
   *
   * @return generated code
   */
  public String getGeneratedCode() {
    // If the stack is size 0 we don't return a DataFrame (e.g. to_sql)
    if (this.tableGenStack.size() == 1) {
      BodoEngineTable returnTable = this.tableGenStack.pop();
      Variable outVar = generatedCode.getSymbolTable().genDfVar();
      // Generate an index.
      Variable indexVar = genIndexVar();
      Expr.Call lenCall = new Expr.Call("len", returnTable);
      Expr.Call indexCall =
          new Expr.Call(
              "bodo.hiframes.pd_index_ext.init_range_index",
              List.of(
                  Expr.Companion.getZero(), lenCall, Expr.Companion.getOne(), Expr.None.INSTANCE));
      generatedCode.add(new Op.Assign(indexVar, indexCall));
      Expr.Tuple tableTuple = new Expr.Tuple(returnTable);
      // Generate the column names global
      List<Expr.StringLiteral> colNamesLiteral =
          stringsToStringLiterals(returnTable.getRowType().getFieldNames());
      Expr.Tuple colNamesTuple = new Expr.Tuple(colNamesLiteral);
      Variable colNamesMeta = lowerAsColNamesMetaType(colNamesTuple);
      Expr dfExpr =
          new Expr.Call(
              "bodo.hiframes.pd_dataframe_ext.init_dataframe",
              List.of(tableTuple, indexVar, colNamesMeta));
      generatedCode.add(new Op.Assign(outVar, dfExpr));
      this.generatedCode.add(new Op.ReturnStatement(outVar));
    }
    assert this.tableGenStack.size() == 0
        : "Internal error: tableGenStack should contain 1 or 0 values";

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
    if (node instanceof PandasTableScan) {
      this.visitPandasTableScan((PandasTableScan) node, !(parent instanceof Filter));
    } else if (node instanceof TableScan) {
      this.visitTableScan((TableScan) node, !(parent instanceof Filter));
    } else if (node instanceof PandasJoin) {
      this.visitPandasJoin((PandasJoin) node);
    } else if (node instanceof PandasSort) {
      this.visitPandasSort((PandasSort) node);
    } else if (node instanceof PandasAggregate) {
      this.visitPandasAggregate((PandasAggregate) node);
    } else if (node instanceof PandasMinRowNumberFilter) {
      this.visitPandasMinRowNumberFilter((PandasMinRowNumberFilter) node);
    } else if (node instanceof PandasUnion) {
      this.visitPandasUnion((PandasUnion) node);
    } else if (node instanceof PandasIntersect) {
      this.visitLogicalIntersect((PandasIntersect) node);
    } else if (node instanceof PandasMinus) {
      this.visitLogicalMinus((PandasMinus) node);
    } else if (node instanceof PandasValues) {
      this.visitLogicalValues((PandasValues) node);
    } else if (node instanceof PandasTableModify) {
      this.visitPandasTableModify((PandasTableModify) node);
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

    // If we're in a distributed situation, we expect our child to return a
    // distributed dataframe,
    // and a flag that indicates if it's run out of output.
    this.visit(node.getInput(0), 0, node);

    Variable inputTableVar = tableGenStack.pop();
    int operatorID = this.generatedCode.newOperatorID();
    // Generate the list we are accumulating into.
    Variable batchAccumulatorVariable = this.genBatchAccumulatorVar();
    StreamingPipelineFrame activePipeline = this.generatedCode.getCurrentStreamingPipeline();

    // get memory estimate of input
    RelMetadataQuery mq = node.getCluster().getMetadataQuery();
    Double inputRows = mq.getRowCount(node.getInput(0));
    Double averageInputRowSize =
        Optional.ofNullable(mq.getAverageRowSize(node.getInput(0))).orElse(8.0);
    int memoryEstimate = Double.valueOf(Math.ceil(inputRows * averageInputRowSize)).intValue();
    activePipeline.initializeStreamingState(
        operatorID,
        new Op.Assign(
            batchAccumulatorVariable,
            new Expr.Call(
                "bodo.libs.table_builder.init_table_builder_state",
                new Expr.IntegerLiteral(operatorID))),
        OperatorType.ACCUMULATE_TABLE,
        memoryEstimate);

    // Append to the list at the end of the loop.
    List<Expr> args = new ArrayList<>();
    args.add(batchAccumulatorVariable);
    args.add(inputTableVar);
    Op appendStatement =
        new Op.Stmt(new Expr.Call("bodo.libs.table_builder.table_builder_append", args));
    generatedCode.add(appendStatement);
    // Finally, concatenate the batches in the accumulator into a table to use in
    // regular code.
    Variable accumulatedTable = genTableVar();
    Expr concatenatedTable =
        new Expr.Call(
            "bodo.libs.table_builder.table_builder_finalize", List.of(batchAccumulatorVariable));
    Op.Assign assign = new Op.Assign(accumulatedTable, concatenatedTable);
    activePipeline.deleteStreamingState(operatorID, assign);
    // Pop the pipeline
    StreamingPipelineFrame finishedPipeline = generatedCode.endCurrentStreamingPipeline();
    // Append the pipeline
    generatedCode.add(new Op.StreamingPipeline(finishedPipeline));
    tableGenStack.push(new BodoEngineTable(accumulatedTable.emit(), node));
  }

  private void visitSeparateStreamExchange(SeparateStreamExchange node) {
    // For information on how this node handles codegen, please see:
    // https://bodo.atlassian.net/wiki/spaces/B/pages/1337524225/Code+Generation+Design+WIP

    this.visit(node.getInput(0), 0, node);
    // Since input is single-batch, we know the RHS of the pair must be null
    Variable inTable = tableGenStack.pop();

    // Create the variable that "drives" the loop / loop exits when flag is false
    Variable exitCond = genFinishedStreamingFlag();
    // Synchronization and Slicing Counter Variable
    Variable iterVar = genIterVar();
    generatedCode.startStreamingPipelineFrame(exitCond, iterVar);
    StreamingPipelineFrame streamingInfo = generatedCode.getCurrentStreamingPipeline();

    // Precompute and save local table length
    Variable localLen = genGenericTempVar();
    Op.Assign localLenAssn =
        new Op.Assign(localLen, new Expr.Call("bodo.hiframes.table.local_len", inTable));
    streamingInfo.addInitialization(localLenAssn);

    Variable outTableVar = genTableVar();

    // Slice the table locally
    Expr sliceStart =
        new Expr.Binary("*", iterVar, new Expr.IntegerLiteral(streamingOptions.getChunkSize()));
    Expr sliceEnd =
        new Expr.Binary(
            "*",
            new Expr.Binary("+", iterVar, new Expr.IntegerLiteral(1)),
            new Expr.IntegerLiteral(streamingOptions.getChunkSize()));
    generatedCode.add(
        new Op.Assign(
            outTableVar,
            new Expr.Call(
                "bodo.hiframes.table.table_local_filter",
                inTable,
                new Expr.Call("slice", sliceStart, sliceEnd))));

    // Check if we're at the end of the local piece
    Expr done = new Expr.Binary(">=", sliceStart, localLen);
    generatedCode.add(new Op.Assign(exitCond, done));

    this.tableGenStack.push(new BodoEngineTable(outTableVar.emit(), node));
  }

  /**
   * Generic visitor method for any RelNode that implements PandasRel.
   *
   * <p>This method handles node caching, visiting inputs, passing those inputs to the node itself
   * to emit code into the Module.Builder, and generating timers when requested.
   *
   * <p>The resulting variable that is generated for this RelNode is placed on the tableGenStack.
   *
   * @param node the node to emit code for.
   */
  private void visitPandasRel(PandasRel node) {
    // Determine if this node has already been cached.
    // If it has, just return that immediately.
    RelationalOperatorCache operatorCache = generatedCode.getRelationalOperatorCache();
    if (operatorCache.isNodeCached(node)) {
      tableGenStack.push(operatorCache.getCachedTable(node));
      return;
    }

    // Note: All timer handling is done in emit
    BodoEngineTable out = node.emit(new Implementor(node));

    // Place the output variable in the tableCache and tableGenStack.
    operatorCache.tryCacheNode(node, out);
    tableGenStack.push(out);
  }

  /**
   * Visitor method for logicalValue Nodes
   *
   * @param node RelNode to be visited
   */
  private void visitLogicalValues(PandasValues node) {
    List<List<Expr>> exprCodes = new ArrayList<>();
    for (ImmutableList<RexLiteral> row : node.getTuples()) {
      List<Expr> rowLiterals = new ArrayList<>();
      for (RexLiteral colLiteral : row) {
        // We cannot be within a case statement since LogicalValues is a RelNode and
        // cannot be inside a case statement (which is a RexNode)
        Expr literalExpr = this.visitLiteralScan(colLiteral, false);
        rowLiterals.add(literalExpr);
      }
      exprCodes.add(rowLiterals);
    }
    singleBatchTimer(
        node,
        () -> {
          Expr logicalValuesExpr = generateLogicalValuesCode(exprCodes, node.getRowType(), this);
          Variable outVar = this.genTableVar();
          this.generatedCode.add(new Op.Assign(outVar, logicalValuesExpr));
          tableGenStack.push(new BodoEngineTable(outVar.emit(), node));
        });
  }

  /**
   * Visitor for streaming Pandas Union node
   *
   * @param node PandasUnion node to be visited
   */
  private void visitStreamingPandasUnion(PandasUnion node, RelationalOperatorCache operatorCache) {
    StreamingRelNodeTimer timerInfo =
        StreamingRelNodeTimer.createStreamingTimer(
            this.generatedCode,
            this.verboseLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());

    // Visit the input node
    this.visit(node.getInput(0), 0, node);
    timerInfo.initializeTimer();

    // Create the state variables
    StateVariable stateVar = genStateVar();
    kotlin.Pair<String, Expr> isAll = new kotlin.Pair<>("all", new Expr.BooleanLiteral(node.all));
    int operatorID = this.generatedCode.newOperatorID();
    Expr.Call stateCall =
        new Expr.Call(
            "bodo.libs.stream_union.init_union_state",
            List.of(new Expr.IntegerLiteral(operatorID)),
            List.of(isAll));
    Op.Assign unionInit = new Op.Assign(stateVar, stateCall);

    // Fetch the streaming pipeline
    StreamingPipelineFrame inputPipeline = generatedCode.getCurrentStreamingPipeline();
    // Add Initialization Code
    timerInfo.insertStateStartTimer();
    // UNION ALL is implemented as a ChunkedTableBuilder and doesn't need tracked in the memory
    // budget comptroller
    if (node.all) {
      inputPipeline.addInitialization(unionInit);
    } else {
      RelMetadataQuery mq = node.getCluster().getMetadataQuery();
      inputPipeline.initializeStreamingState(
          operatorID, unionInit, OperatorType.UNION, node.estimateBuildMemory(mq));
    }
    timerInfo.insertStateEndTimer();

    // In all but the last UNION call, we should set is_final_pipeline=False
    for (int i = 0; i < node.getInputs().size() - 1; i++) {
      if (i != 0) {
        RelNode input = node.getInput(i);
        this.visit(input, i, node);
      }

      StreamingPipelineFrame pipeline = generatedCode.getCurrentStreamingPipeline();
      Variable batchExitCond = pipeline.getExitCond();

      Variable newExitCond = genGenericTempVar();

      // Add Union Consume Code in Pipeline Loop
      BodoEngineTable inputTable = tableGenStack.pop();
      Expr.Call consumeCall =
          new Expr.Call(
              "bodo.libs.stream_union.union_consume_batch",
              List.of(
                  stateVar,
                  inputTable,
                  batchExitCond,
                  /*is_final_pipeline*/
                  new Expr.BooleanLiteral(false)));

      timerInfo.insertLoopOperationStartTimer();
      generatedCode.add(new Op.Assign(newExitCond, consumeCall));
      timerInfo.insertLoopOperationEndTimer();
      pipeline.endSection(newExitCond);

      // Finalize and add the batch pipeline.
      generatedCode.add(new Op.StreamingPipeline(generatedCode.endCurrentStreamingPipeline()));
    }

    // For the last UNION consume call, we need to set is_final_pipeline=True
    RelNode input = node.getInput(node.getInputs().size() - 1);
    this.visit(input, node.getInputs().size() - 1, node);

    StreamingPipelineFrame pipeline = generatedCode.getCurrentStreamingPipeline();
    Variable batchExitCond = pipeline.getExitCond();

    Variable newExitCond = genGenericTempVar();

    BodoEngineTable inputTable = tableGenStack.pop();
    Expr.Call consumeCall =
        new Expr.Call(
            "bodo.libs.stream_union.union_consume_batch",
            List.of(
                stateVar,
                inputTable,
                batchExitCond,
                /*is_final_pipeline*/
                new Expr.BooleanLiteral(true)));

    timerInfo.insertLoopOperationStartTimer();
    generatedCode.add(new Op.Assign(newExitCond, consumeCall));
    timerInfo.insertLoopOperationEndTimer();
    pipeline.endSection(newExitCond);
    StreamingPipelineFrame finishedPipeline = generatedCode.endCurrentStreamingPipeline();
    generatedCode.add(new Op.StreamingPipeline(finishedPipeline));

    // Union's produce pipeline is only a ChunkedTableBuilder with no need for memory budget so
    // ending after last input
    if (!(node.all)) {
      generatedCode.forceEndOperatorAtCurPipeline(operatorID, finishedPipeline);
    }

    // Create a new pipeline for output of table_builder
    Variable newFlag = genFinishedStreamingFlag();
    Variable iterVar = genIterVar();
    generatedCode.startStreamingPipelineFrame(newFlag, iterVar);
    StreamingPipelineFrame outputPipeline = generatedCode.getCurrentStreamingPipeline();
    // Add the output side
    Variable outTable = genTableVar();
    Variable outputControl = genOutputControlVar();
    outputPipeline.addOutputControl(outputControl);

    Expr.Call outputCall =
        new Expr.Call(
            "bodo.libs.stream_union.union_produce_batch", List.of(stateVar, outputControl));
    timerInfo.insertLoopOperationStartTimer();
    generatedCode.add(new Op.TupleAssign(List.of(outTable, newFlag), outputCall));
    timerInfo.insertLoopOperationEndTimer();

    // Append the code to delete the table builder state
    // Union state is deleted automatically by Numba
    Op.Stmt deleteState =
        new Op.Stmt(new Expr.Call("bodo.libs.stream_union.delete_union_state", List.of(stateVar)));
    outputPipeline.addTermination(deleteState);

    // Add the output table from last pipeline to the stack
    BodoEngineTable outEngineTable = new BodoEngineTable(outTable.emit(), node);
    operatorCache.tryCacheNode(node, outEngineTable);
    tableGenStack.push(outEngineTable);
    timerInfo.terminateTimer();
  }

  /**
   * Visitor for single batch / non-streaming Pandas Union
   *
   * @param node PandasUnion node to be visited
   */
  private void visitSingleBatchPandasUnion(
      PandasUnion node, RelationalOperatorCache operatorCache) {
    List<Variable> childExprs = new ArrayList<>();
    BuildContext ctx = new BuildContext(node);
    // Visit all the inputs
    for (int i = 0; i < node.getInputs().size(); i++) {
      RelNode input = node.getInput(i);
      this.visit(input, i, node);
      BodoEngineTable inputTable = tableGenStack.pop();
      childExprs.add(ctx.convertTableToDf(inputTable));
    }

    singleBatchTimer(
        node,
        () -> {
          Variable outVar = this.genDfVar();
          List<String> columnNames = node.getRowType().getFieldNames();
          Expr unionExpr = generateUnionCode(columnNames, childExprs, node.all, this);
          this.generatedCode.add(new Op.Assign(outVar, unionExpr));
          BodoEngineTable outTable = ctx.convertDfToTable(outVar, node);
          operatorCache.tryCacheNode(node, outTable);
          tableGenStack.push(outTable);
        });
  }

  /**
   * Visitor for Pandas Union node. Code generation for UNION [ALL/DISTINCT] in SQL
   *
   * @param node PandasUnion node to be visited
   */
  private void visitPandasUnion(PandasUnion node) {
    RelationalOperatorCache operatorCache = generatedCode.getRelationalOperatorCache();

    if (operatorCache.isNodeCached(node)) {
      tableGenStack.push(operatorCache.getCachedTable(node));
    } else {
      if (node.isStreaming()) {
        visitStreamingPandasUnion(node, operatorCache);
      } else {
        visitSingleBatchPandasUnion(node, operatorCache);
      }
    }
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
    BuildContext ctx = new BuildContext(node);

    // Visit the two inputs
    RelNode lhs = node.getInput(0);
    this.visit(lhs, 0, node);
    BodoEngineTable lhsTable = tableGenStack.pop();
    Variable lhsExpr = ctx.convertTableToDf(lhsTable);
    List<String> lhsColNames = lhs.getRowType().getFieldNames();

    RelNode rhs = node.getInput(1);
    this.visit(rhs, 1, node);
    BodoEngineTable rhsTable = tableGenStack.pop();
    Variable rhsExpr = ctx.convertTableToDf(rhsTable);
    List<String> rhsColNames = rhs.getRowType().getFieldNames();

    Variable outVar = this.genDfVar();
    List<String> colNames = node.getRowType().getFieldNames();
    singleBatchTimer(
        node,
        () -> {
          this.generatedCode.addAll(
              generateIntersectCode(
                  outVar, lhsExpr, lhsColNames, rhsExpr, rhsColNames, colNames, node.all, this));
          tableGenStack.push(ctx.convertDfToTable(outVar, node));
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
    BuildContext ctx = new BuildContext(node);

    // Visit the two inputs
    RelNode lhs = node.getInput(0);
    this.visit(lhs, 0, node);
    BodoEngineTable lhsTable = tableGenStack.pop();
    Variable lhsVar = ctx.convertTableToDf(lhsTable);
    List<String> lhsColNames = lhs.getRowType().getFieldNames();

    RelNode rhs = node.getInput(1);
    this.visit(rhs, 1, node);
    BodoEngineTable rhsTable = tableGenStack.pop();
    Variable rhsVar = ctx.convertTableToDf(rhsTable);
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
          tableGenStack.push(ctx.convertDfToTable(outVar, node));
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
    BuildContext ctx = new BuildContext(node);
    singleBatchTimer(
        node,
        () -> {
          List<String> colNames = input.getRowType().getFieldNames();
          /* handle case for queries with "ORDER BY" clause */
          List<RelFieldCollation> sortOrders = node.getCollation().getFieldCollations();
          BodoEngineTable inTable = tableGenStack.pop();
          Variable inVar = ctx.convertTableToDf(inTable);
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
                      + sqlTypenameToPandasTypename(
                          typeName, true, fetch.getType().getPrecision()));
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
                      + sqlTypenameToPandasTypename(
                          typeName, true, offset.getType().getPrecision()));
            }

            // Offset is either a named Parameter or a literal from parsing.
            // We visit the node to resolve the name.
            RexToPandasTranslator translator = getRexTranslator(node, inVar.emit(), input);
            offsetStr = offset.accept(translator).emit();
          }

          Expr sortExpr = generateSortCode(inVar, colNames, sortOrders, limitStr, offsetStr);
          Variable sortVar = this.genDfVar();
          this.generatedCode.add(new Op.Assign(sortVar, sortExpr));
          tableGenStack.push(ctx.convertDfToTable(sortVar, node));
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
      CatalogSchema outputSchemaAsCatalog,
      IfExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType) {
    BuildContext ctx = new BuildContext(node);
    singleBatchTimer(
        node,
        () -> {
          BodoEngineTable inTable = tableGenStack.pop();
          Variable inDf = ctx.convertTableToDf(inTable);
          this.generatedCode.add(
              new Op.Stmt(
                  outputSchemaAsCatalog.generateWriteCode(
                      this, inDf, node.getTableName(), ifExists, createTableType, node.getMeta())));
        });
  }

  /**
   * Generate Code for Streaming CREATE TABLE
   *
   * @param node Create Table Node that Code is Generated For
   * @param schema Catalog of Output Table
   * @param ifExists Action if Table Already Exists
   * @param createTableType Type of Table to Create
   * @param columnPrecisions Name of the metatype tuple storing the precision of each column.
   */
  public void genStreamingTableCreate(
      PandasTableCreate node,
      CatalogSchema schema,
      IfExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType,
      Expr columnPrecisions) {
    // Generate Streaming Code in this case
    // Get or create current streaming pipeline
    StreamingPipelineFrame currentPipeline = this.generatedCode.getCurrentStreamingPipeline();
    int operatorID = this.generatedCode.newOperatorID();
    // TODO: Move to a wrapper function to avoid the timerInfo calls.
    // This requires more information about the high level design of the streaming
    // operators since there are several parts (e.g. state, multiple loop sections,
    // etc.)
    // At this time it seems like it would be too much work to have a clean
    // interface.
    // There may be a need to pass in several lambdas, so other changes may be
    // needed to avoid
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

    // Get column names for write calls
    List<String> colNames = node.getRowType().getFieldNames();
    List<Expr.StringLiteral> colNamesList = stringsToStringLiterals(colNames);
    Variable colNamesGlobal = lowerAsColNamesMetaType(new Expr.Tuple(colNamesList));

    // Generate write destination information.
    WriteTarget writeTarget =
        schema.getCreateTableWriteTarget(
            node.getTableName(), createTableType, ifExists, colNamesGlobal);

    // First, create the writer state before the loop
    timerInfo.insertStateStartTimer();
    Expr writeState =
        writeTarget.streamingCreateTableInit(new Expr.IntegerLiteral(operatorID), createTableType);
    Variable writerVar = this.genWriterVar();
    currentPipeline.initializeStreamingState(
        operatorID,
        new Op.Assign(writerVar, writeState),
        OperatorType.SNOWFLAKE_WRITE,
        SNOWFLAKE_WRITE_MEMORY_ESTIMATE);
    timerInfo.insertStateEndTimer();

    // Second, append the Table to the writer
    timerInfo.insertLoopOperationStartTimer();
    BodoEngineTable inTable = tableGenStack.pop();

    // Generate append call
    Variable globalIsLast = genGenericTempVar();
    Expr writerAppendCall =
        writeTarget.streamingWriteAppend(
            this,
            writerVar,
            inTable,
            currentPipeline.getExitCond(),
            currentPipeline.getIterVar(),
            columnPrecisions,
            node.getMeta());
    this.generatedCode.add(new Op.Assign(globalIsLast, writerAppendCall));
    currentPipeline.endSection(globalIsLast);
    timerInfo.insertLoopOperationEndTimer();

    // Lastly, end the loop
    timerInfo.terminateTimer();
    StreamingPipelineFrame finishedPipeline = this.generatedCode.endCurrentStreamingPipeline();
    this.generatedCode.add(new Op.StreamingPipeline(finishedPipeline));
    this.generatedCode.forceEndOperatorAtCurPipeline(operatorID, finishedPipeline);
    this.generatedCode.add(writeTarget.streamingCreateTableFinalize());
  }

  public void visitLogicalTableCreate(PandasTableCreate node) {
    this.visit(node.getInput(0), 0, node);
    CatalogSchema outputSchema = node.getSchema();

    IfExistsBehavior ifExists = node.getIfExistsBehavior();
    SqlCreateTable.CreateTableType createTableType = node.getCreateTableType();

    // No streaming or single batch case
    if (node.isStreaming()) {
      List<RelDataTypeField> columnTypes = node.getRowType().getFieldList();
      List<Expr> precisions = new ArrayList<Expr>();
      for (RelDataTypeField typ : columnTypes) {
        precisions.add(new Expr.IntegerLiteral(typ.getType().getPrecision()));
      }
      Variable columnPrecisions = lowerAsMetaType(new Expr.Tuple(precisions));
      genStreamingTableCreate(node, outputSchema, ifExists, createTableType, columnPrecisions);
    } else {
      genSingleBatchTableCreate(node, outputSchema, ifExists, createTableType);
    }
  }

  /**
   * Visitor for PandasTableModify, which is used to support certain SQL write operations.
   *
   * @param node PandasTableModify node to be visited
   */
  public void visitPandasTableModify(PandasTableModify node) {
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
    BuildContext ctx = new BuildContext(node);

    RelNode input = node.getInput();
    this.visit(input, 0, node);
    singleBatchTimer(
        node,
        () -> {
          BodoEngineTable deltaTableVar = tableGenStack.pop();
          List<String> currentDeltaDfColNames = input.getRowType().getFieldNames();

          if (this.debuggingDeltaTable) {
            // If this environment variable is set, we're only testing the generation of the
            // delta
            // table.
            // Just return the delta table.
            // We drop no-ops from the delta table, as a few Calcite Optimizations can
            // result in
            // their
            // being removed from the table, and their presence/lack thereof shouldn't
            // impact
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
                            .append(deltaTableVar.emit())
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
            assert mergeIntoTargetTable != null;
            assert fileListAndSnapshotIdArgs != null;

            RelOptTableImpl relOptTable = (RelOptTableImpl) node.getTable();
            BodoSqlTable bodoSqlTable = (BodoSqlTable) relOptTable.table();
            if (!(bodoSqlTable.isWriteable() && bodoSqlTable.getDBType().equals("ICEBERG"))) {
              throw new BodoSQLCodegenException(
                  "MERGE INTO is only supported with Iceberg table destinations provided via the"
                      + " the SQL TablePath API");
            }

            // note table.getColumnNames does NOT include ROW_ID or
            // MERGE_ACTION_ENUM_COL_NAME
            // column names, because of the way they are added plan in calcite (extension
            // fields)
            // We know that the row ID and merge columns exist in the input table due to our
            // code
            // invariants
            List<String> targetTableFinalColumnNames = bodoSqlTable.getColumnNames();
            List<String> deltaTableExpectedColumnNames;
            deltaTableExpectedColumnNames = new ArrayList<>(targetTableFinalColumnNames);
            deltaTableExpectedColumnNames.add(ROW_ID_COL_NAME);
            deltaTableExpectedColumnNames.add(MERGE_ACTION_ENUM_COL_NAME);

            Variable writebackDf = genDfVar();
            Variable targetDf = genDfVar();
            Variable deltaDf = genDfVar();

            List<Expr.StringLiteral> colNamesLiteral =
                stringsToStringLiterals(this.mergeIntoTargetNode.getRowType().getFieldNames());
            Expr.Tuple colNamesTuple = new Expr.Tuple(colNamesLiteral);
            Variable targetColNamesMeta = lowerAsColNamesMetaType(colNamesTuple);

            colNamesLiteral = stringsToStringLiterals(deltaTableExpectedColumnNames);
            colNamesTuple = new Expr.Tuple(colNamesLiteral);
            Expr deltaColNamesMeta = lowerAsColNamesMetaType(colNamesTuple);

            this.generatedCode.add(
                new Op.Assign(
                    targetDf,
                    new Expr.Call(
                        "bodo.hiframes.pd_dataframe_ext.pushdown_safe_init_df",
                        this.mergeIntoTargetTable,
                        targetColNamesMeta)));
            this.generatedCode.add(
                new Op.Assign(
                    deltaDf,
                    new Expr.Call(
                        "bodo.hiframes.pd_dataframe_ext.pushdown_safe_init_df",
                        deltaTableVar,
                        deltaColNamesMeta)));

            this.generatedCode.add(
                new Op.Assign(
                    writebackDf,
                    new Expr.Call(
                        "bodosql.libs.iceberg_merge_into.do_delta_merge_with_target",
                        targetDf,
                        deltaDf)));

            // TODO: this can just be cast, since we handled rename
            Expr renamedWriteBackDfExpr =
                handleRenameBeforeWrite(writebackDf, targetTableFinalColumnNames, bodoSqlTable);
            Variable castedAndRenamedWriteBackDfVar = this.genDfVar();
            this.generatedCode.add(
                new Op.Assign(castedAndRenamedWriteBackDfVar, renamedWriteBackDfExpr));
            this.generatedCode.add(
                new Op.Stmt(
                    bodoSqlTable.generateWriteCode(
                        this, castedAndRenamedWriteBackDfVar, this.fileListAndSnapshotIdArgs)));
          }
        });
  }

  /**
   * Generate Code for Single Batch, Non-Streaming INSERT INTO
   *
   * @param node LogicalTableModify node to be visited
   * @param inputTable Input Var containing table to write
   * @param colNames List of column names
   * @param bodoSqlTable Reference to Table to Write to
   */
  void genSingleBatchInsertInto(
      PandasTableModify node,
      BodoEngineTable inputTable,
      List<String> colNames,
      BodoSqlTable bodoSqlTable) {
    singleBatchTimer(
        node,
        () -> {
          BuildContext ctx = new BuildContext(node);
          Variable inDf = ctx.convertTableToDf(inputTable);
          Expr castedAndRenamedDfExpr = handleRenameBeforeWrite(inDf, colNames, bodoSqlTable);
          Variable castedAndRenamedDfVar = this.genDfVar();
          this.generatedCode.add(new Op.Assign(castedAndRenamedDfVar, castedAndRenamedDfExpr));
          this.generatedCode.add(
              new Op.Stmt(bodoSqlTable.generateWriteCode(this, castedAndRenamedDfVar)));
        });
  }

  /**
   * Generate Code for Streaming INSERT INTO
   *
   * @param node LogicalTableModify node to be visited
   * @param inputTable Input Var containing table to write
   * @param colNames List of column names
   * @param bodoSqlTable Reference to Table to Write to
   */
  void genStreamingInsertInto(
      PandasTableModify node,
      BodoEngineTable inputTable,
      List<String> colNames,
      BodoSqlTable bodoSqlTable) {
    // Generate Streaming Code in this case
    // Get or create current streaming pipeline
    StreamingPipelineFrame currentPipeline = this.generatedCode.getCurrentStreamingPipeline();
    int operatorID = this.generatedCode.newOperatorID();

    // Get column names for write append call
    Variable colNamesGlobal =
        lowerAsColNamesMetaType(new Expr.Tuple(stringsToStringLiterals(colNames)));
    WriteTarget writeTarget = bodoSqlTable.getInsertIntoWriteTarget(colNamesGlobal);

    // TODO: Move to a wrapper function to avoid the timerInfo calls.
    // This requires more information about the high level design of the streaming
    // operators since there are several parts (e.g. state, multiple loop sections,
    // etc.)
    // At this time it seems like it would be too much work to have a clean
    // interface.
    // There may be a need to pass in several lambdas, so other changes may be
    // needed to avoid
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
    Expr writeInitCode = writeTarget.streamingInsertIntoInit(new Expr.IntegerLiteral(operatorID));
    Variable writerVar = this.genWriterVar();
    currentPipeline.initializeStreamingState(
        operatorID,
        new Op.Assign(writerVar, writeInitCode),
        OperatorType.SNOWFLAKE_WRITE,
        SNOWFLAKE_WRITE_MEMORY_ESTIMATE);
    timerInfo.insertStateEndTimer();

    // Second, append the Table to the writer
    timerInfo.insertLoopOperationStartTimer();
    Variable globalIsLast = genGenericTempVar();
    // Generate append call
    Expr writerAppendCall =
        writeTarget.streamingWriteAppend(
            this,
            writerVar,
            inputTable,
            currentPipeline.getExitCond(),
            currentPipeline.getIterVar(),
            Expr.None.INSTANCE,
            new SnowflakeCreateTableMetadata());
    this.generatedCode.add(new Op.Assign(globalIsLast, writerAppendCall));
    currentPipeline.endSection(globalIsLast);
    timerInfo.insertLoopOperationEndTimer();

    // Lastly, end the loop
    timerInfo.terminateTimer();
    StreamingPipelineFrame finishedPipeline = this.generatedCode.endCurrentStreamingPipeline();
    this.generatedCode.add(new Op.StreamingPipeline(finishedPipeline));
    this.generatedCode.forceEndOperatorAtCurPipeline(operatorID, finishedPipeline);
    this.generatedCode.add(writeTarget.streamingInsertIntoFinalize());
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
    BodoEngineTable inTable = tableGenStack.pop();
    RelOptTableImpl relOptTable = (RelOptTableImpl) node.getTable();
    BodoSqlTable bodoSqlTable = (BodoSqlTable) relOptTable.table();
    if (!bodoSqlTable.isWriteable()) {
      throw new BodoSQLCodegenException(
          "Insert Into is only supported with table destinations provided via the Snowflake"
              + "catalog or the SQL TablePath API");
    }

    if (node.isStreaming()) {
      genStreamingInsertInto(node, inTable, colNames, bodoSqlTable);
    } else {
      genSingleBatchInsertInto(node, inTable, colNames, bodoSqlTable);
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
  public Expr handleRenameBeforeWrite(
      Variable inVar, List<String> colNames, BodoSqlTable bodoSqlTable) {
    Expr outputExpr = inVar;
    Variable intermediateDf = this.genDfVar();
    this.generatedCode.add(new Op.Assign(intermediateDf, outputExpr));

    // Update column names to their write names.
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
          // Only generate the rename if at least 1 column needs renaming to avoid any
          // empty
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
    singleBatchTimer(
        node,
        () -> {
          Variable outputVar = this.genDfVar();
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
            tableGenStack.push(new BodoEngineTable(outputVar.emit(), node));
          } else {
            throw new BodoSQLCodegenException(
                "Delete only supported when all source tables are found within a user's Snowflake"
                    + " account and are provided via the Snowflake catalog.");
          }
        });
  }

  /**
   * Visitor for Pandas Aggregate, support for Aggregations in SQL such as SUM, COUNT, MIN, MAX.
   *
   * @param node BatchingProperty node to be visited
   */
  public void visitPandasAggregate(PandasAggregate node) {
    RelationalOperatorCache operatorCache = generatedCode.getRelationalOperatorCache();
    if (operatorCache.isNodeCached(node)) {
      tableGenStack.push(operatorCache.getCachedTable(node));
    } else {
      if (node.getTraitSet().contains(BatchingProperty.STREAMING)) {
        visitStreamingPandasAggregate(node);
      } else {
        visitSingleBatchedPandasAggregate(node);
      }
    }
  }

  /**
   * Visitor for Aggregate without streaming.
   *
   * @param node aggregate node being visited
   */
  private void visitSingleBatchedPandasAggregate(PandasAggregate node) {
    BuildContext ctx = new BuildContext(node);
    final List<Integer> groupingVariables = node.getGroupSet().asList();
    final List<ImmutableBitSet> groups = node.getGroupSets();

    // Based on the calcite code that we've seen generated, we assume that every
    // Logical Aggregation
    // node has
    // at least one grouping set.
    assert groups.size() > 0;

    List<String> expectedOutputCols = node.getRowType().getFieldNames();
    Variable outVar = this.genDfVar();
    final List<AggregateCall> aggCallList = node.getAggCallList();
    // Remove any LITERAL_AGG nodes.
    final List<AggregateCall> filteredAggregateCallList = literalAggPrunedAggList(aggCallList);

    // Expected output column names according to the calcite plan, contains any/all
    // of the
    // expected aliases

    List<String> aggCallNames = new ArrayList<>();
    for (int i = 0; i < filteredAggregateCallList.size(); i++) {
      AggregateCall aggregateCall = filteredAggregateCallList.get(i);

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
          BodoEngineTable inTable = tableGenStack.pop();
          Variable inVar = ctx.convertTableToDf(inTable);
          List<String> outputDfNames = new ArrayList<>();

          // If any group is missing a column we may need to do a concat.
          boolean hasMissingColsGroup = false;

          boolean distIfNoGroup = groups.size() > 1;

          // Naive implementation for handling multiple aggregation groups, where we
          // repeatedly
          // call group by, and append the dataframes together
          for (int i = 0; i < groups.size(); i++) {
            List<Integer> curGroup = groups.get(i).toList();

            hasMissingColsGroup = hasMissingColsGroup || curGroup.size() < groupingVariables.size();
            Expr curGroupAggExpr;
            /* First rename any input keys to the output. */

            /* group without aggregation : e.g. select B from table1 groupby A */
            if (filteredAggregateCallList.isEmpty()) {
              curGroupAggExpr = generateAggCodeNoAgg(inVar, inputColumnNames, curGroup);
            }
            /* aggregate without group : e.g. select sum(A) from table1 */
            else if (curGroup.isEmpty()) {
              curGroupAggExpr =
                  generateAggCodeNoGroupBy(
                      inVar,
                      inputColumnNames,
                      filteredAggregateCallList,
                      aggCallNames,
                      distIfNoGroup,
                      this);
            }
            /* group with aggregation : e.g. select sum(B) from table1 groupby A */
            else {
              Pair<Expr, @Nullable Op> curGroupAggExprAndAdditionalGeneratedCode =
                  handleLogicalAggregateWithGroups(
                      inVar, inputColumnNames, filteredAggregateCallList, aggCallNames, curGroup);

              curGroupAggExpr = curGroupAggExprAndAdditionalGeneratedCode.getKey();
              @Nullable Op prependOp = curGroupAggExprAndAdditionalGeneratedCode.getValue();
              if (prependOp != null) {
                this.generatedCode.add(prependOp);
              }
            }
            // assign each of the generated dataframes their own variable, for greater
            // clarity in
            // the
            // generated code
            Variable outDf = this.genDfVar();
            outputDfNames.add(outDf.getName());
            this.generatedCode.add(new Op.Assign(outDf, curGroupAggExpr));
          }
          // If we have multiple groups, append the dataframes together
          if (groups.size() > 1 || hasMissingColsGroup) {
            // It is not guaranteed that a particular input column exists in any of the
            // output
            // dataframes,
            // but Calcite expects
            // All input dataframes to be carried into the output. It is also not
            // guaranteed that the output dataframes contain the columns in the order
            // expected by
            // calcite.
            // In order to ensure that we have all the input columns in the output,
            // we create a dummy dataframe that has all the columns with
            // a length of 0. The ordering is handled by a loc after the concat

            // We initialize the dummy column like this, as Bodo will default these columns
            // to
            // string type if we initialize empty columns.
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

            // Sort the output dataframe, so that they are in the ordering expected by
            // Calcite
            // Needed in the case that the topmost dataframe in the concat does not contain
            // all
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
          // Generate a table using just the node members that aren't LITERAL_AGG
          final int numCols =
              node.getRowType().getFieldCount()
                  - (aggCallList.size() - filteredAggregateCallList.size());
          BodoEngineTable intermediateTable = ctx.convertDfToTable(finalOutVar, numCols);
          final BodoEngineTable outTable;
          if (node.getRowType().getFieldCount() != numCols) {
            // Insert the LITERAL_AGG results
            outTable = concatenateLiteralAggValue(this.generatedCode, ctx, intermediateTable, node);
          } else {
            outTable = intermediateTable;
          }
          RelationalOperatorCache operatorCache = generatedCode.getRelationalOperatorCache();
          operatorCache.tryCacheNode(node, outTable);
          tableGenStack.push(outTable);
        });
  }

  /**
   * Visitor for Aggregate with streaming.
   *
   * @param node aggregate node being visited
   */
  private void visitStreamingPandasAggregate(PandasAggregate node) {
    StreamingRelNodeTimer timerInfo =
        StreamingRelNodeTimer.createStreamingTimer(
            this.generatedCode,
            this.verboseLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());

    // Visit the input node
    this.visit(node.getInput(), 0, node);
    timerInfo.initializeTimer();
    Variable buildTable = tableGenStack.pop();

    // Create the state var.
    int operatorID = generatedCode.newOperatorID();
    StateVariable groupbyStateVar = genStateVar();
    List<Expr.IntegerLiteral> keyIndiciesList = getStreamingGroupByKeyIndices(node.getGroupSet());
    Variable keyIndices = this.lowerAsMetaType(new Expr.Tuple(keyIndiciesList));
    List<AggregateCall> filteredAggCallList = literalAggPrunedAggList(node.getAggCallList());
    Pair<Variable, Variable> offsetAndCols =
        getStreamingGroupByOffsetAndCols(filteredAggCallList, this, keyIndiciesList.get(0));
    Variable offset = offsetAndCols.left;
    Variable cols = offsetAndCols.right;
    Variable fnames = getStreamingGroupbyFnames(filteredAggCallList, this);
    Expr.Call stateCall =
        new Expr.Call(
            "bodo.libs.stream_groupby.init_groupby_state",
            new Expr.IntegerLiteral(operatorID),
            keyIndices,
            fnames,
            offset,
            cols);
    Op.Assign groupbyInit = new Op.Assign(groupbyStateVar, stateCall);
    // Fetch the streaming pipeline
    StreamingPipelineFrame inputPipeline = generatedCode.getCurrentStreamingPipeline();
    timerInfo.insertStateStartTimer();
    RelMetadataQuery mq = node.getCluster().getMetadataQuery();
    inputPipeline.initializeStreamingState(
        operatorID, groupbyInit, OperatorType.GROUPBY, node.estimateBuildMemory(mq));
    timerInfo.insertStateEndTimer();
    Variable batchExitCond = inputPipeline.getExitCond();
    Variable newExitCond = genGenericTempVar();
    inputPipeline.endSection(newExitCond);
    // is_final_pipeline is always True in the regular Groupby case.
    Expr.Call batchCall =
        new Expr.Call(
            "bodo.libs.stream_groupby.groupby_build_consume_batch",
            List.of(
                groupbyStateVar,
                buildTable,
                batchExitCond,
                /*is_final_pipeline*/
                new Expr.BooleanLiteral(true)));
    timerInfo.insertLoopOperationStartTimer();
    generatedCode.add(new Op.Assign(newExitCond, batchCall));
    timerInfo.insertLoopOperationEndTimer();
    // Finalize and add the batch pipeline.
    StreamingPipelineFrame finishedPipeline = generatedCode.endCurrentStreamingPipeline();
    generatedCode.add(new Op.StreamingPipeline(finishedPipeline));
    // Only Groupby build needs a memory budget since output only has a ChunkedTableBuilder
    generatedCode.forceEndOperatorAtCurPipeline(operatorID, finishedPipeline);

    // Create a new pipeline
    Variable newFlag = genGenericTempVar();
    Variable iterVar = genIterVar();
    generatedCode.startStreamingPipelineFrame(newFlag, iterVar);
    StreamingPipelineFrame outputPipeline = generatedCode.getCurrentStreamingPipeline();
    // Add the output side
    Variable outTable = genTableVar();
    Variable outputControl = genOutputControlVar();
    outputPipeline.addOutputControl(outputControl);
    Expr.Call outputCall =
        new Expr.Call(
            "bodo.libs.stream_groupby.groupby_produce_output_batch",
            List.of(groupbyStateVar, outputControl));
    timerInfo.insertLoopOperationStartTimer();
    generatedCode.add(new Op.TupleAssign(List.of(outTable, newFlag), outputCall));
    BodoEngineTable intermediateTable = new BodoEngineTable(outTable.emit(), node);
    final BodoEngineTable table;
    if (filteredAggCallList.size() != node.getAggCallList().size()) {
      // Append any Literal data if it exists.
      final BuildContext ctx = new BuildContext(node);
      table = concatenateLiteralAggValue(this.generatedCode, ctx, intermediateTable, node);
    } else {
      table = intermediateTable;
    }
    timerInfo.insertLoopOperationEndTimer();

    // Append the code to delete the state
    Op.Stmt deleteState =
        new Op.Stmt(
            new Expr.Call(
                "bodo.libs.stream_groupby.delete_groupby_state", List.of(groupbyStateVar)));
    outputPipeline.addTermination(deleteState);
    // Add the table to the stack
    tableGenStack.push(table);
    // Update the cache.
    RelationalOperatorCache operatorCache = generatedCode.getRelationalOperatorCache();
    operatorCache.tryCacheNode(node, table);
    timerInfo.terminateTimer();
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
              this.genGroupbyApplyAggFnVar(),
              this);

    } else {

      // Otherwise generate groupby.agg
      Expr output =
          generateAggCodeWithGroupBy(inVar, inputColumnNames, group, aggCallList, aggCallNames);

      exprAndAdditionalGeneratedCode = new Pair<>(output, null);
    }

    return exprAndAdditionalGeneratedCode;
  }

  /** Visitor for MinRowNumberFilter. */
  public void visitPandasMinRowNumberFilter(PandasMinRowNumberFilter node) {
    RelationalOperatorCache operatorCache = generatedCode.getRelationalOperatorCache();
    if (operatorCache.isNodeCached(node)) {
      tableGenStack.push(operatorCache.getCachedTable(node));
    } else {
      if (node.getTraitSet().contains(BatchingProperty.STREAMING)) {
        visitStreamingPandasMinRowNumberFilter(node);
      } else {
        // If non-streaming, fall back to regular PandasFilter codegen
        RelNode asProjectFilter = node.asPandasProjectFilter();
        this.visitPandasRel((PandasRel) asProjectFilter);
      }
    }
  }

  /** Visitor for MinRowNumberFilter with streaming. */
  private void visitStreamingPandasMinRowNumberFilter(PandasMinRowNumberFilter node) {
    StreamingRelNodeTimer timerInfo =
        StreamingRelNodeTimer.createStreamingTimer(
            this.generatedCode,
            this.verboseLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());

    // Visit the input node
    this.visit(node.getInput(), 0, node);
    timerInfo.initializeTimer();
    Variable buildTable = tableGenStack.pop();

    // Create the state var.
    int operatorID = generatedCode.newOperatorID();
    StateVariable groupbyStateVar = genStateVar();

    // Generate the global variables for the partition keys
    List<Expr> partitionKeys = new ArrayList<>();
    List<Integer> partitionColsIntegers = new ArrayList<>();
    for (int i : node.getPartitionColSet()) {
      partitionKeys.add(new Expr.IntegerLiteral(i));
      partitionColsIntegers.add(i);
    }
    Variable partitionGlobal = this.lowerAsMetaType(new Expr.Tuple(partitionKeys));

    // Generate the global variables for the order keys, ascending tuple, and null position tuple
    List<Expr> orderKeys = new ArrayList<>();
    List<Expr> ascendingKeys = new ArrayList<>();
    List<Expr> nullPosKeys = new ArrayList<>();
    for (int i = 0; i < node.getOrderColSet().size(); i++) {
      int orderCol = node.getOrderColSet().get(i);
      // Skip any order columns that are also partition columns
      if (node.getPartitionColSet().get(orderCol)) continue;
      // Otherwise, add the order column to the relevant lists
      boolean asc = node.getAscendingList().get(i);
      boolean nullpos = node.getNullPosList().get(i);
      orderKeys.add(new Expr.IntegerLiteral(orderCol));
      ascendingKeys.add(new Expr.BooleanLiteral(asc));
      nullPosKeys.add(new Expr.BooleanLiteral(nullpos));
    }

    Variable orderGlobal = this.lowerAsMetaType(new Expr.Tuple(orderKeys));
    Variable ascendingGlobal = this.lowerAsMetaType(new Expr.Tuple(ascendingKeys));
    Variable nullPosGlobal = this.lowerAsMetaType(new Expr.Tuple(nullPosKeys));

    // Generate the global variables for the inputs to keep
    List<Expr> keepKeys = new ArrayList<>();
    for (int i : node.getInputsToKeep()) {
      keepKeys.add(new Expr.IntegerLiteral(i));
    }
    Variable keepGlobal = this.lowerAsMetaType(new Expr.Tuple(keepKeys));

    // Generate the global variables for regular streaming groupby
    List<Integer> inColsList = new ArrayList();
    for (int i = 0; i < node.getInput().getRowType().getFieldCount(); i++) {
      if (!partitionColsIntegers.contains(i)) {
        inColsList.add(i);
      }
    }
    List<Expr> inColsExprs = new ArrayList();
    for (int i : inColsList) {
      inColsExprs.add(new Expr.IntegerLiteral(i));
    }
    Variable inColsGlobal = this.lowerAsMetaType(new Expr.Tuple(inColsExprs));
    Variable inOffsetsGlobal =
        this.lowerAsMetaType(
            new Expr.Tuple(Expr.Companion.getZero(), new Expr.IntegerLiteral(inColsList.size())));
    Variable fnamesGlobal =
        this.lowerAsMetaType(new Expr.Tuple(new Expr.StringLiteral("min_row_number_filter")));

    // Fetch the streaming pipeline
    StreamingPipelineFrame inputPipeline = generatedCode.getCurrentStreamingPipeline();
    timerInfo.insertStateStartTimer();
    RelMetadataQuery mq = node.getCluster().getMetadataQuery();

    // Insert the state initialization call
    List<Expr> positionalArgs = new ArrayList<>();
    positionalArgs.add(new Expr.IntegerLiteral(operatorID));
    positionalArgs.add(partitionGlobal);
    positionalArgs.add(fnamesGlobal);
    positionalArgs.add(inOffsetsGlobal);
    positionalArgs.add(inColsGlobal);

    List<kotlin.Pair<String, Expr>> keywordArgs = new ArrayList<>();
    keywordArgs.add(new kotlin.Pair("mrnf_sort_col_inds", orderGlobal));
    keywordArgs.add(new kotlin.Pair("mrnf_sort_col_asc", ascendingGlobal));
    keywordArgs.add(new kotlin.Pair("mrnf_sort_col_na", nullPosGlobal));
    keywordArgs.add(new kotlin.Pair("mrnf_col_inds_keep", keepGlobal));
    Expr.Call stateCall =
        new Expr.Call("bodo.libs.stream_groupby.init_groupby_state", positionalArgs, keywordArgs);
    Op.Assign mrnfInit = new Op.Assign(groupbyStateVar, stateCall);
    inputPipeline.initializeStreamingState(
        operatorID, mrnfInit, OperatorType.GROUPBY, node.estimateBuildMemory(mq));

    // Insert the consume code at the end of the current pipeline
    timerInfo.insertStateEndTimer();
    Variable batchExitCond = inputPipeline.getExitCond();
    Variable newExitCond = genGenericTempVar();
    inputPipeline.endSection(newExitCond);

    // is_final_pipeline is always True in the regular MinRowNumberFilter case.
    Expr.Call batchCall =
        new Expr.Call(
            "bodo.libs.stream_groupby.groupby_build_consume_batch",
            List.of(
                groupbyStateVar,
                buildTable,
                batchExitCond,
                /*is_final_pipeline*/
                new Expr.BooleanLiteral(true)));
    timerInfo.insertLoopOperationStartTimer();
    generatedCode.add(new Op.Assign(newExitCond, batchCall));
    timerInfo.insertLoopOperationEndTimer();

    // Finalize and add the batch pipeline.
    StreamingPipelineFrame finishedPipeline = generatedCode.endCurrentStreamingPipeline();
    generatedCode.add(new Op.StreamingPipeline(finishedPipeline));

    // Only MRNF build needs a memory budget since output only has a ChunkedTableBuilder
    generatedCode.forceEndOperatorAtCurPipeline(operatorID, finishedPipeline);

    // Create a new pipeline
    Variable newFlag = genGenericTempVar();
    Variable iterVar = genIterVar();
    generatedCode.startStreamingPipelineFrame(newFlag, iterVar);
    StreamingPipelineFrame outputPipeline = generatedCode.getCurrentStreamingPipeline();

    // Add the output side
    Variable outTable = genTableVar();
    Variable outputControl = genOutputControlVar();
    outputPipeline.addOutputControl(outputControl);
    Expr.Call outputCall =
        new Expr.Call(
            "bodo.libs.stream_groupby.groupby_produce_output_batch",
            List.of(groupbyStateVar, outputControl));
    timerInfo.insertLoopOperationStartTimer();
    generatedCode.add(new Op.TupleAssign(List.of(outTable, newFlag), outputCall));
    BodoEngineTable table = new BodoEngineTable(outTable.emit(), node);
    timerInfo.insertLoopOperationEndTimer();

    // Append the code to delete the state
    Op.Stmt deleteState =
        new Op.Stmt(
            new Expr.Call(
                "bodo.libs.stream_groupby.delete_groupby_state", List.of(groupbyStateVar)));
    outputPipeline.addTermination(deleteState);
    // Add the table to the stack
    tableGenStack.push(table);
    // Update the cache.
    RelationalOperatorCache operatorCache = generatedCode.getRelationalOperatorCache();
    operatorCache.tryCacheNode(node, table);
    timerInfo.terminateTimer();
  }

  /**
   * Visitor for Rex Literals.
   *
   * @param node RexLiteral being visited
   * @param node isSingleRow flag for if table references refer to a single row or the whole table.
   *     This is used for determining if an expr returns a scalar or a column. Only CASE statements
   *     set this to True currently.
   * @return Expr for the literal rexpression.
   */
  public Expr visitLiteralScan(RexLiteral node, boolean isSingleRow) {
    return generateLiteralCode(node, isSingleRow, this);
  }

  /**
   * Visitor for Table Scan. It acts as a somewhat simple wrapper for PandasRel.emit with some
   * special checks
   *
   * @param node TableScan node being visited
   * @param canLoadFromCache Can we load the variable from cache? This is set to False if we have a
   *     filter that wasn't previously cached to enable filter pushdown.
   */
  public void visitPandasTableScan(PandasTableScan node, boolean canLoadFromCache) {
    // Determine if this node has already been cached.
    // If it has, just return that immediately.
    RelationalOperatorCache operatorCache = generatedCode.getRelationalOperatorCache();
    if (canLoadFromCache && operatorCache.isNodeCached(node)) {
      tableGenStack.push(operatorCache.getCachedTable(node));
      return;
    }

    // Note: All timer handling is done in emit
    BodoEngineTable out = node.emit(new Implementor(node));

    // Place the output variable in the tableCache (if possible) and tableGenStack.
    operatorCache.tryCacheNode(node, out);
    tableGenStack.push(out);
  }

  /**
   * Visitor for Table Scan.
   *
   * @param node TableScan node being visited
   * @param canLoadFromCache Can we load the variable from cache? This is set to False if we have a
   *     filter that wasn't previously cached to enable filter pushdown.
   */
  public void visitTableScan(TableScan node, boolean canLoadFromCache) {
    boolean isTargetTableScan = node instanceof PandasTargetTableScan;
    if (!isTargetTableScan) {
      throw new BodoSQLCodegenException(
          "Internal error: unsupported tableScan node generated:" + node.toString());
    }
    PandasTargetTableScan nodeCasted = (PandasTargetTableScan) node;

    singleBatchTimer(
        nodeCasted,
        () -> {
          BodoEngineTable outTable;
          RelationalOperatorCache operatorCache = generatedCode.getRelationalOperatorCache();
          if (canLoadFromCache && operatorCache.isNodeCached(nodeCasted)) {
            outTable = operatorCache.getCachedTable(nodeCasted);
          } else {
            BodoSqlTable table;

            // TODO(jsternberg): The proper way to do this is to have the individual nodes
            // handle the code generation. Due to the way the code generation is
            // constructed,
            // we can't really do that so we're just going to hack around it for now to
            // avoid
            // a large refactor
            RelOptTableImpl relTable = (RelOptTableImpl) nodeCasted.getTable();
            table = (BodoSqlTable) relTable.table();

            // IsTargetTableScan is always true, we check for this at the start of the
            // function.
            outTable = visitSingleBatchTableScanCommon(nodeCasted, table, isTargetTableScan);
          }
          operatorCache.tryCacheNode(nodeCasted, outTable);
          tableGenStack.push(outTable);
        });
  }

  /**
   * Helper function that contains the code needed to perform a read of the specified table.
   *
   * @param node The rel node for the scan
   * @param table The BodoSqlTable to read
   * @param isTargetTableScan Is the read a TargetTableScan (used in MERGE INTO)
   * @return outVar The returned dataframe variable
   */
  public BodoEngineTable visitSingleBatchTableScanCommon(
      TableScan node, BodoSqlTable table, boolean isTargetTableScan) {
    Expr readCode;
    Op readAssign;

    Variable readVar = this.genTableVar();
    // Add the table to cached values
    if (isTargetTableScan) {
      // TODO: Properly restrict to Iceberg.
      if (!(table instanceof LocalTable) || !table.getDBType().equals("ICEBERG")) {
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
      this.generatedCode.add(readAssign);
      mergeIntoTargetNode = node;
      mergeIntoTargetTable = readVar;
    } else {
      readCode = table.generateReadCode(false, streamingOptions);
      readAssign = new Op.Assign(readVar, readCode);
      this.generatedCode.add(readAssign);
    }

    BodoEngineTable outTable;
    BuildContext ctx = new BuildContext((PandasRel) node);

    Expr castExpr = table.generateReadCastCode(readVar);
    if (!castExpr.equals(readVar)) {
      // Generate a new outVar to avoid shadowing
      Variable outVar = this.genDfVar();
      this.generatedCode.add(new Op.Assign(outVar, castExpr));
      outTable = ctx.convertDfToTable(outVar, node);
    } else {
      outTable = new BodoEngineTable(readVar.emit(), node);
    }

    return outTable;
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
    BuildContext ctx = new BuildContext(node);
    /* get left/right tables */

    List<String> outputColNames = node.getRowType().getFieldNames();
    RelationalOperatorCache operatorCache = generatedCode.getRelationalOperatorCache();
    if (operatorCache.isNodeCached(node)) {
      BodoEngineTable tableVar = operatorCache.getCachedTable(node);
      tableGenStack.push(tableVar);
    } else {
      Variable outDfVar = this.genDfVar();
      this.visit(node.getLeft(), 0, node);
      List<String> leftColNames = node.getLeft().getRowType().getFieldNames();
      Variable leftTable = ctx.convertTableToDf(tableGenStack.pop());
      this.visit(node.getRight(), 1, node);
      List<String> rightColNames = node.getRight().getRowType().getFieldNames();
      Variable rightTable = ctx.convertTableToDf(tableGenStack.pop());
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
                    outDfVar,
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

            BodoEngineTable outTable = ctx.convertDfToTable(outDfVar, node);
            operatorCache.tryCacheNode(node, outTable);
            tableGenStack.push(outTable);
          });
    }
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
    // operators since there are several parts (e.g. state, multiple loop sections,
    // etc.)
    // At this time it seems like it would be too much work to have a clean
    // interface.
    // There may be a need to pass in several lambdas, so other changes may be
    // needed to avoid
    // constant rewriting.
    int operatorID = this.generatedCode.newOperatorID();
    StreamingRelNodeTimer timerInfo =
        StreamingRelNodeTimer.createStreamingTimer(
            this.generatedCode,
            this.verboseLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());
    StateVariable joinStateVar = visitStreamingPandasJoinBatch(node, timerInfo, operatorID);
    visitStreamingPandasJoinProbe(node, joinStateVar, timerInfo, operatorID);
    timerInfo.terminateTimer();
  }

  private StateVariable visitStreamingPandasJoinState(
      PandasJoin node, StreamingRelNodeTimer timerInfo, int operatorID) {
    // Extract the Hash Join information
    timerInfo.initializeTimer();
    timerInfo.insertStateStartTimer();
    JoinInfo joinInfo = node.analyzeCondition();
    Pair<Variable, Variable> keyIndices = getStreamingJoinKeyIndices(joinInfo, this);
    // SQL convention is that probe table is on the left and build table is on the
    // right.
    List<String> probeNodeNames = node.getLeft().getRowType().getFieldNames();
    List<String> buildNodeNames = node.getRight().getRowType().getFieldNames();
    // Fetch the names for each child.
    List<Expr.StringLiteral> probeColNames = stringsToStringLiterals(probeNodeNames);
    List<Expr.StringLiteral> buildColNames = stringsToStringLiterals(buildNodeNames);
    Variable probeNamesGlobal = lowerAsColNamesMetaType(new Expr.Tuple(probeColNames));
    Variable buildNamesGlobal = lowerAsColNamesMetaType(new Expr.Tuple(buildColNames));
    // Get the non equi-join info
    Expr nonEquiCond =
        visitNonEquiConditions(joinInfo.nonEquiConditions, probeNodeNames, buildNodeNames);
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
    StateVariable joinStateVar = genStateVar();
    Expr.BooleanLiteral isLeftOuter =
        new Expr.BooleanLiteral(node.getJoinType().generatesNullsOnRight());
    Expr.BooleanLiteral isRightOuter =
        new Expr.BooleanLiteral(node.getJoinType().generatesNullsOnLeft());
    Expr.Call stateCall =
        new Expr.Call(
            "bodo.libs.stream_join.init_join_state",
            List.of(
                new Expr.IntegerLiteral(operatorID),
                keyIndices.right,
                keyIndices.left,
                buildNamesGlobal,
                probeNamesGlobal,
                isRightOuter,
                isLeftOuter),
            namedArgs);
    Op.Assign joinInit = new Op.Assign(joinStateVar, stateCall);

    RelMetadataQuery mq = node.getCluster().getMetadataQuery();
    batchPipeline.initializeStreamingState(
        operatorID, joinInit, OperatorType.JOIN, node.estimateBuildMemory(mq));

    timerInfo.insertStateEndTimer();
    return joinStateVar;
  }

  private StateVariable visitStreamingPandasJoinBatch(
      PandasJoin node, StreamingRelNodeTimer timerInfo, int operatorID) {
    // Visit the batch side
    this.visit(node.getRight(), 0, node);
    BodoEngineTable buildTable = tableGenStack.pop();
    StateVariable joinStateVar = visitStreamingPandasJoinState(node, timerInfo, operatorID);
    timerInfo.insertLoopOperationStartTimer();
    // Fetch the batch state
    StreamingPipelineFrame batchPipeline = generatedCode.getCurrentStreamingPipeline();
    Variable batchExitCond = batchPipeline.getExitCond();
    Variable newExitCond = genGenericTempVar();
    batchPipeline.endSection(newExitCond);
    Expr.Call batchCall =
        new Expr.Call(
            "bodo.libs.stream_join.join_build_consume_batch",
            List.of(joinStateVar, buildTable, batchExitCond));
    generatedCode.add(new Op.Assign(newExitCond, batchCall));
    timerInfo.insertLoopOperationEndTimer();
    // Finalize and add the batch pipeline.
    generatedCode.add(new Op.StreamingPipeline(generatedCode.endCurrentStreamingPipeline()));
    return joinStateVar;
  }

  private void visitStreamingPandasJoinProbe(
      PandasJoin node,
      StateVariable joinStateVar,
      StreamingRelNodeTimer timerInfo,
      int operatorID) {
    // Visit the probe side
    this.visit(node.getLeft(), 1, node);
    timerInfo.insertLoopOperationStartTimer();
    BodoEngineTable probeTable = tableGenStack.pop();
    StreamingPipelineFrame probePipeline = generatedCode.getCurrentStreamingPipeline();

    Variable oldFlag = probePipeline.getExitCond();
    // Change the probe condition
    Variable newFlag = genGenericTempVar();
    probePipeline.endSection(newFlag);
    // Add the probe side
    Variable outTable = genTableVar();
    Variable outputControl = genOutputControlVar();
    Variable inputRequest = genInputRequestVar();
    Expr.Call probeCall =
        new Expr.Call(
            "bodo.libs.stream_join.join_probe_consume_batch",
            List.of(joinStateVar, probeTable, oldFlag, outputControl));
    generatedCode.add(new Op.TupleAssign(List.of(outTable, newFlag, inputRequest), probeCall));
    probePipeline.addInputRequest(inputRequest);
    probePipeline.addOutputControl(outputControl);

    timerInfo.insertLoopOperationEndTimer();
    // Append the code to delete the state
    Op.Stmt deleteState =
        new Op.Stmt(
            new Expr.Call("bodo.libs.stream_join.delete_join_state", List.of(joinStateVar)));
    probePipeline.deleteStreamingState(operatorID, deleteState);
    // Add the table to the stack
    tableGenStack.push(new BodoEngineTable(outTable.emit(), node));
  }

  /**
   * Visitor for RowSample: Supports SAMPLE clause in SQL with a fixed number of rows.
   *
   * @param node rowSample node being visited
   */
  public void visitRowSample(PandasRowSample node) {
    // We always assume row sample has exactly one input
    assert node.getInputs().size() == 1;
    BuildContext ctx = new BuildContext(node);

    // Visit the input
    RelNode inp = node.getInput(0);
    this.visit(inp, 0, node);
    singleBatchTimer(
        node,
        () -> {
          BodoEngineTable inTable = tableGenStack.pop();
          Variable inDf = ctx.convertTableToDf(inTable);
          Expr rowSampleExpr = generateRowSampleCode(inDf, node.getParams());
          Variable outVar = this.genDfVar();
          this.generatedCode.add(new Op.Assign(outVar, rowSampleExpr));
          BodoEngineTable outTable = ctx.convertDfToTable(outVar, node);
          tableGenStack.push(outTable);
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
    BuildContext ctx = new BuildContext(node);

    // Visit the input
    RelNode inp = node.getInput(0);
    this.visit(inp, 0, node);
    BodoEngineTable inTable = tableGenStack.pop();
    Variable inDf = ctx.convertTableToDf(inTable);

    Variable outVar = this.genDfVar();
    singleBatchTimer(
        node,
        () -> {
          this.generatedCode.add(new Op.Assign(outVar, generateSampleCode(inDf, node.getParams())));
          BodoEngineTable outTable = ctx.convertDfToTable(outVar, node);
          tableGenStack.push(outTable);
        });
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
  private BodoEngineTable singleBatchTimer(PandasRel node, Supplier<BodoEngineTable> fn) {
    SingleBatchRelNodeTimer timerInfo =
        SingleBatchRelNodeTimer.createSingleBatchTimer(
            this.generatedCode,
            this.verboseLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());
    timerInfo.insertStartTimer();
    BodoEngineTable res = fn.get();
    timerInfo.insertEndTimer();
    return res;
  }

  private RexToPandasTranslator getRexTranslator(RelNode self, String inVar, RelNode input) {
    return getRexTranslator(self, new BodoEngineTable(inVar, input));
  }

  private RexToPandasTranslator getRexTranslator(RelNode self, BodoEngineTable input) {
    return getRexTranslator(self.getId(), input);
  }

  private RexToPandasTranslator getRexTranslator(int nodeId, BodoEngineTable input) {
    return new RexToPandasTranslator(this, this.generatedCode, this.typeSystem, nodeId, input);
  }

  private RexToPandasTranslator getRexTranslator(
      int nodeId, BodoEngineTable input, List<? extends Expr> localRefs) {
    return new RexToPandasTranslator(
        this, this.generatedCode, this.typeSystem, nodeId, input, localRefs);
  }

  private ArrayRexToPandasTranslator getArrayRexTranslator(int nodeId, BodoEngineTable input) {
    return new ArrayRexToPandasTranslator(this, this.generatedCode, this.typeSystem, nodeId, input);
  }

  private StreamingRexToPandasTranslator getStreamingRexTranslator(
      int nodeId,
      BodoEngineTable input,
      List<? extends Expr> localRefs,
      @NotNull StateVariable stateVar) {
    return new StreamingRexToPandasTranslator(
        this, this.generatedCode, this.typeSystem, nodeId, input, localRefs, stateVar);
  }

  private class Implementor implements PandasRel.Implementor {
    private final @NotNull PandasRel node;

    public Implementor(@NotNull PandasRel node) {
      this.node = node;
    }

    @NotNull
    @Override
    public BodoEngineTable visitChild(@NotNull final RelNode input, final int ordinal) {
      visit(input, ordinal, node);
      return tableGenStack.pop();
    }

    @NotNull
    @Override
    public List<BodoEngineTable> visitChildren(@NotNull final List<? extends RelNode> inputs) {
      return DefaultImpls.visitChildren(this, inputs);
    }

    @NotNull
    @Override
    public BodoEngineTable build(
        @NotNull final Function1<? super PandasRel.BuildContext, BodoEngineTable> fn) {
      int operatorID = generatedCode.newOperatorID();
      return singleBatchTimer(node, () -> fn.invoke(new PandasCodeGenVisitor.BuildContext(node)));
    }

    @NotNull
    @Override
    public BodoEngineTable buildStreaming(
        @NotNull Function1<? super PandasRel.BuildContext, StateVariable> initFn,
        @NotNull
            Function2<? super PandasRel.BuildContext, ? super StateVariable, BodoEngineTable>
                bodyFn,
        @NotNull Function2<? super PandasRel.BuildContext, ? super StateVariable, Unit> deleteFn) {
      // TODO: Move to a wrapper function to avoid the timerInfo calls.
      // This requires more information about the high level design of the streaming
      // operators since there are several parts (e.g. state, multiple loop sections,
      // etc.)
      // At this time it seems like it would be too much work to have a clean
      // interface.
      // There may be a need to pass in several lambdas, so other changes may be
      // needed to avoid
      // constant rewriting.
      StreamingRelNodeTimer timerInfo =
          StreamingRelNodeTimer.createStreamingTimer(
              generatedCode,
              verboseLevel,
              node.operationDescriptor(),
              node.loggingTitle(),
              node.nodeDetails(),
              node.getTimerType());
      int operatorID = generatedCode.newOperatorID();
      PandasCodeGenVisitor.BuildContext buildContext =
          new PandasCodeGenVisitor.BuildContext(node, operatorID);
      timerInfo.initializeTimer();
      // Init the state and time it.
      timerInfo.insertStateStartTimer();
      StateVariable stateVar = initFn.invoke(buildContext);
      timerInfo.insertStateEndTimer();
      // Handle the loop body
      timerInfo.insertLoopOperationStartTimer();
      BodoEngineTable res = bodyFn.invoke(buildContext, stateVar);
      timerInfo.insertLoopOperationEndTimer();
      // Delete the state
      deleteFn.invoke(buildContext, stateVar);
      timerInfo.terminateTimer();
      return res;
    }

    @Override
    public void createStreamingPipeline() {
      Variable exitCond = genFinishedStreamingFlag();
      Variable iterVariable = genIterVar();
      generatedCode.startStreamingPipelineFrame(exitCond, iterVariable);
    }
  }

  private class BuildContext implements PandasRel.BuildContext {
    private final @NotNull PandasRel node;
    private final int operatorID;

    public BuildContext(@NotNull PandasRel node) {
      this.node = node;
      this.operatorID = -1;
    }

    public BuildContext(@NotNull PandasRel node, int operatorID) {
      this.node = node;
      this.operatorID = operatorID;
    }

    @Override
    public int operatorID() {
      return operatorID;
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
    public Variable lowerAsMetaType(@NotNull final Expr expression) {
      return PandasCodeGenVisitor.this.lowerAsMetaType(expression);
    }

    @NotNull
    @Override
    public RexToPandasTranslator rexTranslator(@NotNull final BodoEngineTable input) {
      return getRexTranslator(node.getId(), input);
    }

    @NotNull
    @Override
    public RexToPandasTranslator rexTranslator(
        @NotNull final BodoEngineTable input, @NotNull final List<? extends Expr> localRefs) {
      return getRexTranslator(node.getId(), input, localRefs);
    }

    @NotNull
    @Override
    public ArrayRexToPandasTranslator arrayRexTranslator(@NotNull final BodoEngineTable input) {
      return getArrayRexTranslator(node.getId(), input);
    }

    @NotNull
    @Override
    public BodoEngineTable returns(@NotNull final Expr result) {
      Variable destination = generatedCode.getSymbolTable().genTableVar();
      generatedCode.add(new Op.Assign(destination, result));
      return new BodoEngineTable(destination.emit(), node);
    }

    /**
     * Converts a DataFrame into a Table using an input rel node to infer the number of columns in
     * the DataFrame.
     *
     * @param df The DataFrame that is being converted into a Table.
     * @param node The rel node whose output schema is used to infer the number of columns in the
     *     DataFrame.
     * @return The BodoEngineTable that the DataFrame has been converted into.
     */
    @NotNull
    @Override
    public BodoEngineTable convertDfToTable(Variable df, RelNode node) {
      return convertDfToTable(df, node.getRowType().getFieldCount());
    }

    /**
     * Converts a DataFrame into a Table using an input row type to infer the number of columns in
     * the DataFrame.
     *
     * @param df The DataFrame that is being converted into a Table.
     * @param rowType The output schema that is used to infer the number of columns in the
     *     DataFrame.
     * @return The BodoEngineTable that the DataFrame has been converted into.
     */
    @NotNull
    @Override
    public BodoEngineTable convertDfToTable(Variable df, RelDataType rowType) {
      return convertDfToTable(df, rowType.getFieldCount());
    }

    @NotNull
    @Override
    public BodoEngineTable convertDfToTable(Variable df, int numCols) {
      Variable outVar = generatedCode.getSymbolTable().genTableVar();
      List<Expr.IntegerLiteral> buildIndices = integerLiteralArange(numCols);
      Variable buildColNums = lowerAsMetaType(new Expr.Tuple(buildIndices));
      Expr dfData = new Expr.Call("bodo.hiframes.pd_dataframe_ext.get_dataframe_all_data", df);
      Expr tableExpr =
          new Expr.Call(
              "bodo.hiframes.table.logical_table_to_table",
              dfData,
              new Expr.Tuple(),
              buildColNums,
              new Expr.IntegerLiteral(numCols));
      generatedCode.add(new Op.Assign(outVar, tableExpr));
      return new BodoEngineTable(outVar.emit(), node);
    }

    /**
     * Converts a Table into a DataFrame.
     *
     * @param table The table that is being converted into a DataFrame.
     * @return The variable used to store the new DataFrame.
     */
    @NotNull
    @Override
    public Variable convertTableToDf(BodoEngineTable table) {
      Variable outVar = generatedCode.getSymbolTable().genDfVar();
      // Generate an index.
      Variable indexVar = genIndexVar();
      Expr.Call lenCall = new Expr.Call("len", table);
      Expr.Call indexCall =
          new Expr.Call(
              "bodo.hiframes.pd_index_ext.init_range_index",
              List.of(
                  Expr.Companion.getZero(), lenCall, Expr.Companion.getOne(), Expr.None.INSTANCE));
      generatedCode.add(new Op.Assign(indexVar, indexCall));
      Expr.Tuple tableTuple = new Expr.Tuple(table);
      // Generate the column names global
      List<Expr.StringLiteral> colNamesLiteral =
          stringsToStringLiterals(table.getRowType().getFieldNames());
      Expr.Tuple colNamesTuple = new Expr.Tuple(colNamesLiteral);
      Variable colNamesMeta = lowerAsColNamesMetaType(colNamesTuple);
      Expr dfExpr =
          new Expr.Call(
              "bodo.hiframes.pd_dataframe_ext.init_dataframe",
              List.of(tableTuple, indexVar, colNamesMeta));
      generatedCode.add(new Op.Assign(outVar, dfExpr));
      return outVar;
    }

    @NotNull
    @Override
    public StreamingOptions streamingOptions() {
      return streamingOptions;
    }

    @NotNull
    @Override
    public StreamingRexToPandasTranslator streamingRexTranslator(
        @NotNull BodoEngineTable input,
        @NotNull List<? extends Expr> localRefs,
        @NotNull StateVariable stateVar) {
      return getStreamingRexTranslator(node.getId(), input, localRefs, stateVar);
    }
  }
}
