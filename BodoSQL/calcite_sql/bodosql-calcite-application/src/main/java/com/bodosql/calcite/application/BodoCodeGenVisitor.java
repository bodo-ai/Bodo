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
import static com.bodosql.calcite.application.JoinCondVisitor.getStreamingJoinKeyIndices;
import static com.bodosql.calcite.application.JoinCondVisitor.visitJoinCond;
import static com.bodosql.calcite.application.JoinCondVisitor.visitNonEquiConditions;
import static com.bodosql.calcite.application.utils.AggHelpers.aggContainsFilter;
import static com.bodosql.calcite.application.utils.BodoArrayHelpers.sqlTypeToBodoArrayType;
import static com.bodosql.calcite.application.utils.Utils.concatenateLiteralAggValue;
import static com.bodosql.calcite.application.utils.Utils.integerLiteralArange;
import static com.bodosql.calcite.application.utils.Utils.isSnowflakeCatalogTable;
import static com.bodosql.calcite.application.utils.Utils.literalAggPrunedAggList;
import static com.bodosql.calcite.application.utils.Utils.makeQuoted;
import static com.bodosql.calcite.application.utils.Utils.stringsToStringLiterals;

import com.bodosql.calcite.adapter.bodo.ArrayRexToBodoTranslator;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalAggregate;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalIntersect;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalJoin;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalMinRowNumberFilter;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalMinus;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalRowSample;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalSample;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalTableCreate;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalTableModify;
import com.bodosql.calcite.adapter.bodo.BodoPhysicalValues;
import com.bodosql.calcite.adapter.bodo.RexToBodoTranslator;
import com.bodosql.calcite.adapter.bodo.ScalarRexToBodoTranslator;
import com.bodosql.calcite.adapter.bodo.StreamingOptions;
import com.bodosql.calcite.adapter.bodo.StreamingRexToBodoTranslator;
import com.bodosql.calcite.adapter.common.TimerSupportedRel;
import com.bodosql.calcite.adapter.common.TreeReverserDuplicateTracker;
import com.bodosql.calcite.adapter.pandas.PandasRel;
import com.bodosql.calcite.adapter.pandas.PandasTargetTableScan;
import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer;
import com.bodosql.calcite.application.timers.StreamingRelNodeTimer;
import com.bodosql.calcite.application.utils.RelationalOperatorCache;
import com.bodosql.calcite.application.write.WriteTarget;
import com.bodosql.calcite.application.write.WriteTarget.IfExistsBehavior;
import com.bodosql.calcite.codeGeneration.OperatorEmission;
import com.bodosql.calcite.ddl.GenerateDDLTypes;
import com.bodosql.calcite.ir.BodoEngineTable;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Module;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.OperatorID;
import com.bodosql.calcite.ir.OperatorType;
import com.bodosql.calcite.ir.StateVariable;
import com.bodosql.calcite.ir.StreamingPipelineFrame;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.schema.CatalogSchema;
import com.bodosql.calcite.sql.ddl.CreateTableMetadata;
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
import java.util.Map;
import java.util.Optional;
import java.util.Stack;
import java.util.function.Supplier;
import kotlin.jvm.functions.Function1;
import org.apache.calcite.prepare.RelOptTableImpl;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelVisitor;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Correlate;
import org.apache.calcite.rel.core.JoinInfo;
import org.apache.calcite.rel.core.TableCreate;
import org.apache.calcite.rel.core.TableModify;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Pair;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.jetbrains.annotations.NotNull;

/** Visitor class for parsed SQL nodes to generate Bodo code from SQL code. */
public class BodoCodeGenVisitor extends RelVisitor {

  /** Stack of generated tables T1, T2, etc. */
  private final Stack<BodoEngineTable> tableGenStack = new Stack<>();

  // TODO: Add this to the docs as banned
  private final Module.Builder generatedCode;

  // Cache for relational operators.
  // TODO: Move this to an explicit state class that isn't code generation.
  private final RelationalOperatorCache relationalOperatorCache = new RelationalOperatorCache();

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

  // The type system, used to access timezone info during codegen
  private final RelDataTypeSystem typeSystem;

  // Bodo verbose level. This is used to generate code/compiler information
  // with extra debugging or logging. 0 is the default verbose level which
  // means no action should be taken. As verboseLevel increases more detailed
  // information can be shown.
  private final int verboseLevel;

  // Bodo tracing level. Similar to verboseLevel, but for tracing related codegen
  private final int tracingLevel;

  // These Variables track the target table for merge into
  private @Nullable Variable mergeIntoTargetTable;
  private @Nullable TableScan mergeIntoTargetNode;
  // Extra arguments to pass to the write code for the fileList and Snapshot
  // id in the form of "argName1=varName1, argName2=varName2"
  private @Nullable String fileListAndSnapshotIdArgs;

  // Types of any dynamic parameters in the query.
  private final List<RelDataType> dynamicParamTypes;
  // Type of any named parameters in the query
  private final Map<String, RelDataType> namedParamTypeMap;

  // TODO(aneesh) consider moving this to the C++ code, or derive the
  // chunksize for writing from the allocated budget.
  // Writes only use a constant amount of memory of 256MB. We multiply
  // that by 1.5 to allow for some wiggle room.
  private final int SNOWFLAKE_WRITE_MEMORY_ESTIMATE = ((int) (1.5 * 256 * 1024 * 1024));
  private BodoSqlTable table;

  private @Nullable TreeReverserDuplicateTracker reverseVisitor = null;

  @Override
  public @Nullable RelNode go(RelNode p) {
    reverseVisitor = new TreeReverserDuplicateTracker();
    reverseVisitor.go(p);
    return super.go(p);
  }

  @NotNull
  public HashMap<RelNode, List<RelNode>> getParentMappings() {
    return reverseVisitor.getReversedTree();
  }

  public BodoCodeGenVisitor(
      HashMap<String, String> loweredGlobalVariablesMap,
      String originalSQLQuery,
      RelDataTypeSystem typeSystem,
      int verboseLevel,
      int tracingLevel,
      int batchSize,
      List<RelDataType> dynamicParamTypes,
      Map<String, RelDataType> namedParamTypeMap,
      Map<Integer, Integer> idMapping,
      boolean hideOperatorIDs) {
    super();
    this.loweredGlobals = loweredGlobalVariablesMap;
    this.originalSQLQuery = originalSQLQuery;
    this.typeSystem = typeSystem;
    this.mergeIntoTargetTable = null;
    this.mergeIntoTargetNode = null;
    this.fileListAndSnapshotIdArgs = null;
    this.verboseLevel = verboseLevel;
    this.tracingLevel = tracingLevel;
    this.generatedCode = new Module.Builder();
    this.generatedCode.setHideOperatorIDs(hideOperatorIDs);
    this.streamingOptions = new StreamingOptions(batchSize);
    this.dynamicParamTypes = dynamicParamTypes;
    this.namedParamTypeMap = namedParamTypeMap;
    this.generatedCode.setIDMapping(idMapping);
  }

  /**
   * Generate the new dataframe variable name for step by step Bodo codegen
   *
   * @return variable
   */
  public Variable genDfVar() {
    return generatedCode.getSymbolTable().genDfVar();
  }

  /**
   * Generate the new table variable for step by step Bodo codegen
   *
   * @return variable
   */
  public Variable genTableVar() {
    return generatedCode.getSymbolTable().genTableVar();
  }

  /**
   * Generate the new Series variable for step by step Bodo codegen
   *
   * @return variable
   */
  public Variable genSeriesVar() {
    return generatedCode.getSymbolTable().genSeriesVar();
  }

  /**
   * Generate the new Series variable for step by step Bodo codegen
   *
   * @return variable
   */
  public Variable genArrayVar() {
    return generatedCode.getSymbolTable().genArrayVar();
  }

  /**
   * Generate a new index variable for step by step Bodo codegen
   *
   * @return variable
   */
  public Variable genIndexVar() {
    return generatedCode.getSymbolTable().genIndexVar();
  }

  /**
   * Generate a new iter variable for step by step Bodo codegen
   *
   * @return variable
   */
  public Variable genIterVar() {
    return generatedCode.getSymbolTable().genIterVar();
  }

  /**
   * Generate a new output control variable for step by step Bodo codegen
   *
   * @return variable
   */
  public Variable genOutputControlVar() {
    return generatedCode.getSymbolTable().genOutputControlVar();
  }

  /**
   * Generate a new input request variable for step by step Bodo codegen
   *
   * @return variable
   */
  public Variable genInputRequestVar() {
    return generatedCode.getSymbolTable().genInputRequestVar();
  }

  /**
   * Generate the new temporary variable for step by step Bodo codegen.
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
   * Generate the new temporary variable for step by step Bodo codegen.
   *
   * @return variable
   */
  public Variable genWriterVar() {
    return generatedCode.getSymbolTable().genWriterVar();
  }

  /**
   * generate a new variable for group by apply functions in agg.
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

  public BodoTZInfo genDefaultTZ() {
    return BodoTZInfo.getDefaultTZInfo(typeSystem);
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
   * Return the final code after step by step Bodo codegen. Coerces the final answer to a DataFrame
   * if it is a Table, lowering a new global in the process.
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
    if (node instanceof PandasTargetTableScan) {
      this.visitPandasTargetTableScan((PandasTargetTableScan) node);
    } else if (node instanceof PandasRel) {
      this.visitPandasRel((PandasRel) node);
    } else if (node instanceof BodoPhysicalJoin) {
      this.visitBodoJoin((BodoPhysicalJoin) node);
    } else if (node instanceof BodoPhysicalAggregate) {
      this.visitBodoAggregate((BodoPhysicalAggregate) node);
    } else if (node instanceof BodoPhysicalMinRowNumberFilter) {
      this.visitBodoMinRowNumberFilter((BodoPhysicalMinRowNumberFilter) node);
    } else if (node instanceof BodoPhysicalIntersect) {
      this.visitBodoIntersect((BodoPhysicalIntersect) node);
    } else if (node instanceof BodoPhysicalMinus) {
      this.visitBodoMinus((BodoPhysicalMinus) node);
    } else if (node instanceof BodoPhysicalValues) {
      this.visitBodoValues((BodoPhysicalValues) node);
    } else if (node instanceof BodoPhysicalTableModify) {
      this.visitBodoTableModify((BodoPhysicalTableModify) node);
    } else if (node instanceof BodoPhysicalTableCreate) {
      this.visitBodoTableCreate((BodoPhysicalTableCreate) node);
    } else if (node instanceof BodoPhysicalRowSample) {
      this.visitBodoRowSample((BodoPhysicalRowSample) node);
    } else if (node instanceof BodoPhysicalSample) {
      this.visitBodoSample((BodoPhysicalSample) node);
    } else if (node instanceof Correlate) {
      throw new BodoSQLCodegenException(
          "Internal Error: BodoSQL does not support Correlated Queries");
    } else if (node instanceof CombineStreamsExchange) {
      this.visitCombineStreamsExchange((CombineStreamsExchange) node);
    } else if (node instanceof SeparateStreamExchange) {
      this.visitSeparateStreamExchange((SeparateStreamExchange) node);
    } else if (node instanceof BodoPhysicalRel) {
      this.visitBodoPhysicalRel((BodoPhysicalRel) node);
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
    OperatorID operatorID = this.generatedCode.newOperatorID(node);
    // Generate the list we are accumulating into.
    Variable batchAccumulatorVariable = this.genBatchAccumulatorVar();
    StreamingPipelineFrame activePipeline = this.generatedCode.getCurrentStreamingPipeline();

    StreamingRelNodeTimer timer =
        StreamingRelNodeTimer.createStreamingTimer(
            operatorID,
            this.generatedCode,
            verboseLevel,
            tracingLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());

    // get memory estimate of input
    RelMetadataQuery mq = node.getCluster().getMetadataQuery();
    Double inputRows = mq.getRowCount(node.getInput(0));
    Double averageInputRowSize =
        Optional.ofNullable(mq.getAverageRowSize(node.getInput(0))).orElse(8.0);
    int memoryEstimate = Double.valueOf(Math.ceil(inputRows * averageInputRowSize)).intValue();
    timer.insertStateStartTimer(0);
    activePipeline.initializeStreamingState(
        operatorID,
        new Op.Assign(
            batchAccumulatorVariable,
            new Expr.Call("bodo.libs.table_builder.init_table_builder_state", operatorID.toExpr())),
        OperatorType.ACCUMULATE_TABLE,
        memoryEstimate);
    timer.insertStateEndTimer(0);

    // Append to the list at the end of the loop.
    timer.insertLoopOperationStartTimer(1);
    List<Expr> args = new ArrayList<>();
    args.add(batchAccumulatorVariable);
    args.add(inputTableVar);
    Op appendStatement =
        new Op.Stmt(new Expr.Call("bodo.libs.table_builder.table_builder_append", args));
    generatedCode.add(appendStatement);
    timer.updateRowCount(1, inputTableVar);
    timer.insertLoopOperationEndTimer(1);

    // Finally, concatenate the batches in the accumulator into a table to use in
    // regular code.
    timer.insertLoopOperationStartTimer(2, true);
    Variable accumulatedTable = genTableVar();
    Expr concatenatedTable =
        new Expr.Call(
            "bodo.libs.table_builder.table_builder_finalize", List.of(batchAccumulatorVariable));
    Op.Assign assign = new Op.Assign(accumulatedTable, concatenatedTable);
    activePipeline.deleteStreamingState(operatorID, assign);
    timer.insertLoopOperationEndTimer(2, true);

    // Pop the pipeline
    StreamingPipelineFrame finishedPipeline = generatedCode.endCurrentStreamingPipeline();
    // Append the pipeline
    generatedCode.add(new Op.StreamingPipeline(finishedPipeline));
    BodoEngineTable outEngineTable = new BodoEngineTable(accumulatedTable.emit(), node);
    tableGenStack.push(outEngineTable);
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
   * Generic visitor method for any RelNode that implements BodoPhysicalRel.
   *
   * <p>This method handles node caching, visiting inputs, passing those inputs to the node itself
   * to emit code into the Module.Builder, and generating timers when requested.
   *
   * <p>The resulting variable that is generated for this RelNode is placed on the tableGenStack.
   *
   * @param node the node to emit code for.
   */
  private void visitBodoPhysicalRel(BodoPhysicalRel node) {
    // Note: All timer handling is done in emit
    BodoEngineTable out = node.emit(new Implementor(node, this.getParentMappings()));
    if (out != null) {
      tableGenStack.push(out);
    } else {
      // Only write should not emit a table.
      if (!(node instanceof TableModify) && !(node instanceof TableCreate)) {
        throw new BodoSQLCodegenException(
            "Internal Error: BodoPhysicalRel node did not emit a table: " + node.getClass());
      }
    }
  }

  /**
   * Visitor method for logicalValue Nodes
   *
   * @param node RelNode to be visited
   */
  private void visitBodoValues(BodoPhysicalValues node) {
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
   * Visitor for Bodo Intersect node. Code generation for INTERSECT [ALL/DISTINCT] in SQL
   *
   * @param node LogicalIntersect node to be visited
   */
  private void visitBodoIntersect(BodoPhysicalIntersect node) {
    // We always assume intersect is between exactly two inputs
    if (node.getInputs().size() != 2) {
      throw new BodoSQLCodegenException(
          "Internal Error: Intersect should be between exactly two inputs");
    }
    BuildContext ctx = new BuildContext(node, genDefaultTZ(), this.getParentMappings());

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
  private void visitBodoMinus(BodoPhysicalMinus node) {
    // We always assume minus is between exactly two inputs
    if (node.getInputs().size() != 2) {
      throw new BodoSQLCodegenException(
          "Internal Error: Except should be between exactly two inputs");
    }
    BuildContext ctx = new BuildContext(node, genDefaultTZ(), this.getParentMappings());

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
   * Generate Code for Non-Streaming / Single Batch CREATE TABLE
   *
   * @param node Create Table Node that Code is Generated For
   * @param outputSchemaAsCatalog Catalog of Output Table
   * @param ifExists Action if Table Already Exists
   * @param createTableType Type of Table to Create
   */
  public void genSingleBatchTableCreate(
      BodoPhysicalTableCreate node,
      CatalogSchema outputSchemaAsCatalog,
      IfExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType) {
    BuildContext ctx = new BuildContext(node, genDefaultTZ(), this.getParentMappings());
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
      BodoPhysicalTableCreate node,
      CatalogSchema schema,
      IfExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType,
      Expr columnPrecisions) {
    // Generate Streaming Code in this case
    // Get or create current streaming pipeline
    StreamingPipelineFrame currentPipeline = this.generatedCode.getCurrentStreamingPipeline();
    OperatorID operatorID = this.generatedCode.newOperatorID(node);
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
            operatorID,
            this.generatedCode,
            this.verboseLevel,
            this.tracingLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());

    // Get column names for write calls
    List<String> colNames = node.getRowType().getFieldNames();
    List<Expr.StringLiteral> colNamesList = stringsToStringLiterals(colNames);
    Variable colNamesGlobal = lowerAsColNamesMetaType(new Expr.Tuple(colNamesList));

    // Generate write destination information.
    WriteTarget writeTarget =
        schema.getCreateTableWriteTarget(
            node.getTableName(), createTableType, ifExists, colNamesGlobal);

    // First, create the writer state before the loop
    timerInfo.insertStateStartTimer(0);
    Expr writeState =
        writeTarget.streamingCreateTableInit(this, operatorID, createTableType, node.getMeta());
    Variable writerVar = this.genWriterVar();
    currentPipeline.initializeStreamingState(
        operatorID,
        new Op.Assign(writerVar, writeState),
        OperatorType.SNOWFLAKE_WRITE,
        SNOWFLAKE_WRITE_MEMORY_ESTIMATE);
    timerInfo.insertStateEndTimer(0);

    // Second, append the Table to the writer
    timerInfo.insertLoopOperationStartTimer(1);
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
    timerInfo.insertLoopOperationEndTimer(1);

    // Lastly, end the loop
    timerInfo.terminateTimer();
    StreamingPipelineFrame finishedPipeline = this.generatedCode.endCurrentStreamingPipeline();
    this.generatedCode.add(new Op.StreamingPipeline(finishedPipeline));
    this.generatedCode.forceEndOperatorAtCurPipeline(operatorID, finishedPipeline);
    this.generatedCode.add(writeTarget.streamingCreateTableFinalize());
  }

  public void visitBodoTableCreate(BodoPhysicalTableCreate node) {
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
   * Visitor for BodoPhysicalTableModify, which is used to support certain SQL write operations.
   *
   * @param node BodoPhysicalTableModify node to be visited
   */
  public void visitBodoTableModify(BodoPhysicalTableModify node) {
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
                + node.getOperation());
    }
  }

  /**
   * Visitor for MERGE INTO operation for SQL write. Currently, it just returns the delta table, for
   * testing purposes.
   *
   * @param node
   */
  public void visitMergeInto(BodoPhysicalTableModify node) {
    assert node.getOperation() == TableModify.Operation.MERGE;
    BuildContext ctx = new BuildContext(node, genDefaultTZ(), this.getParentMappings());

    RelNode input = node.getInput();
    this.visit(input, 0, node);
    singleBatchTimer(
        node,
        () -> {
          BodoEngineTable deltaTableVar = tableGenStack.pop();

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
      BodoPhysicalTableModify node,
      BodoEngineTable inputTable,
      List<String> colNames,
      BodoSqlTable bodoSqlTable) {
    singleBatchTimer(
        node,
        () -> {
          BuildContext ctx = new BuildContext(node, genDefaultTZ(), this.getParentMappings());
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
      BodoPhysicalTableModify node,
      BodoEngineTable inputTable,
      List<String> colNames,
      BodoSqlTable bodoSqlTable) {
    // Generate Streaming Code in this case
    // Get or create current streaming pipeline
    StreamingPipelineFrame currentPipeline = this.generatedCode.getCurrentStreamingPipeline();
    OperatorID operatorID = this.generatedCode.newOperatorID(node);

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
            operatorID,
            this.generatedCode,
            this.verboseLevel,
            this.tracingLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());

    // First, create the writer state before the loop
    timerInfo.insertStateStartTimer(0);
    Expr writeInitCode = writeTarget.streamingInsertIntoInit(this, operatorID);
    Variable writerVar = this.genWriterVar();
    currentPipeline.initializeStreamingState(
        operatorID,
        new Op.Assign(writerVar, writeInitCode),
        OperatorType.SNOWFLAKE_WRITE,
        SNOWFLAKE_WRITE_MEMORY_ESTIMATE);
    timerInfo.insertStateEndTimer(0);

    // Second, append the Table to the writer
    timerInfo.insertLoopOperationStartTimer(1);
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
            new CreateTableMetadata());
    this.generatedCode.add(new Op.Assign(globalIsLast, writerAppendCall));
    currentPipeline.endSection(globalIsLast);
    timerInfo.insertLoopOperationEndTimer(1);

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
  public void visitInsertInto(BodoPhysicalTableModify node) {
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
  public void visitDelete(BodoPhysicalTableModify node) {
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
   * Visitor for Bodo Aggregate, support for Aggregations in SQL such as SUM, COUNT, MIN, MAX.
   *
   * @param node BatchingProperty node to be visited
   */
  public void visitBodoAggregate(BodoPhysicalAggregate node) {
    if (node.getTraitSet().contains(BatchingProperty.STREAMING)) {
      visitStreamingBodoAggregate(node);
    } else {
      visitSingleBatchedBodoAggregate(node);
    }
  }

  /**
   * Visitor for Aggregate without streaming.
   *
   * @param node aggregate node being visited
   */
  private void visitSingleBatchedBodoAggregate(BodoPhysicalAggregate node) {
    BuildContext ctx = new BuildContext(node, genDefaultTZ(), this.getParentMappings());
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
          tableGenStack.push(outTable);
        });
  }

  /**
   * Visitor for Aggregate with streaming.
   *
   * @param node aggregate node being visited
   */
  private void visitStreamingBodoAggregate(BodoPhysicalAggregate node) {

    // Visit the input node
    this.visit(node.getInput(), 0, node);
    Variable buildTable = tableGenStack.pop();

    // Create the state var.
    OperatorID operatorID = generatedCode.newOperatorID(node);
    StreamingRelNodeTimer timerInfo =
        StreamingRelNodeTimer.createStreamingTimer(
            operatorID,
            this.generatedCode,
            this.verboseLevel,
            this.tracingLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());

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
            operatorID.toExpr(),
            keyIndices,
            fnames,
            offset,
            cols);
    Op.Assign groupbyInit = new Op.Assign(groupbyStateVar, stateCall);
    // Fetch the streaming pipeline
    StreamingPipelineFrame inputPipeline = generatedCode.getCurrentStreamingPipeline();
    timerInfo.insertStateStartTimer(0);
    RelMetadataQuery mq = node.getCluster().getMetadataQuery();
    inputPipeline.initializeStreamingState(
        operatorID, groupbyInit, OperatorType.GROUPBY, node.estimateBuildMemory(mq));
    timerInfo.insertStateEndTimer(0);
    Variable batchExitCond = inputPipeline.getExitCond();
    Variable newExitCond = genGenericTempVar();
    inputPipeline.endSection(newExitCond);
    Variable inputRequest = genInputRequestVar();
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
    timerInfo.insertLoopOperationStartTimer(1);
    generatedCode.add(new Op.TupleAssign(List.of(newExitCond, inputRequest), batchCall));
    inputPipeline.addInputRequest(inputRequest);
    timerInfo.insertLoopOperationEndTimer(1);
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
    timerInfo.insertLoopOperationStartTimer(2);
    generatedCode.add(new Op.TupleAssign(List.of(outTable, newFlag), outputCall));
    BodoEngineTable intermediateTable = new BodoEngineTable(outTable.emit(), node);
    final BodoEngineTable table;
    if (filteredAggCallList.size() != node.getAggCallList().size()) {
      // Append any Literal data if it exists.
      final BuildContext ctx = new BuildContext(node, genDefaultTZ(), this.getParentMappings());
      table = concatenateLiteralAggValue(this.generatedCode, ctx, intermediateTable, node);
    } else {
      table = intermediateTable;
    }
    timerInfo.insertLoopOperationEndTimer(2);

    // Append the code to delete the state
    Op.Stmt deleteState =
        new Op.Stmt(
            new Expr.Call(
                "bodo.libs.stream_groupby.delete_groupby_state", List.of(groupbyStateVar)));
    outputPipeline.addTermination(deleteState);
    // Add the table to the stack
    tableGenStack.push(table);
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
  public void visitBodoMinRowNumberFilter(BodoPhysicalMinRowNumberFilter node) {
    if (node.getTraitSet().contains(BatchingProperty.STREAMING)) {
      visitStreamingBodoMinRowNumberFilter(node);
    } else {
      // If non-streaming, fall back to regular BodoPhysicalFilter codegen
      RelNode asProjectFilter = node.asBodoProjectFilter();
      this.visitBodoPhysicalRel((BodoPhysicalRel) asProjectFilter);
    }
  }

  /** Visitor for MinRowNumberFilter with streaming. */
  private void visitStreamingBodoMinRowNumberFilter(BodoPhysicalMinRowNumberFilter node) {

    // Visit the input node
    this.visit(node.getInput(), 0, node);
    Variable buildTable = tableGenStack.pop();

    // Create the state var.
    OperatorID operatorID = generatedCode.newOperatorID(node);
    StreamingRelNodeTimer timerInfo =
        StreamingRelNodeTimer.createStreamingTimer(
            operatorID,
            this.generatedCode,
            this.verboseLevel,
            this.tracingLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());

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

    // Generate the global variables for regular streaming group by
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
    timerInfo.insertStateStartTimer(0);
    RelMetadataQuery mq = node.getCluster().getMetadataQuery();

    // Insert the state initialization call
    List<Expr> positionalArgs = new ArrayList<>();
    positionalArgs.add(operatorID.toExpr());
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
    timerInfo.insertStateEndTimer(0);
    Variable batchExitCond = inputPipeline.getExitCond();
    Variable newExitCond = genGenericTempVar();
    inputPipeline.endSection(newExitCond);
    Variable inputRequest = genInputRequestVar();

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
    timerInfo.insertLoopOperationStartTimer(1);
    generatedCode.add(new Op.TupleAssign(List.of(newExitCond, inputRequest), batchCall));
    inputPipeline.addInputRequest(inputRequest);
    timerInfo.insertLoopOperationEndTimer(1);

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
    timerInfo.insertLoopOperationStartTimer(2);
    generatedCode.add(new Op.TupleAssign(List.of(outTable, newFlag), outputCall));
    BodoEngineTable table = new BodoEngineTable(outTable.emit(), node);
    timerInfo.insertLoopOperationEndTimer(2);

    // Append the code to delete the state
    Op.Stmt deleteState =
        new Op.Stmt(
            new Expr.Call(
                "bodo.libs.stream_groupby.delete_groupby_state", List.of(groupbyStateVar)));
    outputPipeline.addTermination(deleteState);
    // Add the table to the stack
    tableGenStack.push(table);
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
    return generateLiteralCode(node, this);
  }

  /**
   * Generic visitor for PandasRel nodes.
   *
   * @param node Node being visited
   */
  public void visitPandasRel(PandasRel node) {
    // Note: All timer handling is done in emit
    BodoEngineTable out = node.emit(new Implementor(node, this.getParentMappings()));
    tableGenStack.push(out);
  }

  /**
   * Visitor for Table Scan.
   *
   * @param node TableScan node being visited
   */
  public void visitPandasTargetTableScan(PandasTargetTableScan node) {
    singleBatchTimer(
        node,
        () -> {
          BodoSqlTable table;

          // TODO(jsternberg): The proper way to do this is to have the individual nodes
          // handle the code generation. Due to the way the code generation is
          // constructed,
          // we can't really do that so we're just going to hack around it for now to
          // avoid
          // a large refactor
          RelOptTableImpl relTable = (RelOptTableImpl) node.getTable();
          table = (BodoSqlTable) relTable.table();

          // IsTargetTableScan is always true, we check for this at the start of the
          // function.
          Expr readCode;
          Op readAssign;

          Variable readVar = this.genTableVar();
          // Add the table to cached values
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
                      readVar.emit(),
                      icebergFileListVarName,
                      icebergSnapshotIDName,
                      readCode.emit()));
          this.generatedCode.add(readAssign);
          mergeIntoTargetNode = node;
          mergeIntoTargetTable = readVar;
          BodoEngineTable outTable;
          BuildContext ctx = new BuildContext(node, genDefaultTZ(), this.getParentMappings());

          Expr castExpr = table.generateReadCastCode(readVar);
          if (!castExpr.equals(readVar)) {
            // Generate a new outVar to avoid shadowing
            Variable outVar = this.genDfVar();
            this.generatedCode.add(new Op.Assign(outVar, castExpr));
            outTable = ctx.convertDfToTable(outVar, node);
          } else {
            outTable = new BodoEngineTable(readVar.emit(), node);
          }
          tableGenStack.push(outTable);
        });
  }

  /**
   * Visitor for Join: Supports JOIN clause in SQL.
   *
   * @param node join node being visited
   */
  private void visitBodoJoin(BodoPhysicalJoin node) {
    if (node.getTraitSet().contains(BatchingProperty.STREAMING)) {
      visitStreamingBodoJoin(node);
    } else {
      visitBatchedBodoJoin(node);
    }
  }

  /**
   * Visitor for Join without streaming.
   *
   * @param node join node being visited
   */
  private void visitBatchedBodoJoin(BodoPhysicalJoin node) {
    BuildContext ctx = new BuildContext(node, genDefaultTZ(), this.getParentMappings());
    /* get left/right tables */

    List<String> outputColNames = node.getRowType().getFieldNames();
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
          tableGenStack.push(outTable);
        });
  }

  /**
   * Visitor for BodoPhysicalJoin when it is a streaming join. Both nested loop and hash join use
   * the same API calls and the decision is made based on the values of the arguments.
   */
  private void visitStreamingBodoJoin(BodoPhysicalJoin node) {
    // TODO: Move to a wrapper function to avoid the timerInfo calls.
    // This requires more information about the high level design of the streaming
    // operators since there are several parts (e.g. state, multiple loop sections,
    // etc.)
    // At this time it seems like it would be too much work to have a clean
    // interface.
    // There may be a need to pass in several lambdas, so other changes may be
    // needed to avoid
    // constant rewriting.
    OperatorID operatorID = this.generatedCode.newOperatorID(node);
    StreamingRelNodeTimer timerInfo =
        StreamingRelNodeTimer.createStreamingTimer(
            operatorID,
            this.generatedCode,
            this.verboseLevel,
            this.tracingLevel,
            node.operationDescriptor(),
            node.loggingTitle(),
            node.nodeDetails(),
            node.getTimerType());
    StateVariable joinStateVar = visitStreamingBodoJoinBatch(node, timerInfo, operatorID);
    visitStreamingBodoJoinProbe(node, joinStateVar, timerInfo, operatorID);
    timerInfo.terminateTimer();
  }

  private StateVariable visitStreamingBodoJoinState(
      BodoPhysicalJoin node, StreamingRelNodeTimer timerInfo, OperatorID operatorID) {
    // Extract the Hash Join information
    timerInfo.insertStateStartTimer(0);
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
    Expr.BooleanLiteral forceBroadcast = new Expr.BooleanLiteral(node.getBroadcastBuildSide());
    Expr.Call stateCall =
        new Expr.Call(
            "bodo.libs.stream_join.init_join_state",
            List.of(
                operatorID.toExpr(),
                keyIndices.right,
                keyIndices.left,
                buildNamesGlobal,
                probeNamesGlobal,
                isRightOuter,
                isLeftOuter,
                forceBroadcast),
            namedArgs);
    Op.Assign joinInit = new Op.Assign(joinStateVar, stateCall);

    RelMetadataQuery mq = node.getCluster().getMetadataQuery();
    batchPipeline.initializeStreamingState(
        operatorID, joinInit, OperatorType.JOIN, node.estimateBuildMemory(mq));

    timerInfo.insertStateEndTimer(0);
    // Update the join state cache for runtime filters.
    generatedCode
        .getJoinStateCache()
        .setStreamingJoinStateInfo(
            node.getJoinFilterID(), joinStateVar, node.getOriginalJoinFilterKeyLocations());
    return joinStateVar;
  }

  private StateVariable visitStreamingBodoJoinBatch(
      BodoPhysicalJoin node, StreamingRelNodeTimer timerInfo, OperatorID operatorID) {
    // Visit the batch side
    this.visit(node.getRight(), 0, node);
    BodoEngineTable buildTable = tableGenStack.pop();
    StateVariable joinStateVar = visitStreamingBodoJoinState(node, timerInfo, operatorID);
    timerInfo.insertLoopOperationStartTimer(1);
    // Fetch the batch state
    StreamingPipelineFrame batchPipeline = generatedCode.getCurrentStreamingPipeline();
    Variable batchExitCond = batchPipeline.getExitCond();
    Variable newExitCond = genGenericTempVar();
    Variable inputRequest = genInputRequestVar();
    batchPipeline.endSection(newExitCond);
    Expr.Call batchCall =
        new Expr.Call(
            "bodo.libs.stream_join.join_build_consume_batch",
            List.of(joinStateVar, buildTable, batchExitCond));
    generatedCode.add(new Op.TupleAssign(List.of(newExitCond, inputRequest), batchCall));
    batchPipeline.addInputRequest(inputRequest);
    timerInfo.insertLoopOperationEndTimer(1);
    // Finalize and add the batch pipeline.
    generatedCode.add(new Op.StreamingPipeline(generatedCode.endCurrentStreamingPipeline()));
    return joinStateVar;
  }

  private void visitStreamingBodoJoinProbe(
      BodoPhysicalJoin node,
      StateVariable joinStateVar,
      StreamingRelNodeTimer timerInfo,
      OperatorID operatorID) {
    // Visit the probe side
    this.visit(node.getLeft(), 1, node);
    timerInfo.insertLoopOperationStartTimer(2);
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

    timerInfo.insertLoopOperationEndTimer(2);
    // Append the code to delete the state
    Op.Stmt deleteState =
        new Op.Stmt(
            new Expr.Call("bodo.libs.stream_join.delete_join_state", List.of(joinStateVar)));
    probePipeline.deleteStreamingState(operatorID, deleteState);
    // Add the table to the stack
    BodoEngineTable outEngineTable = new BodoEngineTable(outTable.emit(), node);
    tableGenStack.push(outEngineTable);
  }

  /**
   * Visitor for RowSample: Supports SAMPLE clause in SQL with a fixed number of rows.
   *
   * @param node rowSample node being visited
   */
  public void visitBodoRowSample(BodoPhysicalRowSample node) {
    // We always assume row sample has exactly one input
    assert node.getInputs().size() == 1;
    BuildContext ctx = new BuildContext(node, genDefaultTZ(), this.getParentMappings());

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
  public void visitBodoSample(BodoPhysicalSample node) {
    // We always assume sample has exactly one input
    assert node.getInputs().size() == 1;
    BuildContext ctx = new BuildContext(node, genDefaultTZ(), this.getParentMappings());

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

  public void generateDDLCode(SqlNode ddlNode, GenerateDDLTypes typeGenerator) {
    // Generate the output type.
    RelDataType outputType = typeGenerator.generateType(ddlNode);
    // The BodoSQL context is always in the function signature as bodo_sql_context.
    Variable bodosqlContext = new Variable("bodo_sql_context");
    // TODO: Eventually we want to remove this from the constructor and pass it in as a
    // parameter to this function.
    Expr.StringLiteral query = new Expr.StringLiteral(originalSQLQuery);
    List<Expr> columnTypes = new ArrayList<>();
    for (RelDataTypeField field : outputType.getFieldList()) {
      Expr typeString =
          sqlTypeToBodoArrayType(field.getType(), false, genDefaultTZ().getZoneExpr());
      columnTypes.add(typeString);
    }
    Expr.Tuple columnTypesTuple = new Expr.Tuple(columnTypes);
    Variable columnTypesGlobal = lowerAsMetaType(columnTypesTuple);
    Expr.Call call =
        new Expr.Call(
            "bodosql.ddl_ext.execute_ddl", List.of(bodosqlContext, query, columnTypesGlobal));
    Variable outVar = this.genDfVar();
    this.generatedCode.add(new Op.Assign(outVar, call));
    // Update for return.
    BodoEngineTable outTable = new BodoEngineTable(outVar.emit(), outputType);
    tableGenStack.add(outTable);
  }

  private void singleBatchTimer(TimerSupportedRel node, Runnable fn) {
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
  private BodoEngineTable singleBatchTimer(TimerSupportedRel node, Supplier<BodoEngineTable> fn) {
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

  private RexToBodoTranslator getRexTranslator(RelNode self, String inVar, RelNode input) {
    return getRexTranslator(self, new BodoEngineTable(inVar, input));
  }

  private RexToBodoTranslator getRexTranslator(RelNode self, BodoEngineTable input) {
    return getRexTranslator(input);
  }

  private RexToBodoTranslator getRexTranslator(BodoEngineTable input) {
    return new RexToBodoTranslator(
        this,
        this.generatedCode,
        this.typeSystem,
        input,
        this.dynamicParamTypes,
        this.namedParamTypeMap);
  }

  private RexToBodoTranslator getRexTranslator(
      BodoEngineTable input, List<? extends Expr> localRefs) {
    return new RexToBodoTranslator(
        this,
        this.generatedCode,
        this.typeSystem,
        input,
        this.dynamicParamTypes,
        this.namedParamTypeMap,
        localRefs);
  }

  private ArrayRexToBodoTranslator getArrayRexTranslator(BodoEngineTable input) {
    return new ArrayRexToBodoTranslator(
        this,
        this.generatedCode,
        this.typeSystem,
        input,
        this.dynamicParamTypes,
        this.namedParamTypeMap);
  }

  private ScalarRexToBodoTranslator getScalarRexTranslator() {
    return new ScalarRexToBodoTranslator(
        this, this.generatedCode, this.typeSystem, this.dynamicParamTypes, this.namedParamTypeMap);
  }

  private StreamingRexToBodoTranslator getStreamingRexTranslator(
      BodoEngineTable input, List<? extends Expr> localRefs, @NotNull StateVariable stateVar) {
    return new StreamingRexToBodoTranslator(
        this,
        this.generatedCode,
        this.typeSystem,
        input,
        this.dynamicParamTypes,
        this.namedParamTypeMap,
        localRefs,
        stateVar);
  }

  private class Implementor implements BodoPhysicalRel.Implementor {
    private final @NotNull TimerSupportedRel node;
    private final @NotNull HashMap<RelNode, List<RelNode>> parentMappings;

    public Implementor(
        @NotNull TimerSupportedRel node, @NotNull HashMap<RelNode, List<RelNode>> parentMappings) {
      this.node = node;
      this.parentMappings = parentMappings;
    }

    @NotNull
    @Override
    public BodoEngineTable build(
        @NotNull final Function1<? super BodoPhysicalRel.BuildContext, BodoEngineTable> fn) {
      return singleBatchTimer(
          node,
          () ->
              fn.invoke(new BodoCodeGenVisitor.BuildContext(node, genDefaultTZ(), parentMappings)));
    }

    @Nullable
    @Override
    public BodoEngineTable buildStreaming(@NotNull OperatorEmission operatorEmission) {
      OperatorID operatorID = generatedCode.newOperatorID(node);
      BodoCodeGenVisitor.BuildContext buildContext =
          new BodoCodeGenVisitor.BuildContext(node, operatorID, genDefaultTZ(), parentMappings);
      // Must init the first pipeline before we can generate any timers.
      BodoEngineTable table = operatorEmission.initFirstPipeline(buildContext);
      StreamingRelNodeTimer timerInfo =
          StreamingRelNodeTimer.createStreamingTimer(
              operatorID,
              generatedCode,
              verboseLevel,
              tracingLevel,
              node.operationDescriptor(),
              node.loggingTitle(),
              node.nodeDetails(),
              node.getTimerType());
      BodoEngineTable output = operatorEmission.emitOperator(buildContext, timerInfo, table);
      timerInfo.terminateTimer();
      return output;
    }

    @NotNull
    @Override
    public RelationalOperatorCache getRelationalOperatorCache() {
      return relationalOperatorCache;
    }
  }

  private class BuildContext implements BodoPhysicalRel.BuildContext {
    private final @NotNull TimerSupportedRel node;
    private final OperatorID operatorID;

    private final BodoTZInfo defaultTz;

    private final HashMap<RelNode, List<RelNode>> parentMappings;

    public BuildContext(
        @NotNull TimerSupportedRel node,
        OperatorID operatorID,
        BodoTZInfo defaultTz,
        HashMap<RelNode, List<RelNode>> parentMappings) {
      this.node = node;
      this.operatorID = operatorID;
      this.defaultTz = defaultTz;
      this.parentMappings = parentMappings;
    }

    public BuildContext(
        @NotNull TimerSupportedRel node,
        BodoTZInfo defaultTz,
        HashMap<RelNode, List<RelNode>> parentMappings) {
      this(node, null, defaultTz, parentMappings);
    }

    @Override
    public OperatorID operatorID() {
      return operatorID;
    }

    @NotNull
    @Override
    public BodoEngineTable visitChild(@NotNull final RelNode input, final int ordinal) {
      visit(input, ordinal, node);
      return tableGenStack.pop();
    }

    @NotNull
    @Override
    public Module.Builder builder() {
      return generatedCode;
    }

    @NotNull
    @Override
    public HashMap<RelNode, List<RelNode>> fetchParentMappings() {
      return parentMappings;
    }

    @NotNull
    @Override
    public Variable lowerAsGlobal(@NotNull final Expr expression) {
      return BodoCodeGenVisitor.this.lowerAsGlobal(expression);
    }

    @NotNull
    @Override
    public Variable lowerAsMetaType(@NotNull final Expr expression) {
      return BodoCodeGenVisitor.this.lowerAsMetaType(expression);
    }

    @NotNull
    @Override
    public Variable lowerAsColNamesMetaType(@NotNull final Expr expression) {
      return BodoCodeGenVisitor.this.lowerAsColNamesMetaType(expression);
    }

    @NotNull
    @Override
    public RexToBodoTranslator rexTranslator(@NotNull final BodoEngineTable input) {
      return getRexTranslator(input);
    }

    @NotNull
    @Override
    public RexToBodoTranslator rexTranslator(
        @NotNull final BodoEngineTable input, @NotNull final List<? extends Expr> localRefs) {
      return getRexTranslator(input, localRefs);
    }

    @NotNull
    @Override
    public ArrayRexToBodoTranslator arrayRexTranslator(@NotNull final BodoEngineTable input) {
      return getArrayRexTranslator(input);
    }

    @NotNull
    @Override
    public ScalarRexToBodoTranslator scalarRexTranslator() {
      return getScalarRexTranslator();
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
    public StreamingRexToBodoTranslator streamingRexTranslator(
        @NotNull BodoEngineTable input,
        @NotNull List<? extends Expr> localRefs,
        @NotNull StateVariable stateVar) {
      return getStreamingRexTranslator(input, localRefs, stateVar);
    }

    @NotNull
    @Override
    public BodoTZInfo getDefaultTZ() {
      return this.defaultTz;
    }

    @Override
    public void startPipeline() {
      Variable exitCond = builder().getSymbolTable().genFinishedStreamingFlag();
      Variable iterVariable = builder().getSymbolTable().genIterVar();
      builder().startStreamingPipelineFrame(exitCond, iterVariable);
    }

    @Override
    public void endPipeline() {
      builder().endCurrentStreamingPipeline();
    }
  }
}
