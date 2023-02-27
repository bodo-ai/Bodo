package com.bodosql.calcite.application;

import static com.bodosql.calcite.application.BodoSQLCodeGen.AggCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.BinOpCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CastCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.CondOpCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ConversionCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DateAddCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DateDiffCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ExtractCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.FilterCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.JoinCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.JsonCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.LikeCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.LiteralCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.LogicalValuesCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.NumericCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.PostfixOpCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.PrefixOpCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.ProjectCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.RegexpCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SetOpCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SinceEpochFnCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SortCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.StringFnCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.TimestampDiffCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.TrigCodeGen.*;
import static com.bodosql.calcite.application.BodoSQLCodeGen.WindowAggCodeGen.*;
import static com.bodosql.calcite.application.JoinCondVisitor.*;
import static com.bodosql.calcite.application.Utils.AggHelpers.*;
import static com.bodosql.calcite.application.Utils.Utils.*;
import static com.bodosql.calcite.application.Utils.Utils.getBodoIndent;

import com.bodosql.calcite.application.BodoSQLCodeGen.WindowAggCodeGen;
import com.bodosql.calcite.application.BodoSQLCodeGen.WindowedAggregationArgument;
import com.bodosql.calcite.application.Utils.BodoCtx;
import com.bodosql.calcite.ir.*;
import com.bodosql.calcite.ir.Module;
import com.bodosql.calcite.table.BodoSqlTable;
import com.bodosql.calcite.table.LocalTableImpl;
import com.google.common.collect.Range;
import java.util.*;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.prepare.RelOptTableImpl;
import org.apache.calcite.rel.RelFieldCollation;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelVisitor;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.TableModify;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.logical.*;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.rex.*;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.fun.*;
import org.apache.calcite.sql.type.*;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Pair;
import org.apache.calcite.util.Sarg;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Visitor class for parsed SQL nodes to generate Pandas code from SQL code. */
public class PandasCodeGenVisitor extends RelVisitor {
  /* Stack of generated variables df1, df2 , etc. */
  private final Stack<String> varGenStack = new Stack<>();
  /* Stack of output column names for every node */
  private final Stack<List<String>> columnNamesStack = new Stack<>();
  /* Reserved column name for generating dummy columns. */
  // TODO: Add this to the docs as banned
  private Module.Builder generatedCode = new Module.Builder();
  private int dfVarId = 1;
  private int colVarId = 1;
  private int groupByApplyFnId = 1;
  private int globalVarId = 1;

  // Note that a given query can only have one MERGE INTO statement. Therefore,
  // we can statically define the variable names we'll use for the iceberg file list and snapshot
  // id,
  // since we'll only be using these variables once per query
  private static final String icebergFileListVarName = "__bodo_Iceberg_file_list";
  private static final String icebergSnapshotIDName = "__bodo_Iceberg_snapshot_id";

  private static final String ROW_ID_COL_NAME = "_bodo_row_id";
  private static final String MERGE_ACTION_ENUM_COL_NAME = "_merge_into_change";

  // Mapping from a unique key per node to exprTypes
  private final HashMap<String, BodoSQLExprType.ExprType> exprTypesMap;

  // Mapping from String Key of Search Nodes to the RexNodes expanded
  // TODO: Replace this code with something more with an actual
  // update to the plan.
  // Ideally we can use RexRules when they are available
  // https://issues.apache.org/jira/browse/CALCITE-4559
  private final HashMap<String, RexNode> searchMap;

  // Map of RelNode ID -> <DataFrame variable name, Column Names>
  // Because the logical plan is a tree, Nodes that are at the bottom of
  // the tree must be repeated, even if they are identical. However, when
  // calcite produces identical nodes, it gives them the same node ID. As a
  // result, when finding nodes we wish to cache, we log variable names in this
  // map and load them inplace of segments of generated code.
  // This is currently only implemented for a subset of nodes.
  private final HashMap<Integer, Pair<String, List<String>>> varCache;

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
  private @Nullable String targetTableDf;
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
    this.varCache = new HashMap<Integer, Pair<String, List<String>>>();
    this.loweredGlobals = loweredGlobalVariablesMap;
    this.originalSQLQuery = originalSQLQuery;
    this.typeSystem = typeSystem;
    this.debuggingDeltaTable = debuggingDeltaTable;
    this.targetTableDf = null;
    this.fileListAndSnapshotIdArgs = null;
    this.verboseLevel = verboseLevel;
  }

  /**
   * Generate the new dataframe variable name for step by step pandas codegen
   *
   * @return variable name
   */
  private String genDfVar() {
    return "df" + this.dfVarId++;
  }

  /**
   * Generate the new table variable name for step by step pandas codegen
   *
   * @return variable name
   */
  public String genTableVar() {
    return "T" + this.dfVarId++;
  }

  /**
   * Generate the new Series variable name for step by step pandas codegen
   *
   * @return variable name
   */
  public String genSeriesVar() {
    return "S" + this.dfVarId++;
  }

  /**
   * Generate the new temporary variable name for step by step pandas codegen.
   *
   * @return variable name
   */
  public String genGenericTempVar() {
    return "_temp" + this.dfVarId++;
  }

  /**
   * Generate the new variable name for a precomputed windowed aggregation column.
   *
   * @return variable name
   */
  private String genTempColumnVar() {
    return "__bodo_generated_column__" + this.colVarId++;
  }

  /**
   * Generate the new variable name for a precomputed windowed aggregation dataframe.
   *
   * @return variable name
   */
  private String genWindowedAggDfName() {
    return "__bodo_windowfn_generated_df_" + this.dfVarId++;
  }

  /**
   * generate a new function name for nested aggregation functions.
   *
   * @return variable name
   */
  private String genWindowedAggFnName() {
    return getDummyColNameBase() + "_sql_windowed_apply_fn_" + this.groupByApplyFnId++;
  }

  /**
   * generate a new function name for groupby apply in agg.
   *
   * @return variable name
   */
  private String genGroupbyApplyAggFnName() {
    return getDummyColNameBase() + "_sql_groupby_apply_fn_" + this.groupByApplyFnId++;
  }

  /**
   * Modifies the codegen such that the specified expression will be lowered into the func_text as a
   * global. This is currently only used for lowering metaDataType's and array types.
   *
   * @return string variable name, which will be lowered as a global with a value equal to the
   *     supplied expression
   */
  public String lowerAsGlobal(String expression) {
    String global_var_name = "global_" + this.globalVarId++;
    this.loweredGlobals.put(global_var_name, expression);
    return global_var_name;
  }

  /**
   * pass expression as a MetaType global to the generated output function
   *
   * @param expression to pass, e.g. (2, 3, 1)
   * @return variable name for the global
   */
  public String lowerAsMetaType(String expression) {
    return lowerAsGlobal("MetaType(" + expression + ")");
  }

  /**
   * Modifies the codegen such that the specified expression will be lowered into the func_text as a
   * ColNameMetaType global.
   *
   * @return string variable name, which will conti
   */
  public String lowerAsColNamesMetaType(String expression) {
    return lowerAsGlobal("ColNamesMetaType(" + expression + ")");
  }

  /**
   * return the final code after step by step pandas codegen
   *
   * @return generated code
   */
  public String getGeneratedCode() {
    // If the stack is size 0 we don't return a DataFrame (e.g. to_sql)
    if (this.varGenStack.size() > 0) {
      this.generatedCode.append(String.format("  return %s\n", this.varGenStack.pop()));
    }

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
      this.visitTableScan((TableScan) node, !(parent instanceof LogicalFilter));
    } else if (node instanceof Join) {
      this.visitJoin((Join) node);
    } else if (node instanceof LogicalSort) {
      this.visitLogicalSort((LogicalSort) node);
    } else if (node instanceof LogicalProject) {
      this.visitLogicalProject((LogicalProject) node);
    } else if (node instanceof LogicalAggregate) {
      this.visitLogicalAggregate((LogicalAggregate) node);
    } else if (node instanceof LogicalUnion) {
      this.visitLogicalUnion((LogicalUnion) node);
    } else if (node instanceof LogicalIntersect) {
      this.visitLogicalIntersect((LogicalIntersect) node);
    } else if (node instanceof LogicalMinus) {
      this.visitLogicalMinus((LogicalMinus) node);
    } else if (node instanceof LogicalFilter) {
      this.visitLogicalFilter((LogicalFilter) node);
    } else if (node instanceof LogicalValues) {
      this.visitLogicalValues((LogicalValues) node);
    } else if (node instanceof LogicalTableModify) {
      this.visitLogicalTableModify((LogicalTableModify) node);
    } else if (node instanceof LogicalCorrelate) {
      throw new BodoSQLCodegenException(
          "Internal Error: BodoSQL does not support Correlated Queries");
    } else {
      throw new BodoSQLCodegenException(
          "Internal Error: Encountered Unsupported Calcite Node " + node.getClass().toString());
    }
  }

  /**
   * Visitor method for logicalValue Nodes
   *
   * @param node RelNode to be visited
   */
  private void visitLogicalValues(LogicalValues node) {
    String outVar = this.genDfVar();
    List<String> columnNames = node.getRowType().getFieldNames();
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
    this.generatedCode.append(
        generateLogicalValuesCode(outVar, exprCodes, node.getRowType(), this));
    this.columnNamesStack.push(columnNames);
    this.varGenStack.push(outVar);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor method for logicalFilter nodes to support HAVING clause
   *
   * @param node RelNode to be visited
   */
  private void visitLogicalFilter(LogicalFilter node) {
    String outVar = genDfVar();
    int nodeId = node.getId();
    List<String> columnNames;
    if (this.isNodeCached(node)) {
      // If the node is in the cache load
      Pair<String, List<String>> cacheInfo = this.varCache.get(nodeId);
      this.generatedCode.append(String.format("  %s = %s\n", outVar, cacheInfo.getKey()));
      // Push the column names
      columnNames = cacheInfo.getValue();
    } else {
      RelNode input = node.getInput();
      this.visit(input, 0, node);
      this.genRelnodeTimerStart(node);
      String inVar = varGenStack.pop();
      columnNames = columnNamesStack.pop();
      // The hashset is unused because the result should always be empty.
      // We assume filter is never within an apply.
      RexNodeVisitorInfo filterPair =
          visitRexNode(node.getCondition(), columnNames, node.getId(), inVar, false, new BodoCtx());
      this.generatedCode.append(
          generateFilterCode(
              inVar,
              outVar,
              filterPair.getExprCode(),
              exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(node.getCondition(), nodeId))));
      this.varCache.put(nodeId, new Pair<>(outVar, columnNames));
      this.genRelnodeTimerStop(node);
    }
    this.columnNamesStack.push(columnNames);
    varGenStack.push(outVar);
  }

  /**
   * Visitor for Logical Union node. Code generation for UNION and UNION ALL clauses in SQL
   *
   * @param node LogicalUnion node to be visited
   */
  private void visitLogicalUnion(LogicalUnion node) {
    List<String> childExprs = new ArrayList<>();
    List<List<String>> childExprsColumns = new ArrayList<>();
    // Visit all of the inputs
    for (int i = 0; i < node.getInputs().size(); i++) {
      this.visit(node.getInputs().get(i), i, node);
      childExprs.add(varGenStack.pop());
      childExprsColumns.add(columnNamesStack.pop());
    }
    this.genRelnodeTimerStart(node);
    String outVar = this.genDfVar();
    List<String> columnNames = node.getRowType().getFieldNames();
    this.generatedCode.append(generateUnionCode(outVar, columnNames, childExprs, node.all, this));
    varGenStack.push(outVar);
    columnNamesStack.push(columnNames);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor for Logical Intersect node. Code generation for Intersect clause in SQL
   *
   * @param node LogicalIntersect node to be visited
   */
  private void visitLogicalIntersect(LogicalIntersect node) {
    // We always assume intersect is between exactly two inputs
    assert node.getInputs().size() == 2;

    // Visit the two inputs
    this.visit(node.getInputs().get(0), 0, node);
    String lhsExpr = varGenStack.pop();
    List<String> lhsColNames = columnNamesStack.pop();

    this.visit(node.getInputs().get(1), 1, node);
    String rhsExpr = varGenStack.pop();
    List<String> rhsColNames = columnNamesStack.pop();

    String outVar = this.genDfVar();
    List<String> colNames = node.getRowType().getFieldNames();
    this.genRelnodeTimerStart(node);

    this.generatedCode.append(
        generateIntersectCode(
            outVar, lhsExpr, lhsColNames, rhsExpr, rhsColNames, colNames, node.all));

    varGenStack.push(outVar);
    columnNamesStack.push(colNames);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor for Logical Minus node, equivalent to Except clause in SQL
   *
   * @param node LogicalMinus node to be visited
   */
  private void visitLogicalMinus(LogicalMinus node) {
    // I'm making the assumption that the minus node always has exactly two inputs
    assert node.getInputs().size() == 2;

    // Visit the two inputs
    this.visit(node.getInputs().get(0), 0, node);
    String lhsExpr = varGenStack.pop();
    List<String> lhsColNames = columnNamesStack.pop();

    this.visit(node.getInputs().get(1), 1, node);
    String rhsExpr = varGenStack.pop();
    List<String> rhsColNames = columnNamesStack.pop();

    assert lhsColNames.size() == rhsColNames.size();
    assert lhsColNames.size() > 0 && rhsColNames.size() > 0;

    String outVar = this.genDfVar();
    String throwAwayVar = this.genDfVar();
    List<String> colNames = new ArrayList<>();
    this.genRelnodeTimerStart(node);

    this.generatedCode.append(
        generateExceptCode(
            outVar, throwAwayVar, lhsExpr, lhsColNames, rhsExpr, rhsColNames, colNames));

    varGenStack.push(outVar);
    columnNamesStack.push(colNames);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor for Logical Sort node. Code generation for ORDER BY clauses in SQL
   *
   * @param node Logical Sort node to be visited
   */
  public void visitLogicalSort(LogicalSort node) {
    RelNode input = node.getInput();
    this.visit(input, 0, node);
    this.genRelnodeTimerStart(node);
    String inVar = varGenStack.pop();
    List<String> colNames = columnNamesStack.pop();
    /* handle case for queries with "ORDER BY" clause */
    List<RelFieldCollation> sortOrders = node.getCollation().getFieldCollations();
    String outVar = this.genDfVar();
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
      RexNodeVisitorInfo childExpr =
          visitRexNode(fetch, colNames, node.getId(), inVar, false, new BodoCtx());
      limitStr = childExpr.getExprCode();
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
      RexNodeVisitorInfo childExpr =
          visitRexNode(offset, colNames, node.getId(), inVar, false, new BodoCtx());
      offsetStr = childExpr.getExprCode();
    }
    this.generatedCode.append(
        generateSortCode(inVar, outVar, colNames, sortOrders, limitStr, offsetStr));
    varGenStack.push(outVar);
    columnNamesStack.push(colNames);
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor for LogicalProject node.
   *
   * @param node LogicalProject node to be visited
   */
  public void visitLogicalProject(LogicalProject node) {
    String projectOutVar = this.genDfVar();
    // Output column names selected.
    List<String> outputColumns = new ArrayList();
    int nodeId = node.getId();
    String outputCode = "";
    boolean needsTimerStop = false;
    if (this.isNodeCached(node)) {
      Pair<String, List<String>> cacheInfo = this.varCache.get(nodeId);
      outputCode = String.format("  %s = %s\n", projectOutVar, cacheInfo.getKey());
      // Push the column names
      outputColumns = cacheInfo.getValue();
    } else {
      this.visit(node.getInput(), 0, node);
      this.genRelnodeTimerStart(node);
      needsTimerStop = true;
      String projectInVar = varGenStack.pop();
      List<String> colNames = columnNamesStack.pop();
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
        outputCode =
            generateLocCode(
                projectInVar, projectOutVar, node.getProjects(), colNames, outputColumns);
        if (!renameMap.isEmpty()) {
          outputCode = outputCode + generateRenameCode(projectOutVar, projectOutVar, renameMap);
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
        // Currently we only cache projections if it has a loc. This is because the
        // projection can cause functions that trigger data generation (e.g random
        // values). In the future we should optimize this code to directly check for
        // random data generation and cache in all other cases.
        this.varCache.put(nodeId, new Pair<>(projectOutVar, outputColumns));
      } else {
        List<RelDataTypeField> fields = node.getRowType().getFieldList();
        List<RelDataType> sqlTypes = new ArrayList<>();
        for (int i = 0; i < node.getNamedProjects().size(); i++) {
          Pair<RexNode, String> named_r = node.getNamedProjects().get(i);
          RexNode r = named_r.getKey();
          outputColumns.add(named_r.getValue());
          exprTypes.add(exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(r, nodeId)));
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
            childExpr = visitRexNode(r, colNames, nodeId, projectInVar, false, new BodoCtx());
            childExpr = new RexNodeVisitorInfo(childExpr.getExprCode());
          }
          childExprs.add(childExpr);
        }
        // handle windowed Aggregations
        List<RexNodeVisitorInfo> windowAggInfo =
            visitAggOverOp(windowAggList, colNames, nodeId, projectInVar, false, new BodoCtx());
        // check that these all have the same len
        assert windowAggInfo.size() == windowAggList.size()
            && windowAggInfo.size() == idx_list.size();
        for (int i = 0; i < windowAggList.size(); i++) {
          int origIdx = idx_list.get(i);
          // grab the original alias from the value stored value in childExprs
          RexNodeVisitorInfo realChildExpr =
              new RexNodeVisitorInfo(windowAggInfo.get(i).getExprCode());
          // overwrite the dummy value that we stored earlier in childExpr
          childExprs.set(origIdx, realChildExpr);
        }

        outputCode =
            generateProjectCode(
                projectInVar,
                projectOutVar,
                outputColumns,
                childExprs,
                exprTypes,
                sqlTypes,
                this,
                colNames.size());
      }
    }
    this.generatedCode.append(outputCode);
    this.columnNamesStack.push(outputColumns);
    if (needsTimerStop) {
      this.genRelnodeTimerStop(node);
    }
    varGenStack.push(projectOutVar);
  }

  /**
   * Visitor for LogicalTableModify, which is used to support certain SQL write operations.
   *
   * @param node LogicalTableModify node to be visited
   */
  public void visitLogicalTableModify(LogicalTableModify node) {
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
  public void visitMergeInto(LogicalTableModify node) {
    assert node.getOperation() == TableModify.Operation.MERGE;

    this.visit(node.getInput(0), 0, node);
    this.genRelnodeTimerStart(node);
    String deltaDfVar = this.varGenStack.pop();
    List<String> currentDeltaDfColNames = this.columnNamesStack.pop();

    if (this.debuggingDeltaTable) {
      // If this environment variable is set, we're only testing the generation of the delta table.
      // Just return the delta table.
      // We drop no-ops from the delta table, as a few Calcite Optimizations can result in their
      // being removed from the table, and their presence/lack thereof shouldn't impact anything in
      // the
      // final implementation, but it can cause issues when testing the delta table
      this.generatedCode
          .append(getBodoIndent())
          .append(deltaDfVar)
          .append(" = ")
          .append(deltaDfVar)
          .append(".dropna(subset=[")
          .append(makeQuoted(currentDeltaDfColNames.get(currentDeltaDfColNames.size() - 1)))
          .append("])\n");
      this.generatedCode.append(getBodoIndent()).append("return " + deltaDfVar);
    } else {
      // Assert that we've encountered a LogicalTargetTableScan in the codegen, and
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

      StringBuilder outputCode = new StringBuilder();
      String writebackDf = genDfVar();

      outputCode.append(
          handleRename(deltaDfVar, currentDeltaDfColNames, deltaTableExpectedColumnNames));
      outputCode
          .append(getBodoIndent())
          .append(writebackDf)
          .append(" = bodosql.libs.iceberg_merge_into.do_delta_merge_with_target(")
          .append(this.targetTableDf)
          .append(", ")
          .append(deltaDfVar)
          .append(")\n");

      // TODO: this can just be cast, since we handled rename
      outputCode.append(
          handleCastAndRenameBeforeWrite(writebackDf, targetTableFinalColumnNames, bodoSqlTable));
      outputCode
          .append(getBodoIndent())
          .append(bodoSqlTable.generateWriteCode(writebackDf, this.fileListAndSnapshotIdArgs))
          .append("\n");
      this.generatedCode.append(outputCode);
    }
    this.genRelnodeTimerStop(node);
  }

  /**
   * Visitor for Insert INTO operation for SQL write.
   *
   * @param node LogicalTableModify node to be visited
   */
  public void visitInsertInto(LogicalTableModify node) {
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
    List<RelNode> inputs = node.getInputs();
    this.visit(node.getInput(), 0, node);
    this.genRelnodeTimerStart(node);
    // Generate the to_sql code
    StringBuilder outputCode = new StringBuilder();
    List<String> colNames = this.columnNamesStack.pop();
    String outVar = this.varGenStack.pop();
    RelOptTableImpl relOptTable = (RelOptTableImpl) node.getTable();
    BodoSqlTable bodoSqlTable = (BodoSqlTable) relOptTable.table();
    if (!bodoSqlTable.isWriteable()) {
      throw new BodoSQLCodegenException(
          "Insert Into is only supported with table destinations provided via the Snowflake"
              + "catalog or the SQL TablePath API");
    }

    outputCode.append(handleCastAndRenameBeforeWrite(outVar, colNames, bodoSqlTable));
    outputCode.append(getBodoIndent()).append(bodoSqlTable.generateWriteCode(outVar)).append("\n");
    this.generatedCode.append(outputCode);
    this.genRelnodeTimerStop(node);
  }

  public String handleCastAndRenameBeforeWrite(
      String outVar, List<String> colNames, BodoSqlTable bodoSqlTable) {
    StringBuilder outputCode = new StringBuilder();
    String castExpr = bodoSqlTable.generateWriteCastCode(outVar);
    if (castExpr != "") {
      outputCode.append(getBodoIndent()).append(outVar).append(" = ").append(castExpr).append("\n");
    }
    // Update column names to the write names.
    outputCode.append(handleRename(outVar, colNames, bodoSqlTable.getWriteColumnNames()));
    return outputCode.toString();
  }

  public String handleRename(String outVar, List<String> oldColNames, List<String> newColNames) {
    assert oldColNames.size() == newColNames.size();
    StringBuilder outputCode = new StringBuilder();
    boolean hasRename = false;
    for (int i = 0; i < newColNames.size(); i++) {
      if (!oldColNames.get(i).equals(newColNames.get(i))) {
        if (!hasRename) {
          // Only generate the rename if at least 1 column needs renaming to avoid any empty
          // dictionary issues.
          outputCode
              .append(getBodoIndent())
              .append(outVar)
              .append(" = ")
              .append(outVar)
              .append(".rename(columns={");
          hasRename = true;
        }
        outputCode.append(makeQuoted(oldColNames.get(i)));
        outputCode.append(" : ");
        outputCode.append(makeQuoted(newColNames.get(i)));
        outputCode.append(", ");
      }
    }
    if (hasRename) {
      outputCode.append("}, copy=False)\n");
    }
    return outputCode.toString();
  }

  /**
   * Visitor for SQL Delete Operation with a remote database. We currently only support delete via
   * our Snowflake Catalog.
   *
   * <p>Note: This operation DOES NOT support caching as it has side effects.
   */
  public void visitDelete(LogicalTableModify node) {
    RelOptTableImpl relOptTable = (RelOptTableImpl) node.getTable();
    BodoSqlTable bodoSqlTable = (BodoSqlTable) relOptTable.table();
    String outputVar = this.genDfVar();
    List<String> outputColumns = node.getRowType().getFieldNames();
    if (isSnowflakeCatalogTable(bodoSqlTable)) {
      this.genRelnodeTimerStart(node);
      // If we are updating Snowflake we push the query into Snowflake.
      // We require the Snowflake Catalog to ensure we don't need to remap
      // any names.
      try {
        this.generatedCode
            .append(getBodoIndent())
            .append(outputVar)
            .append(" = ")
            .append(bodoSqlTable.generateRemoteQuery(this.originalSQLQuery))
            .append("\n");
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
      this.generatedCode.append(getBodoIndent()).append(outputVar).append(".columns = ");
      this.generatedCode.append("[");
      for (String colName : outputColumns) {
        this.generatedCode.append(makeQuoted(colName)).append(", ");
      }
      this.generatedCode.append("]\n");
      this.genRelnodeTimerStop(node);
    } else {
      throw new BodoSQLCodegenException(
          "Delete only supported when all source tables are found within a user's Snowflake"
              + " account and are provided via the Snowflake catalog.");
    }
    this.varGenStack.push(outputVar);
    this.columnNamesStack.push(outputColumns);
  }

  /**
   * Visitor for Logical Aggregate, support for Aggregations in SQL such as SUM, COUNT, MIN, MAX.
   *
   * @param node LogicalAggregate node to be visited
   */
  public void visitLogicalAggregate(LogicalAggregate node) {
    final List<Integer> groupingVariables = node.getGroupSet().asList();
    final List<ImmutableBitSet> groups = node.getGroupSets();

    // Based on the calcite code that we've seen generated, we assume that every Logical Aggregation
    // node has
    // at least one grouping set.
    assert groups.size() > 0;

    List<String> expectedOutputCols = node.getRowType().getFieldNames();

    int nodeId = node.getId();
    String finalOutVar = this.genDfVar();
    if (isNodeCached(node)) {
      Pair<String, List<String>> cacheInfo = this.varCache.get(nodeId);
      this.generatedCode.append(String.format("  %s = %s\n", finalOutVar, cacheInfo.getKey()));
      expectedOutputCols = cacheInfo.getValue();
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
      List<String> inputColumnNames = columnNamesStack.pop();
      String inVar = varGenStack.pop();
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
        String curGroupAggExpr;
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
          Pair<String, String> curGroupAggExprAndAdditionalGeneratedCode =
              handleLogicalAggregateWithGroups(
                  inVar, inputColumnNames, aggCallList, aggCallNames, curGroup);

          curGroupAggExpr = curGroupAggExprAndAdditionalGeneratedCode.getKey();
          this.generatedCode.append(curGroupAggExprAndAdditionalGeneratedCode.getValue());
        }
        // assign each of the generated dataframes their own variable, for greater clarity in the
        // generated code
        String outDfName = this.genDfVar();
        outputDfNames.add(outDfName);
        this.generatedCode
            .append(getBodoIndent())
            .append(outDfName)
            .append(" = ")
            .append(curGroupAggExpr)
            .append("\n");
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
          StringBuilder dummyDfExpr = new StringBuilder(inVar).append(".iloc[:0, :]");

          // Assign the dummy df to a variable name,
          this.generatedCode
              .append(getBodoIndent())
              .append(finalOutVar)
              .append(" = ")
              .append(dummyDfExpr)
              .append("\n");
          concatDfs.add(finalOutVar);
        }
        concatDfs.addAll(outputDfNames);

        // Generate the concatenation
        this.generatedCode
            .append(getBodoIndent())
            .append(finalOutVar)
            .append(" = ")
            .append(concatDataFrames(concatDfs));

        // Sort the output dataframes, so that they are in the ordering expected by Calcite
        // Needed in the case that the topmost dataframe in the concat does not contain all the
        // columns in the correct ordering
        this.generatedCode.append(".loc[:, [");

        for (int i = 0; i < expectedOutputCols.size(); i++) {
          this.generatedCode.append(makeQuoted(expectedOutputCols.get(i))).append(", ");
        }
        this.generatedCode.append("]]\n");

      } else {
        finalOutVar = outputDfNames.get(0);
      }
      this.varCache.put(nodeId, new Pair<>(finalOutVar, expectedOutputCols));
      this.genRelnodeTimerStop(node);
    }
    varGenStack.push(finalOutVar);
    columnNamesStack.push(expectedOutputCols);
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
  public Pair<String, String> handleLogicalAggregateWithGroups(
      String inVar,
      List<String> inputColumnNames,
      List<AggregateCall> aggCallList,
      List<String> aggCallNames,
      List<Integer> group) {

    Pair<String, String> exprAndAdditionalGeneratedCode;
    if (aggContainsFilter(aggCallList)) {
      // If we have a Filter we need to generate a groupby apply
      exprAndAdditionalGeneratedCode =
          generateApplyCodeWithGroupBy(
              inVar,
              inputColumnNames,
              group,
              aggCallList,
              aggCallNames,
              this.genGroupbyApplyAggFnName());

    } else {
      // Otherwise generate groupby.agg
      String groupbyExpr =
          generateAggCodeWithGroupBy(inVar, inputColumnNames, group, aggCallList, aggCallNames);
      exprAndAdditionalGeneratedCode = new Pair<>(groupbyExpr, "");
    }

    return exprAndAdditionalGeneratedCode;
  }

  /**
   * Visitor for RexNode
   *
   * @param node RexNode being visited
   * @param colNames List of colNames used in the relational expression
   * @param inputVar Name of dataframe from which InputRefs select Columns
   * @param id The RelNode id used to uniquely identify the table.
   * @param isSingleRow flag for if table references refer to a single row or the whole table. This
   *     is used for determining if an expr returns a scalar or a column. Only CASE statements set
   *     this to True currently.
   * @param ctx A ctx object containing the Hashset of columns used that need null handling, the
   *     List of precomputed column variables that need to be added to the dataframe before an
   *     apply, and the list of named parameters that need to be passed to an apply function as
   *     arguments.
   * @return RexNodeVisitorInfo containing the new column name and the code generated for the
   *     relational expression.
   */
  public RexNodeVisitorInfo visitRexNode(
      RexNode node,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {
    RexNodeVisitorInfo result;
    // TODO: Add more RexNodes here
    if (node instanceof RexInputRef) {
      result = visitInputRef((RexInputRef) node, colNames, inputVar, isSingleRow, ctx);
    } else if (node instanceof RexLiteral) {
      result = visitLiteralScan((RexLiteral) node, isSingleRow);
    } else if (node instanceof RexOver) {
      // Windowed aggregation is special, since it needs to add generated
      // code in order to define functions to be used with groupby apply.
      // Note: RexOver is also a RexCall, so RexOver must be visited before RexCall.
      List<RexOver> tmp = new ArrayList<>();
      tmp.add((RexOver) node);
      result = visitAggOverOp(tmp, colNames, id, inputVar, isSingleRow, ctx).get(0);
    } else if (node instanceof RexCall
        && ((RexCall) node).getOperator() instanceof SqlNullTreatmentOperator) {
      result = visitNullTreatmentOp((RexCall) node, colNames, id, inputVar, isSingleRow, ctx);
    } else if (node instanceof RexCall) {
      result = visitRexCall((RexCall) node, colNames, id, inputVar, isSingleRow, ctx);
    } else if (node instanceof RexNamedParam) {
      result = visitRexNamedParam((RexNamedParam) node, ctx);
    } else {
      throw new BodoSQLCodegenException(
          "Internal Error: Calcite Plan Produced an Unsupported RexNode");
    }
    return result;
  }

  /**
   * Visitor for RexCalls IGNORE NULLS and RESPECT NULLS This function is only called if IGNORE
   * NULLS and RESPECT NULLS is called without an associated window. Otherwise, it is included as a
   * field in the REX OVER node.
   *
   * <p>Currently, we always throw an error when entering this call. Frankly, based on my reading of
   * calcite's syntax, we only reach this node through invalid syntax in Calcite (LEAD/LAG
   * RESPECT/IGNORE NULL's without a window)
   *
   * @param node RexCall being visited
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
  public RexNodeVisitorInfo visitNullTreatmentOp(
      RexCall node,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {

    SqlKind innerCallKind = node.getOperands().get(0).getKind();
    switch (innerCallKind) {
      case LEAD:
      case LAG:
      case NTH_VALUE:
      case FIRST_VALUE:
      case LAST_VALUE:
        throw new BodoSQLCodegenException(
            "Error during codegen: " + innerCallKind.toString() + " requires OVER clause.");
      default:
        throw new BodoSQLCodegenException(
            "Error during codegen: Unreachable code entered while evaluating the following rex"
                + " node in visitNullTreatmentOp: "
                + node.toString());
    }
  }

  /**
   * Visitor for RexCall
   *
   * @param node RexCall being visited
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
  public RexNodeVisitorInfo visitRexCall(
      RexCall node,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {
    RexNodeVisitorInfo result;
    // TODO: Add more call nodes here
    if (node.getOperator() instanceof SqlBinaryOperator
        || node.getOperator() instanceof SqlDatetimePlusOperator
        || node.getOperator() instanceof SqlDatetimeSubtractionOperator) {
      result = visitBinOpScan(node, colNames, id, inputVar, isSingleRow, ctx);
    } else if (node.getOperator() instanceof SqlPostfixOperator) {
      result = visitPostfixOpScan(node, colNames, id, inputVar, isSingleRow, ctx);
    } else if (node.getOperator() instanceof SqlPrefixOperator) {
      result = visitPrefixOpScan(node, colNames, id, inputVar, isSingleRow, ctx);
    } else if (node.getOperator() instanceof SqlInternalOperator) {
      result = visitInternalOp(node, colNames, id, inputVar, isSingleRow, ctx);
    } else if (node.getOperator() instanceof SqlLikeOperator) {
      result = visitLikeOp(node, colNames, id, inputVar, isSingleRow, ctx);
    } else if (node.getOperator() instanceof SqlCaseOperator) {
      result = visitCaseOp(node, colNames, id, inputVar, isSingleRow, ctx, this);
    } else if (node.getOperator() instanceof SqlTimestampDiffFunction) {
      result = visitTimestampDiff(node, colNames, id, inputVar, isSingleRow, ctx);
    } else if (node.getOperator() instanceof SqlCastFunction) {
      result = visitCastScan(node, colNames, id, inputVar, isSingleRow, ctx);
    } else if (node.getOperator() instanceof SqlExtractFunction) {
      result = visitExtractScan(node, colNames, id, inputVar, isSingleRow, ctx);
    } else if (node.getOperator() instanceof SqlSubstringFunction) {
      result = visitSubstringScan(node, colNames, id, inputVar, isSingleRow, ctx);
    } else if (node.getOperator() instanceof SqlFunction) {
      result = visitGenericFuncOp(node, colNames, id, inputVar, isSingleRow, ctx);
    } else {
      throw new BodoSQLCodegenException(
          "Internal Error: Calcite Plan Produced an Unsupported RexCall:" + node.getOperator());
    }
    return result;
  }

  /**
   * Return a pandas expression that replicates EXTRACT. Currently only tested for year extraction.
   *
   * @param node Internal operator RexCall being visited
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
  private RexNodeVisitorInfo visitExtractScan(
      RexCall node,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {
    RexNodeVisitorInfo dateVal =
        visitRexNode(node.operands.get(0), colNames, id, inputVar, isSingleRow, ctx);
    RexNodeVisitorInfo column =
        visitRexNode(node.operands.get(1), colNames, id, inputVar, isSingleRow, ctx);
    boolean isTime = node.operands.get(1).getType().getSqlTypeName().toString().equals("TIME");
    String codeExpr = generateExtractCode(dateVal.getExprCode(), column.getExprCode(), isTime);
    return new RexNodeVisitorInfo(codeExpr);
  }

  /**
   * Use DataFrame.apply to recreate the functionality of CASE.
   *
   * @param node Internal operator RexCall being visited
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
  private RexNodeVisitorInfo visitCaseOp(
      RexCall node,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx,
      PandasCodeGenVisitor pdVisitorClass) {

    // TODO: Technical debt, this should be done in our fork of calcite
    // Calcite optimizes a large number of windowed aggregation functions into case statements,
    // which check if the window is valid. This can be during the parsing step by setting the
    // "allowPartial" variable to be true.

    if (isWindowedAggFn(node)) {
      return visitRexNode(node.getOperands().get(1), colNames, id, inputVar, isSingleRow, ctx);
    }

    List<RexNode> operands = node.getOperands();
    // Even if the contents are scalars, we will always have a dataframe unless we
    // are inside another apply. We choose to generate apply code in this case because
    // we only compile single basic blocks.
    boolean generateApply = !isSingleRow;

    // The body of case may require wrapping everything inside a global variable.
    // In the future this all needs to be wrapped inside the original builder, but we
    // can't support that yet.
    Module.Builder oldBuilder = this.generatedCode;
    Module.Builder innerBuilder = this.generatedCode;

    if (generateApply) {
      // If we're generating an apply, the input set of columns to add should be empty,
      // since we only add columns to the colsToAddList when inside an apply.
      assert ctx.getColsToAddList().size() == 0;
      // If we generate an apply we need to write the generated if/else
      innerBuilder = new Module.Builder();
    }

    List<String> args = new ArrayList<>();

    BodoCtx localCtx = new BodoCtx();

    for (int i = 0; i < operands.size(); i++) {
      RexNodeVisitorInfo visitorInfo =
          visitRexNode(node.operands.get(i), colNames, id, inputVar, true, localCtx);
      args.add(visitorInfo.getExprCode());
    }

    if (generateApply && localCtx.getColsToAddList().size() > 0) {
      // If we do generate an apply, add the columns that we need to the dataframe
      // and change the variables to reference the new dataframe
      final String indent = getBodoIndent();
      String tmp_case_name = "tmp_case_" + genDfVar();
      this.generatedCode.append(
          indent
              + tmp_case_name
              + " = "
              + generateCombinedDf(inputVar, colNames, localCtx.getColsToAddList())
              + "\n");
      args = renameExprsList(args, inputVar, tmp_case_name);
      inputVar = tmp_case_name;
      // get column names including the added columns
      List<String> newColNames = new ArrayList<>();
      for (String col : colNames) newColNames.add(col);
      for (String col : localCtx.getColsToAddList()) newColNames.add(col);
      colNames = newColNames;
    } else if (!generateApply) {
      // If we're not the top level apply, we need to pass back the information so that it is
      // properly handled
      // by the actual top level apply
      // null columns are handled by the cond itself, so don't need to pass those back
      ctx.getNamedParams().addAll(localCtx.getNamedParams());
      ctx.getColsToAddList().addAll(localCtx.getColsToAddList());
      ctx.getUsedColumns().addAll(localCtx.getUsedColumns());
    }

    String codeExpr =
        generateCaseCode(
            args,
            generateApply,
            localCtx,
            inputVar,
            node.getType(),
            pdVisitorClass,
            oldBuilder,
            innerBuilder);

    return new RexNodeVisitorInfo(codeExpr);
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
  private List<RexNodeVisitorInfo> visitAggOverOp(
      List<RexOver> aggOperations,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {

    final String indent = getBodoIndent();
    if (aggOperations.size() == 0) {
      return new ArrayList<>();
    }

    // Check if we can use the optimized groupby.window C++ kernel. If so
    // we directly to that codegen path.
    if (usesOptimizedEngineKernel(aggOperations)) {
      // usesOptimizedEngineKernel enforces that we have exactly 1 aggOperation
      // and exactly 1 order by column.
      RexOver windowFunc = aggOperations.get(0);
      RexWindow window = windowFunc.getWindow();
      // We need to visit all partition keys before generating
      // code for the kernel
      List<RexNodeVisitorInfo> childExprs = new ArrayList<>();
      List<BodoSQLExprType.ExprType> childExprTypes = new ArrayList<>();
      for (int i = 0; i < window.partitionKeys.size(); i++) {
        RexNode node = window.partitionKeys.get(i);
        RexNodeVisitorInfo curInfo = visitRexNode(node, colNames, id, inputVar, false, ctx);
        childExprs.add(new RexNodeVisitorInfo(curInfo.getExprCode()));
        childExprTypes.add(exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(node, id)));
      }
      // Visit the order by key. This is required to be a single column.
      // The RHS of order keys is a set of SqlKind Values. These are used to stored information
      // about ascending and nulls first/last
      RexNode node = window.orderKeys.get(0).left;
      RexNodeVisitorInfo curRexInfo = visitRexNode(node, colNames, id, inputVar, false, ctx);
      childExprs.add(new RexNodeVisitorInfo(curRexInfo.getExprCode()));
      childExprTypes.add(exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(node, id)));
      Variable outputVar = new Variable(this.genWindowedAggDfName());
      Expr outputCode =
          generateOptimizedEngineKernelCode(
              inputVar, outputVar, aggOperations, childExprs, childExprTypes, this.generatedCode);

      return List.of(new RexNodeVisitorInfo(outputCode.emit()));
    }

    // Used to store each distinct window and the list of corresponding
    // window function calls
    List<List<RexWindow>> windows = new ArrayList<>();
    List<List<SqlKind>> fnKinds = new ArrayList<>();
    List<List<String>> fnNames = new ArrayList<>();
    List<List<RexOver>> aggSets = new ArrayList<>();
    List<List<Boolean>> respectNullsLists = new ArrayList<>();

    // Used to store how each output within each groupby-apply corresponds
    // to one of the original column outputs in the SELECT statement
    List<List<Integer>> indices = new ArrayList<>();

    // Eventually used to store the names of each output
    List<String> outputColExprs = new ArrayList<>();

    // Loop over each aggregation and identify if it can be fused with one
    // of the aggregations already added to the aggSets list
    for (Integer j = 0; j < aggOperations.size(); j++) {
      outputColExprs.add(null);

      RexOver agg = aggOperations.get(j);
      Boolean canFuse = false;
      RexWindow window = agg.getWindow();
      SqlKind fnKind = agg.getAggOperator().getKind();
      String fnName = agg.getAggOperator().getName();
      Boolean respectNulls = !agg.ignoreNulls();
      for (Integer i = 0; i < windows.size(); i++) {

        // Check if the window function can be fused with one of the earlier
        // closures. This is true if it has the same partition & order as the
        // first window in that closure.
        if (windows.get(i).get(0).partitionKeys.equals(window.partitionKeys)
            && windows.get(i).get(0).orderKeys.equals(window.orderKeys)) {
          canFuse = true;
          windows.get(i).add(window);
          fnKinds.get(i).add(fnKind);
          fnNames.get(i).add(fnName);
          aggSets.get(i).add(agg);
          respectNullsLists.get(i).add(respectNulls);
          indices.get(i).add(j);
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
        indices.add(new ArrayList<>(Arrays.asList(j)));
      }
    }

    // For each distinct window/function combination, create a new
    // closure and groupby-apply call
    for (int i = 0; i < windows.size(); i++) {
      List<RexWindow> windowList = windows.get(i);
      List<SqlKind> fnKindList = fnKinds.get(i);
      List<String> fnNameList = fnNames.get(i);
      List<Boolean> respectNullsList = respectNullsLists.get(i);
      List<RexOver> aggSet = aggSets.get(i);
      Pair<String, List<String>> out =
          visitAggOverHelper(
              aggSet,
              colNames,
              windowList,
              fnKindList,
              fnNameList,
              respectNullsList,
              id,
              inputVar,
              ctx);

      // Extract the dataframe whose columns contain the output(s) of the
      // window aggregation
      String dfExpr = out.getKey();
      List<String> outputDfColnameList = out.getValue();
      String generatedDfName = this.genWindowedAggDfName();
      this.generatedCode
          .append(indent)
          .append(generatedDfName)
          .append(" = ")
          .append(dfExpr)
          .append("\n");

      // For each aggregation that was fused into this window, extract the
      // corresponding column
      List<Integer> innerIndices = indices.get(i);
      for (int j = 0; j < aggSet.size(); j++) {
        String outputDfColName = outputDfColnameList.get(j);
        String outputCode =
            new StringBuilder("bodo.hiframes.pd_series_ext.get_series_data(")
                .append(generatedDfName)
                .append("[" + makeQuoted(outputDfColName) + "])")
                .toString();

        // Map the output value back to the correct column location
        Integer index = innerIndices.get(j);
        outputColExprs.set(index, outputCode);
      }
    }

    // Verify that all of the output columns were set
    assert !outputColExprs.contains(null);

    List<RexNodeVisitorInfo> outputRexInfoList = new ArrayList<>();

    for (int i = 0; i < outputColExprs.size(); i++) {
      if (isSingleRow) {
        // In the case that we're inside an apply, we precalculate the column
        // and add the column to colsToAddList. This will result in
        // the precomputed column being added to the dataframe before the apply.
        String colName = genTempColumnVar();
        int colInd = colNames.size() + ctx.getColsToAddList().size();
        ctx.getColsToAddList().add(colName);

        this.generatedCode
            .append(indent)
            .append(colName)
            .append(" = ")
            .append(outputColExprs.get(i))
            .append("\n");

        // Since we're adding the column to the dataframe before the apply, we return
        // an expression that references the added column.
        ctx.getUsedColumns().add(colInd);
        // NOTE: Codegen for bodosql_case_placeholder() expects column value
        // accesses
        // (e.g. bodo.utils.indexing.scalar_optional_getitem(T1_1, i))
        String returnExpr =
            "bodo.utils.indexing.scalar_optional_getitem(" + inputVar + "_" + colInd + ", i)";
        outputRexInfoList.add(new RexNodeVisitorInfo(returnExpr));
      } else {
        outputRexInfoList.add(new RexNodeVisitorInfo(outputColExprs.get(i)));
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
   * @param ctx A ctx object containing the Hashset of columns used that need null handling, the
   *     List of precomputed column variables that need to be added to the dataframe before an
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
      BodoCtx ctx) {

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
    List<RexNodeVisitorInfo> childExprs = new ArrayList<>();
    List<BodoSQLExprType.ExprType> exprTypes = new ArrayList<>();
    List<String> groupbyCols = new ArrayList<>();

    // simple incremented variable, used for making sure we don't have duplicate column names
    int col_id_var = 0;

    RexWindow window = windows.get(0);

    if (window.partitionKeys.size() == 0) {
      throw new BodoSQLCodegenException(
          "BODOSQL currently requires a partition column when handling windowed aggregation"
              + " functions");
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
      RexNodeVisitorInfo curInfo = visitRexNode(node, colNames, id, inputVar, false, ctx);
      String colName = "GRPBY_COL_" + col_id_var++;
      childExprNames.add(colName);
      childExprs.add(new RexNodeVisitorInfo(curInfo.getExprCode()));
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
      RexNodeVisitorInfo curRexInfo = visitRexNode(node, colNames, id, inputVar, false, ctx);
      RelFieldCollation.Direction dir = window.orderKeys.get(i).getDirection();
      RelFieldCollation.NullDirection nullDir = window.orderKeys.get(i).getNullDirection();
      if (dir == RelFieldCollation.Direction.ASCENDING) {
        String colName = "ASC_COL_" + col_id_var++;
        childExprNames.add(colName);
        childExprs.add(new RexNodeVisitorInfo(curRexInfo.getExprCode()));
        orderKeys.add(makeQuoted(colName));
      } else {
        assert dir == RelFieldCollation.Direction.DESCENDING;
        String colName = "DEC_COL_" + col_id_var++;
        childExprNames.add(colName);
        childExprs.add(new RexNodeVisitorInfo(curRexInfo.getExprCode()));
        orderKeys.add(makeQuoted(colName));
      }
      orderAscending.add(getAscendingBoolString(dir));
      orderNAPosition.add(getNAPositionString(nullDir));
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
          RexNodeVisitorInfo curInfo =
              visitRexNode(
                  curAggOperation.getOperands().get(i), colNames, id, inputVar, false, ctx);
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
              childExprs.add(new RexNodeVisitorInfo(curInfo.getExprCode()));
              BodoSQLExprType.ExprType curExprType =
                  exprTypesMap.get(
                      ExprTypeVisitor.generateRexNodeKey(curAggOperation.getOperands().get(i), id));
              exprTypes.add(curExprType);
              curArgslist.add(WindowedAggregationArgument.fromColumnName(curAggColName));
            } else {
              // In the case of NTILE, it's always a scalar literal, so we just add it to argsList.
              curArgslist.add(WindowedAggregationArgument.fromLiteralExpr(curInfo.getExprCode()));
            }

          } else {
            // For all other currently supported window functions, all arguments after the 0-th
            // are always scalar literals
            assert exprTypesMap.get(
                    ExprTypeVisitor.generateRexNodeKey(curAggOperation.getOperands().get(i), id))
                == BodoSQLExprType.ExprType.SCALAR;
            curArgslist.add(WindowedAggregationArgument.fromLiteralExpr(curInfo.getExprCode()));
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
              ctx,
              zeroExpr);
      String upperBound =
          extractWindowBound(
              window.getUpperBound(),
              "upper",
              WindowAggCodeGen.unboundedUpperBound,
              colNames,
              id,
              inputVar,
              ctx,
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
    childExprs.add(new RexNodeVisitorInfo("np.arange(len(" + inputVar + "))"));
    exprTypes.add(BodoSQLExprType.ExprType.COLUMN);

    // Create the projection of the input dataframe, which contains only the values which we require
    // in order to
    // perform the window function(s)
    String projection = generateProjectedDataframe(inputVar, childExprNames, childExprs, exprTypes);

    StringBuilder groupedColExpr = new StringBuilder(projection);
    // Projection is done, now we need to produce something like this:
    // projection_df.groupby(whatever cols).apply(my_fn(args))["AGGCOL"]

    // Sort the dataframe by the ORDER BY columns of the window

    // Ascending/Decending/NAPosition have an effect on the output)?

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
    } else {
      throw new BodoSQLCodegenException(
          "Error, cannot currently perform windowed aggregation without a partition clause");
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
      String fn_name = this.genWindowedAggFnName();

      Pair<String, List<String>> out =
          generateWindowedAggFn(
              fn_name,
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
      String fn_text = out.getKey();
      outputColList = out.getValue();

      // The length of the output column list should be the same length as argsListList
      assert argsListList.size() == outputColList.size();
      this.generatedCode.append(fn_text);

      // perform the groupby apply, using the generated function call
      outputExpr.append(groupedColExpr).append(".apply(").append(fn_name).append(")");

    } else {
      throw new BodoSQLCodegenException(
          "Error, cannot currently perform windowed aggregation without a partition clause");
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
   * @param ctx A ctx object containing the Hashset of columns used that need null handling, the
   *     List of precomputed column variables that need to be added to the dataframe before an
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
      BodoCtx ctx,
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
      result =
          "-(" + visitRexNode(boundNode, colNames, id, inputVar, false, ctx).getExprCode() + ")";
      boundExprType = exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(boundNode, id));
    } else if (bound.isFollowing()) {
      result = visitRexNode(boundNode, colNames, id, inputVar, false, ctx).getExprCode();
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

  private RexNodeVisitorInfo visitTimestampDiff(
      RexCall fnOperation,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {

    List<BodoSQLExprType.ExprType> exprTypes = new ArrayList<>();
    for (RexNode node : fnOperation.getOperands()) {
      exprTypes.add(exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(node, id)));
    }

    // The first argument is required to be an interval symbol.
    assert fnOperation.getOperands().get(0) instanceof RexLiteral;

    // Extract all the inputs to the current function
    List<RexNodeVisitorInfo> operandsInfo =
        new ArrayList<RexNodeVisitorInfo>() {
          {
            for (RexNode operand : fnOperation.operands) {
              add(visitRexNode(operand, colNames, id, inputVar, isSingleRow, ctx));
            }
          }
        };

    return generateTimestampDiffInfo(inputVar, exprTypes, operandsInfo, ctx);
  }

  /**
   * Return a pandas expression that replicates an SQL function call
   *
   * @param fnOperation Internal operator RexCall being visited
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
  private RexNodeVisitorInfo visitGenericFuncOp(
      RexCall fnOperation,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {

    String fnName = fnOperation.getOperator().toString();

    List<BodoSQLExprType.ExprType> exprTypes = new ArrayList<>();
    for (RexNode node : fnOperation.getOperands()) {
      exprTypes.add(exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(node, id)));
    }

    // Handle IF, COALESCE, DECODE and their variants seperately
    if (fnName == "COALESCE"
        || fnName == "NVL"
        || fnName == "NVL2"
        || fnName == "BOOLAND"
        || fnName == "BOOLOR"
        || fnName == "BOOLXOR"
        || fnName == "BOOLNOT"
        || fnName == "EQUAL_NULL"
        || fnName == "ZEROIFNULL"
        || fnName == "IFNULL"
        || fnName == "IF"
        || fnName == "IFF"
        || fnName == "DECODE") {
      List<String> codeExprs = new ArrayList<>();
      BodoCtx localCtx = new BodoCtx();
      int j = 0;
      for (RexNode operand : fnOperation.operands) {
        RexNodeVisitorInfo operandInfo =
            visitRexNode(operand, colNames, id, inputVar, isSingleRow, localCtx);
        String expr = operandInfo.getExprCode();
        // Need to unbox scalar timestamp values.
        if (isSingleRow || (exprTypes.get(j) == BodoSQLExprType.ExprType.SCALAR)) {
          expr = "bodo.utils.conversion.unbox_if_tz_naive_timestamp(" + expr + ")";
        }
        codeExprs.add(expr);
        j++;
      }

      RexNodeVisitorInfo result;
      switch (fnName) {
        case "IF":
        case "IFF":
          result = visitIf(fnOperation, codeExprs);
          break;
        case "BOOLNOT":
          result = getSingleArgCondFnInfo(fnName, codeExprs.get(0));
          break;
        case "BOOLAND":
        case "BOOLOR":
        case "BOOLXOR":
        case "EQUAL_NULL":
          result = getDoubleArgCondFnInfo(fnName, codeExprs.get(0), codeExprs.get(1));
          break;
        case "COALESCE":
        case "ZEROIFNULL":
        case "IFNULL":
        case "NVL":
        case "NVL2":
        case "DECODE":
          result = visitVariadic(fnOperation, codeExprs);
          break;
        default:
          throw new BodoSQLCodegenException("Internal Error: reached unreachable code");
      }

      // If we're not the top level apply, we need to pass back the information so that it is
      // properly handled by the actual top level apply
      ctx.getColsToAddList().addAll(localCtx.getColsToAddList());
      ctx.getNamedParams().addAll(localCtx.getNamedParams());
      ctx.getUsedColumns().addAll(localCtx.getUsedColumns());

      return result;
    }

    // These need to be final variable, to make java happy
    final String tmp_input_val = inputVar;
    // Extract all the inputs to the current function
    List<RexNodeVisitorInfo> operandsInfo =
        new ArrayList<RexNodeVisitorInfo>() {
          {
            for (RexNode operand : fnOperation.operands) {
              add(visitRexNode(operand, colNames, id, tmp_input_val, isSingleRow, ctx));
            }
          }
        };

    String expr;
    BodoSQLExprType.ExprType exprType;
    String strExpr;
    switch (fnOperation.getOperator().kind) {
      case CEIL:
      case FLOOR:
        return getSingleArgNumericFnInfo(
            fnOperation.getOperator().toString(), operandsInfo.get(0).getExprCode());
      case MOD:
        return getDoubleArgNumericFnInfo(
            fnOperation.getOperator().toString(),
            operandsInfo.get(0).getExprCode(),
            operandsInfo.get(1).getExprCode());
      case TRIM:
        assert operandsInfo.size() == 3;
        // Calcite expects: TRIM(<chars> FROM <expr>>) or TRIM(<chars>, <expr>)
        // However, Snowflake/BodoSQL expects: TRIM(<expr>, <chars>)
        // So we just need to swap the arguments here.
        return generateTrimFnInfo(
            fnOperation.getOperator().toString(), operandsInfo.get(2), operandsInfo.get(1));
      case NULLIF:
        assert operandsInfo.size() == 2;
        expr =
            "bodo.libs.bodosql_array_kernels.nullif("
                + operandsInfo.get(0).getExprCode()
                + ", "
                + operandsInfo.get(1).getExprCode()
                + ")";
        return new RexNodeVisitorInfo(expr);

      case POSITION:
        return generatePosition(operandsInfo);
      case OTHER:
      case OTHER_FUNCTION:
        boolean isTime;
        String unit;
        String tzStr;
        /* If sqlKind = other function, the only recourse is to match on the name of the function. */
        switch (fnName) {
            // TODO (allai5): update this in a future PR for clean-up so it re-uses the
            // SQLLibraryOperator definition.
          case "LTRIM":
          case "RTRIM":
            if (operandsInfo.size() == 1) { // no optional characters to be trimmed
              return generateTrimFnInfo(
                  fnName,
                  operandsInfo.get(0),
                  new RexNodeVisitorInfo("' '")); // remove spaces by default
            } else if (operandsInfo.size() == 2) {
              return generateTrimFnInfo(fnName, operandsInfo.get(0), operandsInfo.get(1));
            } else {
              throw new BodoSQLCodegenException(
                  "Invalid number of arguments to TRIM: must be either 1 or 2.");
            }
          case "WIDTH_BUCKET":
            {
              int numOps = operandsInfo.size();
              assert numOps == 4 : "WIDTH_BUCKET takes 4 arguments, but found " + numOps;
              StringBuilder exprCode =
                  new StringBuilder("bodo.libs.bodosql_array_kernels.width_bucket(");
              for (int i = 0; i < numOps; i++) {
                exprCode.append(operandsInfo.get(i).getExprCode());
                if (i != (numOps - 1)) {
                  exprCode.append(", ");
                }
              }
              exprCode.append(")");
              return new RexNodeVisitorInfo(exprCode.toString());
            }
          case "HAVERSINE":
            {
              assert operandsInfo.size() == 4;
              StringBuilder exprCode =
                  new StringBuilder("bodo.libs.bodosql_array_kernels.haversine(");
              int numOps = fnOperation.operands.size();
              for (int i = 0; i < numOps; i++) {
                exprCode.append(operandsInfo.get(i).getExprCode());
                if (i != (numOps - 1)) {
                  exprCode.append(", ");
                }
              }
              exprCode.append(")");
              return new RexNodeVisitorInfo(exprCode.toString());
            }
          case "DIV0":
            {
              assert operandsInfo.size() == 2 && fnOperation.operands.size() == 2;
              StringBuilder exprCode = new StringBuilder("bodo.libs.bodosql_array_kernels.div0(");
              exprCode.append(operandsInfo.get(0).getExprCode());
              exprCode.append(", ");
              exprCode.append(operandsInfo.get(1).getExprCode());
              exprCode.append(")");
              return new RexNodeVisitorInfo(exprCode.toString());
            }
          case "NULLIFZERO":
            assert operandsInfo.size() == 1;
            String exprCode =
                "bodo.libs.bodosql_array_kernels.nullif("
                    + operandsInfo.get(0).getExprCode()
                    + ", 0)";
            return new RexNodeVisitorInfo(exprCode);
          case "DATEADD":
          case "TIMEADD":
            // If DATEADD receives 3 arguments, use the Snowflake DATEADD.
            // Otherwise, fall back to the normal DATEADD. TIMEADD is an alias.
            if (operandsInfo.size() == 3) {
              assert exprTypes.get(0) == BodoSQLExprType.ExprType.SCALAR;
              return generateSnowflakeDateAddCode(operandsInfo);
            }
          case "DATE_ADD":
          case "ADDDATE":
          case "SUBDATE":
          case "DATE_SUB":
            assert operandsInfo.size() == 2;
            // If the second argument is a timedelta, switch to manual addition
            boolean manual_addition =
                SqlTypeName.INTERVAL_TYPES.contains(
                    fnOperation.getOperands().get(1).getType().getSqlTypeName());
            // Cast arg0 to from string to timestamp, if needed
            if (SqlTypeName.STRING_TYPES.contains(
                fnOperation.getOperands().get(0).getType().getSqlTypeName())) {
              RelDataType inputType = fnOperation.getOperands().get(0).getType();
              // The output type will always be the timestamp the string is being cast to.
              RelDataType outputType = fnOperation.getType();
              String casted_expr =
                  generateCastCode(
                      operandsInfo.get(0).getExprCode(),
                      inputType,
                      outputType,
                      exprTypes.get(0) == BodoSQLExprType.ExprType.SCALAR || isSingleRow);
              operandsInfo.set(0, new RexNodeVisitorInfo(casted_expr));
            }

            String arg1Expr = operandsInfo.get(1).getExprCode();
            String addExpr =
                generateMySQLDateAddCode(
                    operandsInfo.get(0).getExprCode(), arg1Expr, manual_addition, fnName);
            return new RexNodeVisitorInfo(addExpr);

          case "DATEDIFF":
            RexNodeVisitorInfo arg1;
            RexNodeVisitorInfo arg2;
            unit = "DAY";

            if (operandsInfo.size() == 2) {
              arg1 = operandsInfo.get(1);
              arg2 = operandsInfo.get(0);
            } else if (operandsInfo.size() == 3) { // this is the Snowflake option
              unit = operandsInfo.get(0).getExprCode();
              arg1 = operandsInfo.get(1);
              arg2 = operandsInfo.get(2);
            } else {
              throw new BodoSQLCodegenException(
                  "Invalid number of arguments to DATEDIFF: must be 2 or 3.");
            }
            isTime =
                fnOperation
                    .getOperands()
                    .get(1)
                    .getType()
                    .getSqlTypeName()
                    .toString()
                    .equals("TIME");
            unit = standardizeTimeUnit(fnName, unit, isTime);
            return generateDateDiffFnInfo(unit, arg1, arg2);
          case "TIMEDIFF":
            assert operandsInfo.size() == 3;
            isTime =
                fnOperation
                    .getOperands()
                    .get(1)
                    .getType()
                    .getSqlTypeName()
                    .toString()
                    .equals("TIME");
            unit = standardizeTimeUnit(fnName, operandsInfo.get(0).getExprCode(), isTime);
            arg1 = operandsInfo.get(1);
            arg2 = operandsInfo.get(2);
            return generateDateDiffFnInfo(unit, arg1, arg2);
          case "STR_TO_DATE":
            assert operandsInfo.size() == 2;
            // Format string should be a string literal.
            // This is required by the function definition.
            if (!(fnOperation.operands.get(1) instanceof RexLiteral)) {
              throw new BodoSQLCodegenException(
                  "Error STR_TO_DATE(): 'Format' must be a string literal");
            }
            strExpr =
                generateStrToDateCode(
                    operandsInfo.get(0).getExprCode(),
                    exprTypes.get(0),
                    operandsInfo.get(1).getExprCode());
            return new RexNodeVisitorInfo(strExpr);
          case "TIMESTAMP":
            return generateTimestampFnCode(operandsInfo.get(0).getExprCode());
          case "DATE":
            return generateDateFnCode(operandsInfo.get(0).getExprCode());
          case "TO_DATE":
          case "TRY_TO_DATE":
            return generateToDateFnCode(operandsInfo, fnName);
          case "TO_TIMESTAMP":
          case "TO_TIMESTAMP_NTZ":
          case "TO_TIMESTAMP_LTZ":
          case "TO_TIMESTAMP_TZ":
          case "TRY_TO_TIMESTAMP":
          case "TRY_TO_TIMESTAMP_NTZ":
          case "TRY_TO_TIMESTAMP_LTZ":
          case "TRY_TO_TIMESTAMP_TZ":
            tzStr = "None";
            if (fnOperation.getType() instanceof TZAwareSqlType) {
              tzStr = ((TZAwareSqlType) fnOperation.getType()).getTZInfo().getPyZone();
            }
            return generateToTimestampFnCode(
                operandsInfo, fnOperation.getOperands(), tzStr, fnName);
          case "TRY_TO_BOOLEAN":
          case "TO_BOOLEAN":
            return generateToBooleanFnCode(operandsInfo, fnName);
          case "TRY_TO_BINARY":
          case "TO_BINARY":
            return generateToBinaryFnCode(operandsInfo, fnName);
          case "TO_CHAR":
          case "TO_VARCHAR":
            return generateToCharFnCode(operandsInfo, fnName);
          case "TO_DOUBLE":
          case "TRY_TO_DOUBLE":
            return generateToDoubleFnCode(operandsInfo, fnName);
          case "ASINH":
          case "ACOSH":
          case "ATANH":
          case "SINH":
          case "COSH":
          case "TANH":
          case "COS":
          case "SIN":
          case "TAN":
          case "COT":
          case "ACOS":
          case "ASIN":
          case "ATAN":
          case "DEGREES":
          case "RADIANS":
            return getSingleArgTrigFnInfo(fnName, operandsInfo.get(0).getExprCode());
          case "ATAN2":
            return getDoubleArgTrigFnInfo(
                fnName, operandsInfo.get(0).getExprCode(), operandsInfo.get(1).getExprCode());
          case "ABS":
          case "CBRT":
          case "EXP":
          case "FACTORIAL":
          case "LOG2":
          case "LOG10":
          case "LN":
          case "SIGN":
          case "SQUARE":
          case "SQRT":
          case "BITNOT":
            return getSingleArgNumericFnInfo(fnName, operandsInfo.get(0).getExprCode());
          case "POWER":
          case "POW":
          case "BITAND":
          case "BITOR":
          case "BITXOR":
          case "BITSHIFTLEFT":
          case "BITSHIFTRIGHT":
          case "GETBIT":
            return getDoubleArgNumericFnInfo(
                fnName, operandsInfo.get(0).getExprCode(), operandsInfo.get(1).getExprCode());
          case "TRUNC":
          case "TRUNCATE":
          case "ROUND":
            String arg1_expr_code;
            if (operandsInfo.size() == 1) {
              // If no value is specified by, default to 0
              arg1_expr_code = "0";
            } else {
              assert operandsInfo.size() == 2;
              arg1_expr_code = operandsInfo.get(1).getExprCode();
            }
            return getDoubleArgNumericFnInfo(
                fnName, operandsInfo.get(0).getExprCode(), arg1_expr_code);

          case "LOG":
            return generateLogFnInfo(operandsInfo, exprTypes, isSingleRow);
          case "CONV":
            assert operandsInfo.size() == 3;
            strExpr =
                generateConvCode(
                    operandsInfo.get(0).getExprCode(),
                    operandsInfo.get(1).getExprCode(),
                    operandsInfo.get(2).getExprCode(),
                    isSingleRow
                        || (exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(fnOperation, id))
                            == BodoSQLExprType.ExprType.SCALAR));
            return new RexNodeVisitorInfo(strExpr);
          case "RAND":
            return new RexNodeVisitorInfo("np.random.rand()");
          case "PI":
            return new RexNodeVisitorInfo("np.pi");
          case "CONCAT":
            return generateConcatFnInfo(operandsInfo);
          case "CONCAT_WS":
            assert operandsInfo.size() >= 2;
            return generateConcatWSFnInfo(
                operandsInfo.get(0), operandsInfo.subList(1, operandsInfo.size()));
          case "GETDATE":
          case "CURRENT_TIMESTAMP":
          case "NOW":
          case "LOCALTIMESTAMP":
          case "SYSTIMESTAMP":
            assert operandsInfo.size() == 0;
            assert fnOperation.getType() instanceof TZAwareSqlType;
            BodoTZInfo tzTimestampInfo = ((TZAwareSqlType) fnOperation.getType()).getTZInfo();
            return generateCurrTimestampCode(fnName, tzTimestampInfo);
          case "CURRENT_TIME":
          case "LOCALTIME":
            assert operandsInfo.size() == 0;
            BodoTZInfo tzTimeInfo = this.typeSystem.getDefaultTZInfo();
            return generateCurrTimeCode(fnName, tzTimeInfo);
          case "SYSDATE":
          case "UTC_TIMESTAMP":
            assert operandsInfo.size() == 0;
            return generateUTCTimestampCode(fnName);
          case "UTC_DATE":
            assert operandsInfo.size() == 0;
            return generateUTCDateCode();
          case "MAKEDATE":
            assert operandsInfo.size() == 2;
            return generateMakeDateInfo(operandsInfo.get(0), operandsInfo.get(1));
          case "DATE_FORMAT":
            if (!(operandsInfo.size() == 2
                && exprTypes.get(1) == BodoSQLExprType.ExprType.SCALAR)) {
              throw new BodoSQLCodegenException(
                  "Error, invalid argument types passed to DATE_FORMAT");
            }
            if (!(fnOperation.operands.get(1) instanceof RexLiteral)) {
              throw new BodoSQLCodegenException(
                  "Error DATE_FORMAT(): 'Format' must be a string literal");
            }
            return generateDateFormatCode(
                operandsInfo.get(0), exprTypes.get(0), operandsInfo.get(1), isSingleRow);
          case "CURRENT_DATE":
          case "CURDATE":
            assert operandsInfo.size() == 0;
            return generateCurdateCode(fnName);
          case "YEARWEEK":
            assert operandsInfo.size() == 1;
            return getYearWeekFnInfo(operandsInfo.get(0));
          case "MONTHNAME":
          case "MONTH_NAME":
          case "DAYNAME":
          case "WEEKDAY":
          case "LAST_DAY":
          case "YEAROFWEEK":
          case "YEAROFWEEKISO":
            assert operandsInfo.size() == 1;
            return getSingleArgDatetimeFnInfo(fnName, operandsInfo.get(0).getExprCode());
          case "NEXT_DAY":
          case "PREVIOUS_DAY":
            assert operandsInfo.size() == 2;
            return getDoubleArgDatetimeFnInfo(
                fnName, operandsInfo.get(0).getExprCode(), operandsInfo.get(1).getExprCode());
          case "DATE_PART":
            assert operandsInfo.size() == 2;
            assert exprTypes.get(0) == BodoSQLExprType.ExprType.SCALAR;
            isTime =
                fnOperation
                    .getOperands()
                    .get(1)
                    .getType()
                    .getSqlTypeName()
                    .toString()
                    .equals("TIME");
            return generateDatePart(operandsInfo, isTime);
          case "TO_DAYS":
            return generateToDaysCode(operandsInfo.get(0));
          case "TO_SECONDS":
            return generateToSecondsCode(operandsInfo.get(0));
          case "FROM_DAYS":
            return generateFromDaysCode(operandsInfo.get(0));
          case "TIME":
          case "TO_TIME":
            return generateToTimeCode(
                fnOperation.getOperands().get(0).getType().getSqlTypeName(),
                operandsInfo.get(0),
                fnName);
          case "DATE_FROM_PARTS":
          case "DATEFROMPARTS":
          case "TIMEFROMPARTS":
          case "TIME_FROM_PARTS":
          case "TIMESTAMP_FROM_PARTS":
          case "TIMESTAMPFROMPARTS":
          case "TIMESTAMP_NTZ_FROM_PARTS":
          case "TIMESTAMPNTZFROMPARTS":
          case "TIMESTAMP_LTZ_FROM_PARTS":
          case "TIMESTAMPLTZFROMPARTS":
          case "TIMESTAMP_TZ_FROM_PARTS":
          case "TIMESTAMPTZFROMPARTS":
            tzStr = "None";
            if (fnOperation.getType() instanceof TZAwareSqlType) {
              tzStr = ((TZAwareSqlType) fnOperation.getType()).getTZInfo().getPyZone();
            }
            return generateDateTimeTypeFromPartsCode(fnName, operandsInfo, tzStr);
          case "TO_NUMBER":
          case "TO_NUMERIC":
          case "TO_DECIMAL":
            return generateToNumberCode(operandsInfo.get(0), fnName);
          case "TRY_TO_NUMBER":
          case "TRY_TO_NUMERIC":
          case "TRY_TO_DECIMAL":
            return generateTryToNumberCode(operandsInfo.get(0), fnName);
          case "UNIX_TIMESTAMP":
            return generateUnixTimestamp();
          case "FROM_UNIXTIME":
            return generateFromUnixTimeCode(operandsInfo.get(0));
          case "JSON_EXTRACT_PATH_TEXT":
            return generateJsonTwoArgsInfo(fnName, operandsInfo.get(0), operandsInfo.get(1));
          case "RLIKE":
          case "REGEXP_LIKE":
            if (!(2 <= operandsInfo.size() && operandsInfo.size() <= 3)) {
              throw new BodoSQLCodegenException(
                  "Error, invalid number of arguments passed to REGEXP_LIKE");
            }
            if (exprTypes.get(1) != BodoSQLExprType.ExprType.SCALAR
                || (operandsInfo.size() == 3
                    && exprTypes.get(2) != BodoSQLExprType.ExprType.SCALAR)) {
              throw new BodoSQLCodegenException(
                  "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
            }
            return generateRegexpLikeInfo(operandsInfo);
          case "REGEXP_COUNT":
            if (!(2 <= operandsInfo.size() && operandsInfo.size() <= 4)) {
              throw new BodoSQLCodegenException(
                  "Error, invalid number of arguments passed to REGEXP_COUNT");
            }
            if (exprTypes.get(1) != BodoSQLExprType.ExprType.SCALAR
                || (operandsInfo.size() == 4
                    && exprTypes.get(3) != BodoSQLExprType.ExprType.SCALAR)) {
              throw new BodoSQLCodegenException(
                  "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
            }
            return generateRegexpCountInfo(operandsInfo);
          case "REGEXP_REPLACE":
            if (!(2 <= operandsInfo.size() && operandsInfo.size() <= 6)) {
              throw new BodoSQLCodegenException(
                  "Error, invalid number of arguments passed to REGEXP_REPLACE");
            }
            if (exprTypes.get(1) != BodoSQLExprType.ExprType.SCALAR
                || (operandsInfo.size() == 6
                    && exprTypes.get(5) != BodoSQLExprType.ExprType.SCALAR)) {
              throw new BodoSQLCodegenException(
                  "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
            }
            return generateRegexpReplaceInfo(operandsInfo);
          case "REGEXP_SUBSTR":
            if (!(2 <= operandsInfo.size() && operandsInfo.size() <= 6)) {
              throw new BodoSQLCodegenException(
                  "Error, invalid number of arguments passed to REGEXP_SUBSTR");
            }
            if (exprTypes.get(1) != BodoSQLExprType.ExprType.SCALAR
                || (operandsInfo.size() > 4
                    && exprTypes.get(4) != BodoSQLExprType.ExprType.SCALAR)) {
              throw new BodoSQLCodegenException(
                  "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
            }
            return generateRegexpSubstrInfo(operandsInfo);
          case "REGEXP_INSTR":
            if (!(2 <= operandsInfo.size() && operandsInfo.size() <= 7)) {
              throw new BodoSQLCodegenException(
                  "Error, invalid number of arguments passed to REGEXP_INSTR");
            }
            if (exprTypes.get(1) != BodoSQLExprType.ExprType.SCALAR
                || (operandsInfo.size() > 5
                    && exprTypes.get(5) != BodoSQLExprType.ExprType.SCALAR)) {
              throw new BodoSQLCodegenException(
                  "Error, PATTERN & FLAG argument for REGEXP functions must be a scalar");
            }
            return generateRegexpInstrInfo(operandsInfo);
          case "ORD":
          case "ASCII":
          case "CHAR":
          case "CHR":
          case "CHAR_LENGTH":
          case "CHARACTER_LENGTH":
          case "LEN":
          case "LENGTH":
          case "REVERSE":
          case "LCASE":
          case "LOWER":
          case "UCASE":
          case "UPPER":
          case "SPACE":
          case "RTRIMMED_LENGTH":
            if (operandsInfo.size() != 1) {
              throw new BodoSQLCodegenException(fnName + " requires providing only 1 argument");
            }
            return getSingleArgStringFnInfo(fnName, operandsInfo.get(0).getExprCode());
          case "FORMAT":
          case "REPEAT":
          case "STRCMP":
          case "RIGHT":
          case "LEFT":
          case "CONTAINS":
          case "INSTR":
          case "STARTSWITH":
          case "ENDSWITH":
            if (operandsInfo.size() != 2) {
              throw new BodoSQLCodegenException(fnName + " requires providing only 2 arguments");
            }
            return getTwoArgStringFnInfo(
                fnName, inputVar, operandsInfo.get(0), operandsInfo.get(1));
          case "RPAD":
          case "LPAD":
          case "SPLIT_PART":
          case "REPLACE":
          case "MID":
          case "SUBSTR":
          case "SUBSTRING_INDEX":
          case "TRANSLATE3":
            if (operandsInfo.size() != 3) {
              throw new BodoSQLCodegenException(fnName + " requires providing only 3 argument");
            }
            return getThreeArgStringFnInfo(
                fnName, operandsInfo.get(0), operandsInfo.get(1), operandsInfo.get(2));
          case "INSERT":
            return generateInsert(operandsInfo);
          case "POSITION":
          case "CHARINDEX":
            return generatePosition(operandsInfo);
          case "STRTOK":
            return generateStrtok(operandsInfo);
          case "EDITDISTANCE":
            return generateEditdistance(operandsInfo);
          case "INITCAP":
            return generateInitcapInfo(operandsInfo);
          case "DATE_TRUNC":
            isTime =
                fnOperation
                    .getOperands()
                    .get(1)
                    .getType()
                    .getSqlTypeName()
                    .toString()
                    .equals("TIME");
            unit = standardizeTimeUnit(fnName, operandsInfo.get(0).getExprCode(), isTime);
            return generateDateTruncCode(unit, operandsInfo.get(1));
          case "MICROSECOND":
          case "SECOND":
          case "MINUTE":
          case "DAY":
          case "DAYOFYEAR":
          case "DAYOFWEEK":
          case "DAYOFWEEKISO":
          case "DAYOFMONTH":
          case "HOUR":
          case "MONTH":
          case "QUARTER":
          case "YEAR":
          case "WEEK":
          case "WEEKOFYEAR":
          case "WEEKISO":
            isTime =
                fnOperation
                    .getOperands()
                    .get(0)
                    .getType()
                    .getSqlTypeName()
                    .toString()
                    .equals("TIME");
            return new RexNodeVisitorInfo(
                generateExtractCode(fnName, operandsInfo.get(0).getExprCode(), isTime));
          case "REGR_VALX":
          case "REGR_VALY":
            return getDoubleArgCondFnInfo(
                fnName, operandsInfo.get(0).getExprCode(), operandsInfo.get(1).getExprCode());
        }
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Function: " + fnOperation.getOperator().toString() + " not supported");
    }
  }

  /**
   * Visitor for RexNodes which are created as internal SQL operations within calcite.
   *
   * <p>Return a relational expression in Pandas that implements the LIKE operator. LIKE filters its
   * input based on a regular expression. The SQL wildcards include '%' and '_' which respectively
   * map to '.*' and '.' in Posix.
   *
   * <p>Notably, this function matches case sensitively to match with pyspark.sql. This is different
   * from other versions of SQL such as MySQL, which match case insensitively.
   *
   * @param node Internal operator RexCall being visited
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
  private RexNodeVisitorInfo visitLikeOp(
      RexCall node,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {

    // The input node has ${index} as its first operand, where
    // ${index} is something like $3, and a SQL regular expression
    // as its second operand. If there is an escape value it will
    // be the third value, although it is not required and only supported
    // for LIKE and ILIKE
    String opName = node.getOperator().getName();
    List<RexNode> operands = node.getOperands();
    RexNode colIndex = operands.get(0);
    RexNodeVisitorInfo arg = visitRexNode(colIndex, colNames, id, inputVar, isSingleRow, ctx);
    String argCode = arg.getExprCode();
    RexNode patternNode = operands.get(1);

    // The regular expression functions only support literal patterns
    if ((opName.equals("REGEXP") || opName.equals("RLIKE"))
        && !(patternNode instanceof RexLiteral)) {
      throw new BodoSQLCodegenException(
          String.format("%s Error: Pattern must be a string literal", opName));
    }
    RexNodeVisitorInfo pattern =
        visitRexNode(patternNode, colNames, id, inputVar, isSingleRow, ctx);
    String sqlPattern = pattern.getExprCode();
    // If escape is missing use the empty string.
    String sqlEscape = "''";
    if (operands.size() == 3) {
      RexNode escapeNode = operands.get(2);
      RexNodeVisitorInfo escape =
          visitRexNode(escapeNode, colNames, id, inputVar, isSingleRow, ctx);
      sqlEscape = escape.getExprCode();
    }
    /* Assumption: Typing in LIKE requires this to be a string type. */
    String likeCode = generateLikeCode(opName, argCode, sqlPattern, sqlEscape);
    return new RexNodeVisitorInfo(likeCode);
  }

  /**
   * Visitor for RexNodes which are created as internal SQL operations within calcite
   *
   * <p>Use the slice command to recreate functionality of substring.
   *
   * @param node Internal operator RexCall being visited
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
  private RexNodeVisitorInfo visitSubstringScan(
      RexCall node,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {
    // node.operands contains
    //  * String to perform the substring operation on
    //  * start index
    //  * substring length
    //  All of these values can be both scalars and columns
    assert node.operands.size() == 3;
    List<BodoSQLExprType.ExprType> exprTypes = new ArrayList<>();
    for (RexNode operand_node : node.operands) {
      exprTypes.add(exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(operand_node, id)));
    }
    List<RexNodeVisitorInfo> operandsInfo = new ArrayList<>();
    for (RexNode childNode : node.operands) {
      operandsInfo.add(visitRexNode(childNode, colNames, id, inputVar, isSingleRow, ctx));
    }
    String fnName = node.getOperator().getName();
    return getThreeArgStringFnInfo(
        fnName, operandsInfo.get(0), operandsInfo.get(1), operandsInfo.get(2));
  }

  /**
   * Visitor for RexNodes which are created as internal SQL operations within calcite
   *
   * @param node Internal operator RexCall being visited
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
  private RexNodeVisitorInfo visitInternalOp(
      RexCall node,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {
    RexNodeVisitorInfo result;
    String expr;
    SqlKind sqlOp = node.getOperator().getKind();
    switch (sqlOp) {
        /* TODO(Ritwika): investigate more possible internal operations as result of optimization rules*/
      case SEARCH:

        // Determine if we can use the optimized is_in codepath. We can take this codepath if
        // the second argument (the elements to search for) consists of exclusively discrete values,
        // and
        // we are not inside a case statement.
        // We can't use the optimized implementation within a case statement due to the Sarg array
        // being
        // lowered as a global, and we can't lower globals within a case statement, as
        // bodosql_case_placeholder
        // doesn't have the doesn't have the same global state as
        // the rest of the main generated code
        SqlTypeName search_val_type = node.getOperands().get(1).getType().getSqlTypeName();
        // TODO: add testing/support for sql types other than string/int:
        // https://bodo.atlassian.net/browse/BE-4046
        // TODO: allow lowering globals within case statements in BodoSQL:
        //
        boolean can_use_isin_codegen =
            !isSingleRow
                & (SqlTypeName.STRING_TYPES.contains(search_val_type)
                    || SqlTypeName.INT_TYPES.contains(search_val_type));

        RexLiteral sargNode = (RexLiteral) node.getOperands().get(1);
        Sarg sargVal = (Sarg) sargNode.getValue();
        Iterator<Range> iter = sargVal.rangeSet.asRanges().iterator();
        // We expect the range to have at least one value,
        // otherwise, this search should have been optimized out
        assert iter.hasNext() : "Internal Error: search Sarg literal had no elements";
        while (iter.hasNext() && can_use_isin_codegen) {
          Range curRange = iter.next();
          // Assert that each element of the range is scalar.
          if (!(curRange.hasLowerBound()
              && curRange.hasUpperBound()
              && curRange.upperEndpoint() == curRange.lowerEndpoint())) {
            can_use_isin_codegen = false;
          }
        }

        if (can_use_isin_codegen) {
          // use the isin array kernel in the case
          // that the second argument does consists of
          // exclusively discrete values
          RexNodeVisitorInfo arg0Info =
              visitRexNode(node.getOperands().get(0), colNames, id, inputVar, isSingleRow, ctx);
          RexNodeVisitorInfo arg1Info =
              visitRexNode(sargNode, colNames, id, inputVar, isSingleRow, ctx);

          expr =
              "bodo.libs.bodosql_array_kernels.is_in("
                  + arg0Info.getExprCode()
                  + ", "
                  + arg1Info.getExprCode()
                  + ")";
          result = new RexNodeVisitorInfo(expr);
        } else {
          // Fallback to generating individual checks
          // in the case that the second argument does not consist of
          // exclusively discrete values

          // Lookup the expanded nodes previously generated
          result =
              visitRexNode(
                  searchMap.get(ExprTypeVisitor.generateRexNodeKey(node, id)),
                  colNames,
                  id,
                  inputVar,
                  isSingleRow,
                  ctx);
        }

        break;
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Internal Operator");
    }
    return result;
  }

  /**
   * Visitor for RexCall with Binary Operator
   *
   * @param operation Binary operator RexCall being visited
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
  private RexNodeVisitorInfo visitBinOpScan(
      RexCall operation,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {
    List<String> args = new ArrayList<>();
    List<BodoSQLExprType.ExprType> exprTypes = new ArrayList<>();
    SqlOperator binOp = operation.getOperator();
    // Store the argument types for TZ-Aware data
    List<RelDataType> argDataTypes = new ArrayList<>();
    for (RexNode operand : operation.operands) {
      RexNodeVisitorInfo info = visitRexNode(operand, colNames, id, inputVar, isSingleRow, ctx);
      args.add(info.getExprCode());
      exprTypes.add(exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(operand, id)));
      argDataTypes.add(operand.getType());
    }
    if (binOp.getKind() == SqlKind.OTHER && binOp.getName().equals("||")) {
      // Support the concat operator by using the concat array kernel.
      List<RexNodeVisitorInfo> argInfos = new ArrayList<>();
      for (String arg : args) {
        argInfos.add(new RexNodeVisitorInfo(arg));
      }
      return generateConcatFnInfo(argInfos);
    }
    String codeGen = generateBinOpCode(args, binOp, argDataTypes);
    return new RexNodeVisitorInfo(codeGen);
  }

  /**
   * Visitor for RexCall with a Postfix Operator
   *
   * @param operation Postfix operator RexCall being visited
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
  private RexNodeVisitorInfo visitPostfixOpScan(
      RexCall operation,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {
    RexNodeVisitorInfo seriesOp =
        visitRexNode(operation.operands.get(0), colNames, id, inputVar, isSingleRow, ctx);
    String codeExpr = generatePostfixOpCode(seriesOp.getExprCode(), operation.getOperator());
    return new RexNodeVisitorInfo(codeExpr);
  }

  /**
   * Visitor for RexCall with a Prefix Operator
   *
   * @param operation Prefix operator RexCall being visited
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
  private RexNodeVisitorInfo visitPrefixOpScan(
      RexCall operation,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {
    RexNodeVisitorInfo seriesOp =
        visitRexNode(operation.operands.get(0), colNames, id, inputVar, isSingleRow, ctx);
    String codeExpr = generatePrefixOpCode(seriesOp.getExprCode(), operation.getOperator());
    return new RexNodeVisitorInfo(codeExpr);
  }

  /**
   * Visitor for RexCall which needs casting to proper data type
   *
   * @param operation Casting RexCall being visited
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
  public RexNodeVisitorInfo visitCastScan(
      RexCall operation,
      List<String> colNames,
      int id,
      String inputVar,
      boolean isSingleRow,
      BodoCtx ctx) {
    RelDataType inputType = operation.operands.get(0).getType();
    RelDataType outputType = operation.getType();

    boolean outputScalar =
        isSingleRow
            || (exprTypesMap.get(ExprTypeVisitor.generateRexNodeKey(operation, id))
                == BodoSQLExprType.ExprType.SCALAR);
    RexNodeVisitorInfo child =
        visitRexNode(operation.operands.get(0), colNames, id, inputVar, isSingleRow, ctx);
    String exprCode = generateCastCode(child.getExprCode(), inputType, outputType, outputScalar);
    return new RexNodeVisitorInfo(exprCode);
  }

  /**
   * Visitor for RexInputRef : Input References in relational expressions.
   *
   * @param node RexInputRef being visited
   * @param colNames List of colNames used in the relational expression
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
  public RexNodeVisitorInfo visitInputRef(
      RexInputRef node, List<String> colNames, String inputVar, boolean isSingleRow, BodoCtx ctx) {
    String refValue =
        String.format(
            "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(%s, %d)", inputVar, node.getIndex());
    if (isSingleRow) {
      ctx.getUsedColumns().add(node.getIndex());
      // NOTE: Codegen for bodosql_case_placeholder() expects column value accesses
      // (e.g. bodo.utils.indexing.scalar_optional_getitem(T1_1, i))
      refValue =
          "bodo.utils.indexing.scalar_optional_getitem("
              + inputVar
              + "_"
              + node.getIndex()
              + ", i)";
    }
    return new RexNodeVisitorInfo(refValue);
  }

  /**
   * Visitor for RexNamedParam : Parameters to Bodo Functions for scalars.
   *
   * @param node RexNamedParam being visited
   * @return RexNodeVisitorInfo containing the new RexNamedParam name and the code generated for the
   *     RexNamedParam.
   */
  public RexNodeVisitorInfo visitRexNamedParam(RexNamedParam node, BodoCtx ctx) {
    String paramName = node.getName();
    // We just return the node name because that should match the input variable name
    ctx.getNamedParams().add(paramName);
    return new RexNodeVisitorInfo(paramName);
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
    String literal = generateLiteralCode(node, isSingleRow, this);
    return new RexNodeVisitorInfo(literal);
  }

  /**
   * Visitor for Table Scan.
   *
   * @param node TableScan node being visited
   * @param canLoadFromCache Can we load the variable from cache? This is set to False if we have a
   *     filter that wasn't previously cached to enable filter pushdown.
   */
  public void visitTableScan(TableScan node, boolean canLoadFromCache) {

    if (!(node instanceof LogicalTableScan || node instanceof LogicalTargetTableScan)) {
      throw new BodoSQLCodegenException(
          "Internal error: unsupported tableScan node generated:" + node.toString());
    }
    boolean isTargetTableScan = node instanceof LogicalTargetTableScan;

    // Determine how many \n characters have appears. This indicates the line
    // in which to insert the IO for table scans.
    String outVar = this.genDfVar();
    List<String> columnNames = node.getRowType().getFieldNames();
    int nodeId = node.getId();
    if (canLoadFromCache && this.isNodeCached(node)) {
      Pair<String, List<String>> cacheInfo = this.varCache.get(nodeId);
      columnNames = cacheInfo.getValue();
      this.generatedCode.append(String.format("  %s = %s\n", outVar, cacheInfo.getKey()));
    } else {
      RelOptTableImpl relTable = (RelOptTableImpl) node.getTable();
      BodoSqlTable table = (BodoSqlTable) relTable.table();
      String readCode;
      String readAssign;
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
        readAssign =
            String.format(
                "  %s, %s, %s = %s\n",
                outVar, icebergFileListVarName, icebergSnapshotIDName, readCode);
        targetTableDf = outVar;
      } else {
        readCode = table.generateReadCode();
        readAssign = String.format("  %s = %s\n", outVar, readCode);
      }

      boolean readUsesIO = table.readRequiresIO();
      if (this.verboseLevel >= 1 && readUsesIO) {
        // If the user has set verbose level >= 1 and there is IO, we generate
        // timers and print the read time.
        String timerVar = this.genGenericTempVar();
        String tableName = table.getName();
        this.generatedCode.append(String.format("  %s = time.time()\n", timerVar));
        this.generatedCode.append(readAssign);
        // Generate the Debug message.
        this.generatedCode.append(
            String.format(
                "  bodo.user_logging.log_message('IO TIMING', f'Finished reading table %s in"
                    + " {time.time() - %s} seconds')\n",
                tableName, timerVar));
      } else {
        this.generatedCode.append(readAssign);
      }

      String castExpr = table.generateReadCastCode(outVar);
      if (castExpr != "") {
        this.generatedCode.append(String.format("  %s = %s\n", outVar, castExpr));
      }
      if (!isTargetTableScan) {
        // Add the table to cached values. We only support this for regular
        // tables and not the target table in merge into.
        this.varCache.put(node.getId(), new Pair<>(outVar, columnNames));
      }
    }

    columnNamesStack.push(columnNames);
    varGenStack.push(outVar);
  }

  /**
   * Visitor for Join: Supports JOIN clause in SQL.
   *
   * @param node join node being visited
   */
  public void visitJoin(Join node) {
    /* get left/right tables */
    String outVar = this.genDfVar();
    int nodeId = node.getId();
    List<String> outputColNames = node.getRowType().getFieldNames();
    if (this.isNodeCached(node)) {
      Pair<String, List<String>> cacheInfo = this.varCache.get(nodeId);
      this.generatedCode.append(String.format("  %s = %s\n", outVar, cacheInfo.getKey()));
      outputColNames = cacheInfo.getValue();
    } else {
      this.visit(node.getLeft(), 0, node);
      List<String> leftColNames = columnNamesStack.pop();
      String leftTable = varGenStack.pop();
      this.visit(node.getRight(), 1, node);
      List<String> rightColNames = columnNamesStack.pop();
      String rightTable = varGenStack.pop();
      this.genRelnodeTimerStart(node);

      RexNode cond = node.getCondition();

      /** Generate the expression for the join condition in a format Bodo supports. */
      HashSet<String> mergeCols = new HashSet<>();
      Pair<String, Boolean> joinCondInfo =
          visitJoinCond(cond, leftColNames, rightColNames, mergeCols);
      String joinCond = joinCondInfo.getKey();

      /* extract join type */
      String joinType = node.getJoinType().lowerName;

      // a join without any conditions is a cross join (how="cross" in pd.merge)
      if (joinCond.equals("True")) {
        joinType = "cross";
      }

      String joinCode =
          generateJoinCode(
              outVar,
              joinType,
              rightTable,
              leftTable,
              rightColNames,
              leftColNames,
              outputColNames,
              joinCond,
              mergeCols);
      this.generatedCode.append(joinCode);
      this.varCache.put(nodeId, new Pair<>(outVar, outputColNames));
      this.genRelnodeTimerStop(node);
    }
    columnNamesStack.push(outputColNames);
    varGenStack.push(outVar);
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
      if (!this.varCache.containsKey(n.getId())) {
        return false;
      }
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
      String msgString =
          new StringBuilder(getBodoIndent())
              .append("t_" + node_id)
              .append(" = time.time()\n")
              .toString();
      this.generatedCode.append(msgString);
    }
  }

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

      this.generatedCode
          .append(getBodoIndent())
          .append("node_string = ")
          .append("'''")
          .append(nodeStr)
          .append("'''\n");
      String msgString =
          new StringBuilder(getBodoIndent())
              .append(
                  "bodo.user_logging.log_message('RELNODE_TIMING', f'''Execution time for RelNode"
                      + " {node_string}: {time.time() - t_")
              .append(node_id)
              .append("}''')\n")
              .toString();
      this.generatedCode.append(msgString);
    }
  }
}
