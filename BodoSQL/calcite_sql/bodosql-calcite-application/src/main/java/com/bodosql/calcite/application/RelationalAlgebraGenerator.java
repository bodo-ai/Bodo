package com.bodosql.calcite.application;

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel;
import com.bodosql.calcite.adapter.bodo.BodoUtilKt;
import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem;
import com.bodosql.calcite.application.utils.RelCostAndMetaDataWriter;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.ddl.DDLExecutionResult;
import com.bodosql.calcite.ddl.GenerateDDLTypes;
import com.bodosql.calcite.prepare.AbstractPlannerImpl;
import com.bodosql.calcite.prepare.PlannerImpl;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.RootSchema;
import com.bodosql.calcite.table.ColumnDataTypeInfo;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlMerge;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.pretty.SqlPrettyWriter;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.tools.RelConversionException;
import org.apache.calcite.tools.ValidationException;
import org.apache.commons.lang3.tuple.Pair;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 *
 * <h1>Generate Relational Algebra</h1>
 *
 * The purpose of this class is to hold the planner, the program, and the configuration for reuse
 * based on a schema that is provided. It can then take sql and convert it to relational algebra.
 *
 * <p>TODO(jsternberg): This class needs a future refactor, but it's also closely tied to the Python
 * code and how it interacts with the Java code. That means that methods defined here can be invoked
 * from the Python code without having an equivalent Java invocation.
 *
 * <p>In order to simplify modifications of this class, the <b>only</b> methods that should be
 * public are ones that Python directly interacts with. Everything else should be private. All
 * public methods should be treated as an API contract and should not be broken or modified without
 * due diligence to transition outside callers.
 *
 * <p>The reason why this class needs a refactor is because this class has a decent amount of
 * mutable state and poorly defined interactions with outside callers. It's not thread-safe and it
 * exposes some APIs for testing that shouldn't be external such as the specific way plans are
 * constructed or processed. It also keeps and handles resources like database connections but
 * doesn't clearly label the lifetimes and ownerships of these resources.
 *
 * @author Bodo
 * @version 1.0
 * @since 2018-10-31
 */
public class RelationalAlgebraGenerator {
  static final Logger LOGGER = LoggerFactory.getLogger(RelationalAlgebraGenerator.class);

  /** Planner that converts sql to relational algebra. */
  private AbstractPlannerImpl planner;

  /**
   * Stores the output of parsing the given SQL query. This is done to allow separating parsing from
   * other steps.
   */
  private SqlNode parseNode = null;

  /*
  Hashmap containing globals that need to be lowered into the resulting func_text. Used for lowering
  metadata types to improve compilation speed. Populated by the Bodo visitor class.
  */
  private HashMap<String, String> loweredGlobalVariables;

  /** Store the catalog being used to close any connections after processing a query. */
  private BodoSQLCatalog catalog;

  /** Store the type system being used to access timezone info during Bodo codegen */
  private final RelDataTypeSystem typeSystem;

  /** Should the planner output streaming code. */
  private final boolean isStreaming;

  /** The Bodo verbose level. This is used to control code generated and/or compilation info. */
  private final int verboseLevel;

  /** The Bodo tracing level. This is used to control code generated and/or compilation info. */
  private final int tracingLevel;

  /** The batch size used for streaming. This is configurable for testing purposes. */
  public static int streamingBatchSize = 4096;

  /**
   * The ratio of distinct rows versus total rows required for dictionary encoding. If the ratio is
   * less than or equal to this value, then dictionary encoding is used for string columns.
   */
  public static double READ_DICT_THRESHOLD = 0.5;

  /**
   * Hide credential information in any generated code. This is used for code generated in Python by
   * convert_to_pandas() or Java tests so the generated code can be shared.
   */
  public static boolean hideCredentials = false;

  /** Should we try read Snowflake tables as Iceberg tables? */
  public static boolean enableSnowflakeIcebergTables = false;

  /** Should we read TIMESTAMP_TZ as its own type instead of TIMESTAMP_LTZ */
  public static boolean enableTimestampTz = false;

  /**
   * Should we insert placeholders for operator IDs to minimize codegen changes with respect to plan
   * changes *
   */
  public static boolean hideOperatorIDs = false;

  /**
   * Which sql dialect should Bodo emulate default from? Currently supported modes: SNOWFLAKE
   * (default) and SPARK.
   */
  public static String sqlStyle = "SNOWFLAKE";

  /** Should we use the covering expression approach to cache or only exact matches. */
  public static boolean coveringExpressionCaching = false;

  /** Should we prefetch metadata location for Snowflake-managed Iceberg tables. */
  public static boolean prefetchSFIceberg = false;

  /**
   * Helper method for RelationalAlgebraGenerator constructors to create a SchemaPlus object from a
   * list of BodoSqlSchemas.
   */
  private List<SchemaPlus> setupSchema(
      BiConsumer<SchemaPlus, ImmutableList.Builder<SchemaPlus>> setup) {
    final SchemaPlus root = RootSchema.Companion.createRootSchema(this.catalog);
    ImmutableList.Builder<SchemaPlus> defaults = ImmutableList.builder();
    setup.accept(root, defaults);
    return defaults.build();
  }

  /**
   * Helper method for both RelationalAlgebraGenerator constructors to build and set the Config and
   * the Planner member variables.
   */
  private void setupPlanner(List<SchemaPlus> defaultSchemas, RelDataTypeSystem typeSystem) {
    PlannerImpl.Config config =
        new PlannerImpl.Config(defaultSchemas, typeSystem, sqlStyle, isStreaming);
    try {
      this.planner = new PlannerImpl(config);
    } catch (Exception e) {
      throw new RuntimeException(
          String.format(
              "Internal Error: Unable to store schema in config and/or setup planner for parsing."
                  + " Error message: %s",
              e));
    }
  }

  /**
   * Constructor for the relational algebra generator class. It will take the schema store it in the
   * config and then set up the program for optimizing and the {@link #planner} for parsing.
   */
  public RelationalAlgebraGenerator(
      @Nullable BodoSQLCatalog catalog,
      @NonNull BodoSqlSchema localSchema,
      @NonNull boolean isStreaming,
      @NonNull int verboseLevel,
      @NonNull int tracingLevel,
      @NonNull int streamingBatchSize,
      @NonNull boolean hideCredentials,
      @NonNull boolean enableSnowflakeIcebergTables,
      @NonNull boolean enableTimestampTz,
      @NonNull boolean enableStreamingSort,
      @NonNull boolean enableStreamingSortLimitOffset,
      @NonNull String sqlStyle,
      @NonNull boolean coveringExpressionCaching,
      @NonNull boolean prefetchSFIceberg,
      @Nullable String defaultTz) {
    this.isStreaming = isStreaming;
    this.catalog = catalog;
    this.verboseLevel = verboseLevel;
    this.tracingLevel = tracingLevel;
    this.streamingBatchSize = streamingBatchSize;
    System.setProperty("calcite.default.charset", "UTF-8");
    List<SchemaPlus> defaultSchemas;
    final RelDataTypeSystem typeSystem;
    if (catalog != null) {
      // Set the default schemas for the catalog.
      List<String> catalogDefaultSchema = catalog.getDefaultSchema(0);
      final @Nullable String currentDatabase;
      if (catalogDefaultSchema.isEmpty()) {
        currentDatabase = null;
      } else {
        currentDatabase = catalogDefaultSchema.get(0);
      }
      defaultSchemas =
          setupSchema(
              (root, defaults) -> {
                // Create a schema object with the name of the catalog,
                // and register all the schemas with this catalog as sub-schemas
                // Note that the order of adding to default matters. Earlier
                // elements are given higher priority during resolution.
                // The correct attempted order of resolution should be:
                //     catalog_default_path1.(table_identifier)
                //     catalog_default_path2.(table_identifier)
                //     ...
                //     __BODOLOCAL__.(table_identifier)
                //     (table_identifier) (Note: this case will never yield a match,
                //     as the root schema is currently always empty. This may change
                //     in the future)

                List<SchemaPlus> schemas = new ArrayList();
                int numLevels = catalog.numDefaultSchemaLevels();
                SchemaPlus parent = root;
                for (int i = 0; i < catalog.numDefaultSchemaLevels(); i++) {
                  List<String> schemaNames = catalog.getDefaultSchema(i);
                  // The current default schema API is awkward and needs to be
                  // rewritten. Snowflake allows there to be multiple current
                  // schemas, but this doesn't generalize to other catalogs as
                  // this can lead to diverging paths. We add this check as a
                  // temporary fix and will revisit the API later.
                  // TODO: Fix the API.
                  if ((i + 1) != numLevels && schemaNames.size() > 1) {
                    throw new RuntimeException(
                        String.format(
                            Locale.ROOT,
                            "BodoSQL only supports multiple default schema paths that differ in the"
                                + " last level"));
                  }
                  SchemaPlus newParent = parent;
                  for (int j = schemaNames.size() - 1; j >= 0; j--) {
                    String schemaName = schemaNames.get(j);
                    SchemaPlus newSchema = parent.getSubSchema(schemaName);
                    if (newSchema == null) {
                      throw new RuntimeException(
                          String.format(
                              Locale.ROOT, "Unable to find default schema: %s", schemaName));
                    }
                    schemas.add(newSchema);
                    newParent = newSchema;
                  }
                  parent = newParent;
                }
                // Add the list in reverse order.
                for (int i = schemas.size() - 1; i >= 0; i--) {
                  defaults.add(schemas.get(i));
                }
                // Add the local schema to the list of schemas.
                defaults.add(root.add(localSchema.getName(), localSchema));
              });
      // Create a type system with the correct default Timezone.
      BodoTZInfo tzInfo = catalog.getDefaultTimezone();
      Integer weekStart = catalog.getWeekStart();
      Integer weekOfYearPolicy = catalog.getWeekOfYearPolicy();
      typeSystem =
          new BodoSQLRelDataTypeSystem(
              tzInfo,
              weekStart,
              weekOfYearPolicy,
              new BodoSQLRelDataTypeSystem.CatalogContext(
                  currentDatabase, catalog.getAccountName()),
              enableStreamingSort,
              enableStreamingSortLimitOffset);

    } else {
      // Set the default schema as just based on local schema.
      defaultSchemas =
          setupSchema(
              (root, defaults) -> {
                defaults.add(root.add(localSchema.getName(), localSchema));
              });
      if (defaultTz != null) {
        BodoTZInfo tzInfo = new BodoTZInfo(defaultTz, "str");
        typeSystem =
            new BodoSQLRelDataTypeSystem(
                tzInfo, 0, 0, null, enableStreamingSort, enableStreamingSortLimitOffset);
      } else {
        typeSystem =
            new BodoSQLRelDataTypeSystem(enableStreamingSort, enableStreamingSortLimitOffset);
      }
    }
    this.typeSystem = typeSystem;
    this.sqlStyle = sqlStyle;
    setupPlanner(defaultSchemas, typeSystem);
    this.hideCredentials = hideCredentials;
    this.enableSnowflakeIcebergTables = enableSnowflakeIcebergTables;
    this.enableTimestampTz = enableTimestampTz;
    this.coveringExpressionCaching = coveringExpressionCaching;
    this.prefetchSFIceberg = prefetchSFIceberg;
  }

  // TODO: Determine a better location for this.
  public static final EnumSet<SqlKind> OTHER_NON_COMPUTE_SET =
      EnumSet.of(
          SqlKind.DESCRIBE_TABLE,
          SqlKind.DESCRIBE_SCHEMA,
          SqlKind.DESCRIBE_VIEW,
          SqlKind.SHOW_SCHEMAS,
          SqlKind.SHOW_OBJECTS,
          SqlKind.SHOW_TABLES,
          SqlKind.SHOW_VIEWS,
          SqlKind.SHOW_TBLPROPERTIES);

  /**
   * Return if a SQLKind generates compute. This includes CREATE_TABLE because of CTAS right now.
   *
   * @param kind The SQLKind to check
   * @return True if the SQLKind generates compute.
   */
  public static boolean isComputeKind(SqlKind kind) {
    return (!OTHER_NON_COMPUTE_SET.contains(kind) && !SqlKind.DDL.contains(kind))
        || (kind == SqlKind.CREATE_TABLE);
  }

  /**
   * Calls "RESET" on the current planner and clears any cached state need to compile a single query
   * (but not configuration).
   */
  void reset() {
    this.planner.close();
    this.parseNode = null;
  }

  /**
   * Parses the SQL query into a SQLNode and updates the relational Algebra Generator's state
   *
   * @param sql Query to parse Sets parseNode to the generated SQLNode
   * @throws SqlSyntaxException if the SQL syntax is incorrect.
   */
  void parseQuery(String sql) throws SqlSyntaxException {
    try {
      this.parseNode = planner.parse(sql);
    } catch (SqlParseException e) {
      planner.close();
      throw new SqlSyntaxException(sql, e);
    }
  }

  private SqlNode validateQuery(
      String sql,
      List<ColumnDataTypeInfo> dynamicParamTypes,
      Map<String, ColumnDataTypeInfo> namedParamTypeMap)
      throws SqlSyntaxException, SqlValidationException {
    if (this.parseNode == null) {
      parseQuery(sql);
    }
    SqlNode parseNode = this.parseNode;
    // Clear the parseNode because we will advance the planner
    this.parseNode = null;
    try {
      if (isComputeKind(parseNode.getKind())) {
        return planner.validate(parseNode, dynamicParamTypes, namedParamTypeMap);
      } else {
        // No need to validate DDL statements. We handle
        // them separately in execution.
        return parseNode;
      }
    } catch (ValidationException e) {
      throw new SqlValidationException(sql, e);
    }
  }

  private Pair<RelRoot, Map<Integer, Integer>> sqlToRel(SqlNode validatedSqlNode)
      throws RelConversionException {
    if (!isComputeKind(validatedSqlNode.getKind())) {
      throw new RelConversionException(
          "DDL statements are not supported by the Relational Algebra Generator");
    }
    RelRoot baseResult = planner.rel(validatedSqlNode);
    RelRoot unoptimizedPlan =
        baseResult.withRel(planner.transform(0, planner.getEmptyTraitSet(), baseResult.rel));
    RelTraitSet requiredOutputTraits =
        planner.getEmptyTraitSet().replace(BodoPhysicalRel.CONVENTION);
    RelRoot optimizedPlan =
        unoptimizedPlan.withRel(planner.transform(1, requiredOutputTraits, unoptimizedPlan.rel));
    RelIDToOperatorIDVisitor s = new RelIDToOperatorIDVisitor();
    s.visit(optimizedPlan.rel, 0, null);
    Map<Integer, Integer> idMapping = s.getIDMapping();
    return Pair.of(optimizedPlan, idMapping);
  }

  @VisibleForTesting
  public Pair<RelRoot, Map<Integer, Integer>> getRelationalAlgebra(
      String sql,
      List<ColumnDataTypeInfo> dynamicParamTypes,
      Map<String, ColumnDataTypeInfo> namedParamTypeMap)
      throws SqlSyntaxException, SqlValidationException, RelConversionException {
    try {
      SqlNode validatedSqlNode = validateQuery(sql, dynamicParamTypes, namedParamTypeMap);
      return sqlToRel(validatedSqlNode);
    } finally {
      planner.close();
      // Close any open connections from catalogs
      if (catalog != null) {
        catalog.closeConnections();
      }
    }
  }

  private String getOptimizedPlanStringFromRoot(
      Pair<RelRoot, Map<Integer, Integer>> root, Boolean includeCosts) {
    RelNode newRoot = BodoUtilKt.bodoPhysicalProject(root.getLeft());
    StringWriter sw = new StringWriter();
    RelCostAndMetaDataWriter costWriter =
        new RelCostAndMetaDataWriter(
            new PrintWriter(sw), newRoot, root.getRight(), includeCosts, hideOperatorIDs);
    newRoot.explain(costWriter);
    costWriter.explainCachedNodes();
    return sw.toString();
  }

  /**
   * Generate the DDL representation. Here we just unparse to SQL filling in any default values.
   *
   * @param ddlNode The DDLNode to represent.
   * @return The DDL representation as a string.
   */
  private String getDDLPlanString(SqlNode ddlNode) {
    StringBuilder sb = new StringBuilder();
    SqlPrettyWriter writer =
        new SqlPrettyWriter(SqlPrettyWriter.config().withAlwaysUseParentheses(true), sb);
    ddlNode.unparse(writer, 0, 0);
    return sb.toString();
  }

  private String getDDLPandasString(
      SqlNode ddlNode,
      String originalSQL,
      List<ColumnDataTypeInfo> dynamicParamTypes,
      Map<String, ColumnDataTypeInfo> namedParamTypeMap) {
    // Note: We can't use dynamic params in DDL yet, but we pass the types, so we can generate a
    // cleaner error message.
    List<RelDataType> dynamicTypes =
        dynamicParamTypes.stream()
            .map(x -> x.convertToSqlType(planner.getTypeFactory()))
            .collect(Collectors.toList());
    Map<String, RelDataType> namedParamTypes =
        namedParamTypeMap.entrySet().stream()
            .collect(
                Collectors.toMap(
                    Map.Entry::getKey,
                    x -> x.getValue().convertToSqlType(planner.getTypeFactory())));
    this.loweredGlobalVariables = new HashMap<>();
    BodoCodeGenVisitor codegen =
        new BodoCodeGenVisitor(
            this.loweredGlobalVariables,
            originalSQL,
            this.typeSystem,
            this.verboseLevel,
            this.tracingLevel,
            this.streamingBatchSize,
            dynamicTypes,
            namedParamTypes,
            Map.of(),
            this.hideOperatorIDs,
            this.prefetchSFIceberg);
    codegen.generateDDLCode(ddlNode, new GenerateDDLTypes(this.planner.getTypeFactory()));
    return codegen.getGeneratedCode();
  }

  private String getPandasStringFromPlan(
      Pair<RelRoot, Map<Integer, Integer>> plan,
      String originalSQL,
      List<ColumnDataTypeInfo> dynamicParamTypes,
      Map<String, ColumnDataTypeInfo> namedParamTypeMap) {
    /*
     * HashMap that maps a Calcite Node using a unique identifier for different "values". To do
     * this, we use two components. First, each RelNode comes with a unique id, which This is used
     * to track exprTypes before code generation.
     */
    List<RelDataType> dynamicTypes =
        dynamicParamTypes.stream()
            .map(x -> x.convertToSqlType(planner.getTypeFactory()))
            .collect(Collectors.toList());
    Map<String, RelDataType> namedParamTypes =
        namedParamTypeMap.entrySet().stream()
            .collect(
                Collectors.toMap(
                    Map.Entry::getKey,
                    x -> x.getValue().convertToSqlType(planner.getTypeFactory())));
    RelNode rel = BodoUtilKt.bodoPhysicalProject(plan.getLeft());
    // Create a mapping for the new root/newly created nodes - this is a bit of a hack, and long
    // term we probably want
    // something that's part of the RelNode itself instead of an auxiliary map to make this safer.
    RelIDToOperatorIDVisitor v = new RelIDToOperatorIDVisitor(new HashMap<>(plan.getRight()));
    v.visit(rel, 0, null);

    this.loweredGlobalVariables = new HashMap<>();
    BodoCodeGenVisitor codegen =
        new BodoCodeGenVisitor(
            this.loweredGlobalVariables,
            originalSQL,
            this.typeSystem,
            this.verboseLevel,
            this.tracingLevel,
            this.streamingBatchSize,
            dynamicTypes,
            namedParamTypes,
            v.getIDMapping(),
            this.hideOperatorIDs,
            this.prefetchSFIceberg);
    codegen.go(rel);
    return codegen.getGeneratedCode();
  }

  Map<String, String> getLoweredGlobalVariables() {
    return this.loweredGlobalVariables;
  }

  // ~~~~~~~~~~~~~Called by the Python Entry Points~~~~~~~~~~~~~~

  /**
   * Get the optimized plan string for the given SQL query.
   *
   * @param sql The SQL query to process.
   * @param includeCosts Should the costs be included in the plan string.
   * @param dynamicParamTypes The dynamic parameter types.
   * @param namedParamTypeMap The named parameter types.
   * @return The optimized plan string for the given SQL query.
   * @throws Exception If an error occurs while processing the SQL query.
   */
  String getOptimizedPlanString(
      String sql,
      Boolean includeCosts,
      List<ColumnDataTypeInfo> dynamicParamTypes,
      Map<String, ColumnDataTypeInfo> namedParamTypeMap)
      throws Exception {
    try {
      SqlNode validatedSqlNode = validateQuery(sql, dynamicParamTypes, namedParamTypeMap);
      if (isComputeKind(validatedSqlNode.getKind())) {
        Pair<RelRoot, Map<Integer, Integer>> root = sqlToRel(validatedSqlNode);
        return getOptimizedPlanStringFromRoot(root, includeCosts);
      } else {
        return getDDLPlanString(validatedSqlNode);
      }
    } finally {
      planner.close();
      // Close any open connections from catalogs
      if (catalog != null) {
        catalog.closeConnections();
      }
    }
  }

  /**
   * Return the optimized plan generated for the given SQL query.
   *
   * @param sql The SQL query to process.
   * @param dynamicParamTypes The dynamic parameter types.
   * @param namedParamTypeMap The named parameter types.
   * @return The optimized plan generated for the given SQL query.
   * @throws Exception If an error occurs while processing the SQL query.
   */
  RelNode getOptimizedPlan(
      @NonNull String sql,
      @NonNull List<ColumnDataTypeInfo> dynamicParamTypes,
      @NonNull Map<String, ColumnDataTypeInfo> namedParamTypeMap)
      throws Exception {
    try {
      SqlNode validatedSqlNode = validateQuery(sql, dynamicParamTypes, namedParamTypeMap);
      Pair<RelRoot, Map<Integer, Integer>> optimizedPlan = sqlToRel(validatedSqlNode);
      return BodoUtilKt.bodoPhysicalProject(optimizedPlan.getLeft());
    } finally {
      planner.close();
      // Close any open connections from catalogs
      if (catalog != null) {
        catalog.closeConnections();
      }
    }
  }

  /**
   * Get the Pandas code and the optimized plan string for the given SQL query.
   *
   * @param sql The SQL query to process.
   * @param includeCosts Should the costs be included in the plan string.
   * @param dynamicParamTypes The dynamic parameter types.
   * @param namedParamTypeMap The named parameter types.
   * @return The Pandas code and the optimized plan string for the given SQL query.
   * @throws Exception If an error occurs while processing the SQL query.
   */
  CodePlanPair getPandasAndPlanString(
      String sql,
      boolean includeCosts,
      List<ColumnDataTypeInfo> dynamicParamTypes,
      Map<String, ColumnDataTypeInfo> namedParamTypeMap)
      throws Exception {
    try {
      SqlNode validatedSqlNode = validateQuery(sql, dynamicParamTypes, namedParamTypeMap);
      if (isComputeKind(validatedSqlNode.getKind())) {
        Pair<RelRoot, Map<Integer, Integer>> optimizedPlan = sqlToRel(validatedSqlNode);
        String pandasString =
            getPandasStringFromPlan(optimizedPlan, sql, dynamicParamTypes, namedParamTypeMap);
        String planString = getOptimizedPlanStringFromRoot(optimizedPlan, includeCosts);
        return new CodePlanPair(pandasString, planString);
      } else {
        String pandasString =
            getDDLPandasString(validatedSqlNode, sql, dynamicParamTypes, namedParamTypeMap);
        String planString = getDDLPlanString(validatedSqlNode);
        return new CodePlanPair(pandasString, planString);
      }
    } finally {
      planner.close();
      // Close any open connections from catalogs
      if (catalog != null) {
        catalog.closeConnections();
      }
    }
  }

  /**
   * Return the Python code generated for the given SQL query.
   *
   * @param sql The SQL query to process.
   * @param dynamicParamTypes The dynamic parameter types.
   * @param namedParamTypeMap The named parameter types.
   * @return The Python code generated for the given SQL query.
   * @throws Exception If an error occurs while processing the SQL query.
   */
  String getPandasString(
      @NonNull String sql,
      @NonNull List<ColumnDataTypeInfo> dynamicParamTypes,
      @NonNull Map<String, ColumnDataTypeInfo> namedParamTypeMap)
      throws Exception {
    try {
      SqlNode validatedSqlNode = validateQuery(sql, dynamicParamTypes, namedParamTypeMap);
      if (isComputeKind(validatedSqlNode.getKind())) {
        Pair<RelRoot, Map<Integer, Integer>> optimizedPlan = sqlToRel(validatedSqlNode);
        return getPandasStringFromPlan(optimizedPlan, sql, dynamicParamTypes, namedParamTypeMap);
      } else {
        return getDDLPandasString(validatedSqlNode, sql, dynamicParamTypes, namedParamTypeMap);
      }
    } finally {
      planner.close();
      // Close any open connections from catalogs
      if (catalog != null) {
        catalog.closeConnections();
      }
    }
  }

  /**
   * Determine the "type" of write produced by this node. The write operation is always assumed to
   * be the top level of the parsed query. It returns the name of operation in question to enable
   * passing the correct write API to the table.
   *
   * <p>Currently supported write types: "MERGE": Merge into "INSERT": Insert Into
   *
   * @param sql The SQL query to parse.
   * @return A string representing the type of write.
   */
  String getWriteType(String sql) throws Exception {
    // Parse the query if we haven't already
    if (this.parseNode == null) {
      parseQuery(sql);
    }
    if (this.parseNode instanceof SqlMerge) {
      return "MERGE";
    } else {
      // Default to insert into for the write type.
      // If there is no write then the return value
      // doesn't matter.
      return "INSERT";
    }
  }

  /**
   * Execute the given DDL statement in an interpreted manner. This assumes/requires that sql is a
   * DDL statement, which should have already been checked.
   *
   * @param sql The DDL statement to execute
   * @return The result of the DDL execution.
   */
  DDLExecutionResult executeDDL(String sql) throws Exception {
    try {
      // DDL doesn't support dynamic or named parameters at this time.
      SqlNode validatedSqlNode = validateQuery(sql, List.of(), Map.of());
      if (RelationalAlgebraGenerator.isComputeKind(validatedSqlNode.getKind())) {
        throw new RuntimeException("Only DDL statements are supported by executeDDL");
      }
      return planner.executeDDL(validatedSqlNode);
    } finally {
      planner.close();
      // Close any open connections from catalogs
      if (catalog != null) {
        catalog.closeConnections();
      }
    }
  }

  /**
   * Determine if the active query is a DDL query that is not treated like compute (not CTAS).
   *
   * @return Is the query DDL?
   */
  boolean isDDLProcessedQuery() {
    if (this.parseNode == null) {
      throw new RuntimeException("No SQL query has been parsed yet. Cannot determine query type");
    }
    return !isComputeKind(this.parseNode.getKind());
  }
}
