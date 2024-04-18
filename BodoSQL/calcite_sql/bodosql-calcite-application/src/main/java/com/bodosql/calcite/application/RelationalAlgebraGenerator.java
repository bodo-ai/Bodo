package com.bodosql.calcite.application;

import com.bodosql.calcite.adapter.pandas.PandasRel;
import com.bodosql.calcite.adapter.pandas.PandasUtilKt;
import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem;
import com.bodosql.calcite.application.utils.RelCostAndMetaDataWriter;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.ddl.DDLExecutionResult;
import com.bodosql.calcite.ddl.GenerateDDLTypes;
import com.bodosql.calcite.prepare.AbstractPlannerImpl;
import com.bodosql.calcite.prepare.PlannerImpl;
import com.bodosql.calcite.prepare.PlannerType;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.RootSchema;
import com.bodosql.calcite.table.ColumnDataTypeInfo;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.stream.Collectors;
import org.apache.calcite.plan.RelOptUtil;
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
  metadata types to improve compilation speed. Populated by the pandas visitor class.
  */
  private HashMap<String, String> loweredGlobalVariables;

  /** Store the catalog being used to close any connections after processing a query. */
  private BodoSQLCatalog catalog;

  /** Store the typesystem being used to access timezone info during Pandas codegen */
  private final RelDataTypeSystem typeSystem;

  /** Which planner should be utilized. */
  private final PlannerType plannerType;

  /** The Bodo verbose level. This is used to control code generated and/or compilation info. */
  private final int verboseLevel;

  /** The batch size used for streaming. This is configurable for testing purposes. */
  private final int streamingBatchSize;

  /**
   * Hide credential information in any generated code. This is used for code generated in Python by
   * convert_to_pandas() or Java tests so the generated code can be shared.
   */
  public static boolean hideCredentials = false;

  /** Should we try read Snowflake tables as Iceberg tables? */
  public static boolean enableSnowflakeIcebergTables = false;

  /** Should we read TIMESTAMP_TZ as its own type instead of TIMESTAMP_LTZ */
  public static boolean enableTimestampTz = false;

  /** Should we enabled planner nodes to insert runtime filters for Joins */
  public static boolean enableRuntimeJoinFilters = false;

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
    PlannerImpl.Config config = new PlannerImpl.Config(defaultSchemas, typeSystem, plannerType);
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

  public static final int VOLCANO_PLANNER = 0;
  public static final int STREAMING_PLANNER = 1;

  /**
   * Constructor for the relational algebra generator class. It will take the schema store it in the
   * config and then set up the program for optimizing and the {@link #planner} for parsing.
   *
   * @param localSchema This is the schema which contains any of our local tables.
   */
  public RelationalAlgebraGenerator(
      BodoSqlSchema localSchema,
      int plannerType,
      int verboseLevel,
      int streamingBatchSize,
      boolean hideCredentials,
      boolean enableSnowflakeIcebergTables,
      boolean enableTimestampTz,
      boolean enableRuntimeJoinFilters) {
    this.catalog = null;
    this.plannerType = choosePlannerType(plannerType);
    this.verboseLevel = verboseLevel;
    this.streamingBatchSize = streamingBatchSize;
    System.setProperty("calcite.default.charset", "UTF-8");
    List<SchemaPlus> defaultSchemas =
        setupSchema(
            (root, defaults) -> {
              defaults.add(root.add(localSchema.getName(), localSchema));
            });
    RelDataTypeSystem typeSystem = new BodoSQLRelDataTypeSystem();
    this.typeSystem = typeSystem;
    setupPlanner(defaultSchemas, typeSystem);
    this.hideCredentials = hideCredentials;
    this.enableSnowflakeIcebergTables = enableSnowflakeIcebergTables;
    this.enableTimestampTz = enableTimestampTz;
    this.enableRuntimeJoinFilters = enableRuntimeJoinFilters;
  }

  /** Constructor for the relational algebra generator class that takes in the default timezone. */
  public RelationalAlgebraGenerator(
      BodoSqlSchema localSchema,
      int plannerType,
      int verboseLevel,
      int streamingBatchSize,
      boolean hideCredentials,
      boolean enableSnowflakeIcebergTables,
      boolean enableTimestampTz,
      boolean enableRuntimeJoinFilters,
      String defaultTz) {
    this.catalog = null;
    this.plannerType = choosePlannerType(plannerType);
    this.verboseLevel = verboseLevel;
    this.streamingBatchSize = streamingBatchSize;
    System.setProperty("calcite.default.charset", "UTF-8");
    List<SchemaPlus> defaultSchemas =
        setupSchema(
            (root, defaults) -> {
              defaults.add(root.add(localSchema.getName(), localSchema));
            });
    BodoTZInfo tzInfo = new BodoTZInfo(defaultTz, "str");
    RelDataTypeSystem typeSystem = new BodoSQLRelDataTypeSystem(tzInfo, 0, 0, null);
    this.typeSystem = typeSystem;
    setupPlanner(defaultSchemas, typeSystem);
    this.hideCredentials = hideCredentials;
    this.enableSnowflakeIcebergTables = enableSnowflakeIcebergTables;
    this.enableTimestampTz = enableTimestampTz;
    this.enableRuntimeJoinFilters = enableRuntimeJoinFilters;
  }

  /**
   * Constructor for the relational algebra generator class that accepts a Catalog and Schema
   * objects. It will take the schema objects in the Catalog as well as the Schema object store it
   * in the schemas and then set up the program for optimizing and the {@link #planner} for parsing.
   */
  public RelationalAlgebraGenerator(
      BodoSQLCatalog catalog,
      BodoSqlSchema localSchema,
      // int is a bad choice for this variable, but we're limited by either
      // forcing py4j to initialize another Java object or use some plain old data
      // that it can use so we're choosing the latter.
      // Something like this can be replaced with a more formal API like GRPC and protobuf.
      int plannerType,
      int verboseLevel,
      int streamingBatchSize,
      boolean hideCredentials,
      boolean enableSnowflakeIcebergTables,
      boolean enableTimestampTz,
      boolean enableRuntimeJoinFilters) {
    this.catalog = catalog;
    this.plannerType = choosePlannerType(plannerType);
    this.verboseLevel = verboseLevel;
    this.streamingBatchSize = streamingBatchSize;
    this.hideCredentials = hideCredentials;
    this.enableSnowflakeIcebergTables = enableSnowflakeIcebergTables;
    this.enableTimestampTz = enableTimestampTz;
    this.enableRuntimeJoinFilters = enableRuntimeJoinFilters;
    System.setProperty("calcite.default.charset", "UTF-8");
    List<String> catalogDefaultSchema = catalog.getDefaultSchema(0);
    final @Nullable String currentDatabase;
    if (catalogDefaultSchema.isEmpty()) {
      currentDatabase = null;
    } else {
      currentDatabase = catalogDefaultSchema.get(0);
    }
    List<SchemaPlus> defaultSchemas =
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
    RelDataTypeSystem typeSystem =
        new BodoSQLRelDataTypeSystem(
            tzInfo,
            weekStart,
            weekOfYearPolicy,
            new BodoSQLRelDataTypeSystem.CatalogContext(currentDatabase, catalog.getAccountName()));
    this.typeSystem = typeSystem;
    setupPlanner(defaultSchemas, typeSystem);
  }

  /**
   * Return if a SQLKind generates compute. This includes CREATE_TABLE because of CTAS right now.
   *
   * @param kind The SQLKind to check
   * @return True if the SQLKind generates compute.
   */
  private boolean isComputeKind(SqlKind kind) {
    return !SqlKind.DDL.contains(kind) || (kind == SqlKind.CREATE_TABLE);
  }

  /**
   * Parses the SQL query into a SQLNode and updates the relational Algebra Generator's state
   *
   * @param sql Query to parse
   * @return The generated SQLNode
   * @throws SqlSyntaxException if the SQL syntax is incorrect.
   */
  public void parseQuery(String sql) throws SqlSyntaxException {
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
    RelTraitSet requiredOutputTraits = planner.getEmptyTraitSet().replace(PandasRel.CONVENTION);
    RelRoot optimizedPlan =
        unoptimizedPlan.withRel(planner.transform(1, requiredOutputTraits, unoptimizedPlan.rel));
    RelIDToOperatorIDVisitor s = new RelIDToOperatorIDVisitor();
    s.visit(optimizedPlan.rel, 0, null);
    Map<Integer, Integer> idMapping = s.getIDMapping();
    return Pair.of(optimizedPlan, idMapping);
  }

  /**
   * Takes a sql statement and converts it into an optimized relational algebra node. The result of
   * this function is a logical plan that has been optimized using a rule based optimizer. This is
   * used when we just want to get the optimized plan and not the pandas code.
   *
   * @param sql a string sql query that is to be parsed, converted into relational algebra, then
   *     optimized
   * @return a RelNode which contains the relational algebra tree generated from the sql statement
   *     provided after an optimization step has been completed.
   * @throws SqlSyntaxException, SqlValidationException, RelConversionException
   */
  @VisibleForTesting
  public Pair<RelRoot, Map<Integer, Integer>> getRelationalAlgebra(String sql)
      throws SqlSyntaxException, SqlValidationException, RelConversionException {
    return getRelationalAlgebra(sql, List.of(), Map.of());
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
    RelNode newRoot = PandasUtilKt.pandasProject(root.getLeft());
    if (includeCosts) {
      StringWriter sw = new StringWriter();
      com.bodosql.calcite.application.utils.RelCostAndMetaDataWriter costWriter =
          new RelCostAndMetaDataWriter(new PrintWriter(sw), newRoot, root.getRight());
      newRoot.explain(costWriter);
      return sw.toString();
    } else {
      return RelOptUtil.toString(newRoot);
    }
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
    PandasCodeGenVisitor codegen =
        new PandasCodeGenVisitor(
            this.loweredGlobalVariables,
            originalSQL,
            this.typeSystem,
            this.verboseLevel,
            this.streamingBatchSize,
            dynamicTypes,
            namedParamTypes,
            Map.of());
    codegen.generateDDLCode(ddlNode, new GenerateDDLTypes(this.planner.getTypeFactory()));
    return codegen.getGeneratedCode();
  }

  private String getPandasStringFromPlan(
      Pair<RelRoot, Map<Integer, Integer>> plan,
      String originalSQL,
      List<ColumnDataTypeInfo> dynamicParamTypes,
      Map<String, ColumnDataTypeInfo> namedParamTypeMap) {
    /**
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
    RelNode rel = PandasUtilKt.pandasProject(plan.getLeft());
    // Create a mapping for the new root/newly created nodes - this is a bit of a hack, and long
    // term we probably want
    // something that's part of the RelNode itself instead of an auxillary map to make this safer.
    RelIDToOperatorIDVisitor v = new RelIDToOperatorIDVisitor(new HashMap<>(plan.getRight()));
    v.visit(rel, 0, null);

    this.loweredGlobalVariables = new HashMap<>();
    PandasCodeGenVisitor codegen =
        new PandasCodeGenVisitor(
            this.loweredGlobalVariables,
            originalSQL,
            this.typeSystem,
            this.verboseLevel,
            this.streamingBatchSize,
            dynamicTypes,
            namedParamTypes,
            v.getIDMapping());
    codegen.go(rel);
    return codegen.getGeneratedCode();
  }

  public HashMap<String, String> getLoweredGlobalVariables() {
    return this.loweredGlobalVariables;
  }

  private static PlannerType choosePlannerType(int plannerType) {
    switch (plannerType) {
      case VOLCANO_PLANNER:
        return PlannerType.VOLCANO;
      case STREAMING_PLANNER:
        return PlannerType.STREAMING;
      default:
        throw new RuntimeException("Unexpected Planner option");
    }
  }

  // ~~~~~~~~~~~~~PYTHON EXPOSED APIS~~~~~~~~~~~~~~
  public String getOptimizedPlanString(String sql, Boolean includeCosts) throws Exception {
    return getOptimizedPlanString(sql, includeCosts, List.of(), Map.of());
  }

  public String getOptimizedPlanString(
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

  public PandasCodeSqlPlanPair getPandasAndPlanString(String sql, boolean includeCosts)
      throws Exception {
    return getPandasAndPlanString(sql, includeCosts, List.of(), Map.of());
  }

  public PandasCodeSqlPlanPair getPandasAndPlanString(
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
        return new PandasCodeSqlPlanPair(pandasString, planString);
      } else {
        String pandasString =
            getDDLPandasString(validatedSqlNode, sql, dynamicParamTypes, namedParamTypeMap);
        String planString = getDDLPlanString(validatedSqlNode);
        return new PandasCodeSqlPlanPair(pandasString, planString);
      }
    } finally {
      planner.close();
      // Close any open connections from catalogs
      if (catalog != null) {
        catalog.closeConnections();
      }
    }
  }

  public String getPandasString(String sql) throws Exception {
    return getPandasString(sql, List.of(), Map.of());
  }

  public String getPandasString(
      String sql,
      List<ColumnDataTypeInfo> dynamicParamTypes,
      Map<String, ColumnDataTypeInfo> namedParamTypeMap)
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
  public String getWriteType(String sql) throws Exception {
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

  public DDLExecutionResult executeDDL(String sql) throws Exception {
    try {
      // DDL doesn't support dynamic or named parameters at this time.
      SqlNode validatedSqlNode = validateQuery(sql, List.of(), Map.of());
      if (!SqlKind.DDL.contains(validatedSqlNode.getKind())) {
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
  public boolean isDDLProcessedQuery() {
    if (this.parseNode == null) {
      throw new RuntimeException("No SQL query has been parsed yet. Cannot determine query type");
    }
    return !isComputeKind(this.parseNode.getKind());
  }
}
