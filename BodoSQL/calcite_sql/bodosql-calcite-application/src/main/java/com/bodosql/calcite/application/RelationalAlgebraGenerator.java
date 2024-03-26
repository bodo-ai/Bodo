package com.bodosql.calcite.application;

import com.bodosql.calcite.adapter.pandas.PandasRel;
import com.bodosql.calcite.adapter.pandas.PandasUtilKt;
import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem;
import com.bodosql.calcite.application.utils.RelCostAndMetaDataWriter;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.prepare.PlannerImpl;
import com.bodosql.calcite.prepare.PlannerType;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.RootSchema;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.function.BiConsumer;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlMerge;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.tools.Planner;
import org.apache.calcite.tools.RelConversionException;
import org.apache.calcite.tools.ValidationException;
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
  private Planner planner;

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
  private void setupPlanner(
      List<SchemaPlus> defaultSchemas, String namedParamTableName, RelDataTypeSystem typeSystem) {
    PlannerImpl.Config config =
        new PlannerImpl.Config(defaultSchemas, typeSystem, namedParamTableName, plannerType);
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
      String namedParamTableName,
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
    setupPlanner(defaultSchemas, namedParamTableName, typeSystem);
    this.hideCredentials = hideCredentials;
    this.enableSnowflakeIcebergTables = enableSnowflakeIcebergTables;
    this.enableTimestampTz = enableTimestampTz;
    this.enableRuntimeJoinFilters = enableRuntimeJoinFilters;
  }

  /** Constructor for the relational algebra generator class that takes in the default timezone. */
  public RelationalAlgebraGenerator(
      BodoSqlSchema localSchema,
      String namedParamTableName,
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
    setupPlanner(defaultSchemas, namedParamTableName, typeSystem);
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
      String namedParamTableName,
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
    setupPlanner(defaultSchemas, namedParamTableName, typeSystem);
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

  private SqlNode validateQuery(String sql) throws SqlSyntaxException, SqlValidationException {
    if (this.parseNode == null) {
      parseQuery(sql);
    }
    SqlNode parseNode = this.parseNode;
    // Clear the parseNode because we will advance the planner
    this.parseNode = null;
    try {
      return planner.validate(parseNode);
    } catch (ValidationException e) {
      planner.close();
      throw new SqlValidationException(sql, e);
    }
  }

  private RelRoot getNonOptimizedRelationalAlgebra(String sql, boolean closePlanner)
      throws SqlSyntaxException, SqlValidationException, RelConversionException {
    SqlNode validatedSqlNode = validateQuery(sql);
    RelRoot result = planner.rel(validatedSqlNode);
    result = result.withRel(planner.transform(0, planner.getEmptyTraitSet(), result.rel));
    if (closePlanner) {
      // TODO(jsternberg): Rework this logic because these are some incredibly leaky abstractions.
      // We won't be doing optimizations so transform the relational algebra to use the pandas nodes
      // now. This is a temporary thing while we transition to using the volcano planner.
      RelTraitSet requiredOutputTraits = planner.getEmptyTraitSet().replace(PandasRel.CONVENTION);
      result = result.withRel(planner.transform(2, requiredOutputTraits, result.rel));
      planner.close();
    }
    // Close any open connections from catalogs
    if (catalog != null) {
      catalog.closeConnections();
    }
    return result;
  }

  private RelRoot getOptimizedRelationalAlgebra(RelRoot nonOptimizedPlan)
      throws RelConversionException {
    RelTraitSet requiredOutputTraits = planner.getEmptyTraitSet().replace(PandasRel.CONVENTION);
    RelRoot optimizedPlan =
        nonOptimizedPlan.withRel(planner.transform(1, requiredOutputTraits, nonOptimizedPlan.rel));
    planner.close();
    return optimizedPlan;
  }

  /**
   * Takes a sql statement and converts it into an optimized relational algebra node. The result of
   * this function is a logical plan that has been optimized using a rule based optimizer.
   *
   * @param sql a string sql query that is to be parsed, converted into relational algebra, then
   *     optimized
   * @return a RelNode which contains the relational algebra tree generated from the sql statement
   *     provided after an optimization step has been completed.
   * @throws SqlSyntaxException, SqlValidationException, RelConversionException
   */
  @VisibleForTesting
  public RelRoot getRelationalAlgebra(String sql, boolean performOptimizations)
      throws SqlSyntaxException, SqlValidationException, RelConversionException {
    RelRoot nonOptimizedPlan = getNonOptimizedRelationalAlgebra(sql, !performOptimizations);
    if (!performOptimizations) {
      return nonOptimizedPlan;
    }
    return getOptimizedRelationalAlgebra(nonOptimizedPlan);
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

  public String getPandasString(String sql, boolean debugDeltaTable) throws Exception {
    RelRoot optimizedPlan = getRelationalAlgebra(sql, true);
    return getPandasStringFromPlan(optimizedPlan, sql, debugDeltaTable);
  }

  // Default debugDeltaTable to false
  public String getPandasString(String sql) throws Exception {
    return getPandasString(sql, false);
  }

  public String getPandasStringUnoptimized(String sql, boolean debugDeltaTable) throws Exception {
    RelRoot unOptimizedPlan = getNonOptimizedRelationalAlgebra(sql, true);
    return getPandasStringFromPlan(unOptimizedPlan, sql, debugDeltaTable);
  }

  // Default debugDeltaTable to false
  public String getPandasStringUnoptimized(String sql) throws Exception {
    return getPandasStringUnoptimized(sql, false);
  }

  private String getPandasStringFromPlan(
      RelRoot plan, String originalSQL, boolean debugDeltaTable) {
    /**
     * HashMap that maps a Calcite Node using a unique identifier for different "values". To do
     * this, we use two components. First, each RelNode comes with a unique id, which This is used
     * to track exprTypes before code generation.
     */
    RelNode rel = PandasUtilKt.pandasProject(plan);
    this.loweredGlobalVariables = new HashMap<>();
    PandasCodeGenVisitor codegen =
        new PandasCodeGenVisitor(
            this.loweredGlobalVariables,
            originalSQL,
            this.typeSystem,
            debugDeltaTable,
            this.verboseLevel,
            this.streamingBatchSize);
    codegen.go(rel);
    return codegen.getGeneratedCode();
  }

  public String getOptimizedPlanString(String sql) throws Exception {
    return getOptimizedPlanString(sql, false);
  }

  public String getOptimizedPlanString(String sql, Boolean includeCosts) throws Exception {
    RelRoot root = getRelationalAlgebra(sql, true);
    RelNode newRoot = PandasUtilKt.pandasProject(root);
    if (includeCosts) {
      StringWriter sw = new StringWriter();
      com.bodosql.calcite.application.utils.RelCostAndMetaDataWriter costWriter =
          new RelCostAndMetaDataWriter(new PrintWriter(sw), newRoot);
      newRoot.explain(costWriter);
      return sw.toString();
    } else {
      return RelOptUtil.toString(newRoot);
    }
  }

  public String getUnoptimizedPlanString(String sql) throws Exception {
    RelRoot root = getRelationalAlgebra(sql, false);
    return RelOptUtil.toString(PandasUtilKt.pandasProject(root));
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
}
