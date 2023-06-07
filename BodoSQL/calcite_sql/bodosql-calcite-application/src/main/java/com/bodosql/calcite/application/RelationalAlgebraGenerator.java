package com.bodosql.calcite.application;

import com.bodosql.calcite.adapter.pandas.PandasRel;
import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.prepare.PlannerImpl;
import com.bodosql.calcite.prepare.PlannerType;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.CatalogSchemaImpl;
import com.google.common.collect.ImmutableList;
import java.sql.Connection;
import java.sql.DriverManager;
import java.util.*;
import java.util.function.BiConsumer;
import javax.annotation.Nullable;
import org.apache.calcite.jdbc.CalciteConnection;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlMerge;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.type.*;
import org.apache.calcite.tools.*;
import org.apache.calcite.util.Pair;
import org.jetbrains.annotations.NotNull;
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
   * Helper method for RelationalAlgebraGenerator constructor to create a Connection object so that
   * SQL queries can be executed within its context.
   */
  public @Nullable CalciteConnection setupCalciteConnection() {
    CalciteConnection calciteConnection = null;
    try {
      Class.forName("org.apache.calcite.jdbc.Driver");

      Properties info = new Properties();
      info.setProperty("lex", "JAVA");
      Connection connection = DriverManager.getConnection("jdbc:calcite:", info);
      calciteConnection = connection.unwrap(CalciteConnection.class);

    } catch (Exception e) {
      throw new RuntimeException(
          String.format(
              "Internal Error: JDBC Driver unable to obtain database connection. Error message: %s",
              e));
    }
    return calciteConnection;
  }

  /**
   * Helper method for RelationalAlgebraGenerator constructors to create a SchemaPlus object from a
   * list of BodoSqlSchemas.
   */
  public List<SchemaPlus> setupSchema(
      @NotNull CalciteConnection calciteConnection,
      BiConsumer<SchemaPlus, ImmutableList.Builder<SchemaPlus>> setup) {
    final SchemaPlus root = calciteConnection.getRootSchema();
    ImmutableList.Builder<SchemaPlus> defaults = ImmutableList.builder();
    setup.accept(root, defaults);
    return defaults.build();
  }

  /**
   * Helper method for both RelationalAlgebraGenerator constructors to build and set the Config and
   * the Planner member variables.
   */
  public void setupPlanner(
      List<SchemaPlus> defaultSchemas,
      String namedParamTableName,
      RelDataTypeSystem typeSystem) {
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

  /**
   * Constructor for the relational algebra generator class. It will take the schema store it in the
   * config and then set up the program for optimizing and the {@link #planner} for parsing.
   *
   * @param newSchema This is the schema which we will be using to validate our query against. This
   *     gets stored in the {@link #planner}
   */
  public RelationalAlgebraGenerator(
      BodoSqlSchema newSchema,
      String namedParamTableName,
      int plannerType,
      int verboseLevel,
      int streamingBatchSize) {
    this.catalog = null;
    this.plannerType = choosePlannerType(plannerType);
    this.verboseLevel = verboseLevel;
    this.streamingBatchSize = streamingBatchSize;
    // Enable/Disable join streaming.
    // TODO: Remove when join code generation is stable.
    System.setProperty("calcite.default.charset", "UTF-8");
    CalciteConnection calciteConnection = setupCalciteConnection();
    List<SchemaPlus> defaultSchemas =
        setupSchema(
            calciteConnection,
            (root, defaults) -> {
              defaults.add(root.add(newSchema.getName(), newSchema));
            });
    RelDataTypeSystem typeSystem = new BodoSQLRelDataTypeSystem();
    this.typeSystem = typeSystem;
    setupPlanner(defaultSchemas, namedParamTableName, typeSystem);
  }

  public static final int HEURISTIC_PLANNER = 1;
  public static final int VOLCANO_PLANNER = 2;
  public static final int STREAMING_PLANNER = 3;

  /**
   * Constructor for the relational algebra generator class that accepts a Catalog and Schema
   * objects. It will take the schema objects in the Catalog as well as the Schema object store it
   * in the schemas and then set up the program for optimizing and the {@link #planner} for parsing.
   */
  public RelationalAlgebraGenerator(
      BodoSQLCatalog catalog,
      BodoSqlSchema newSchema,
      String namedParamTableName,
      // int is a bad choice for this variable, but we're limited by either
      // forcing py4j to initialize another Java object or use some plain old data
      // that it can use so we're choosing the latter.
      // Something like this can be replaced with a more formal API like GRPC and protobuf.
      int plannerType,
      int verboseLevel,
      int streamingBatchSize) {
    this.catalog = catalog;
    this.plannerType = choosePlannerType(plannerType);
    this.verboseLevel = verboseLevel;
    this.streamingBatchSize = streamingBatchSize;
    // Enable/Disable join streaming.
    // TODO: Remove when join code generation is stable.
    System.setProperty("calcite.default.charset", "UTF-8");
    CalciteConnection calciteConnection = setupCalciteConnection();
    List<SchemaPlus> defaultSchemas =
        setupSchema(
            calciteConnection,
            (root, defaults) -> {
              String catalogName = catalog.getCatalogName();
              // Create a schema object with the name of the catalog,
              // and register all the schemas with this catalog as sub-schemas
              // Note that the order of adding to default matters. Earlier
              // elements are given higher priority during resolution.
              // The correct attempted order of resolution should be:
              //     catalog_default_path1.(table_identifier)
              //     catalog_default_path2.(table_identifier)
              //     ...
              //     __bodo_local__.(table_identifier)
              //     (table_identifier) (Note: this case will never yield a match,
              //     as the root schema is currently always empty. This may change
              //     in the future)
              root.add(catalogName, new CatalogSchemaImpl(catalogName, catalog));
              Set<String> remainingSchemaNamesToAdd = catalog.getSchemaNames();

              for (String schemaName : catalog.getDefaultSchema()) {
                SchemaPlus schema =
                    root.getSubSchema(catalogName)
                        .add(schemaName, new CatalogSchemaImpl(schemaName, catalog));
                defaults.add(schema);
                assert schemaName.contains(schemaName)
                    : "Error in RelationalAlgebraGenerator: catalog.getDefaultSchema() returned"
                        + " schema "
                        + schemaName
                        + ", which is not present in catalog.getDefaultSchemaNames()";
                remainingSchemaNamesToAdd.remove(schemaName);
              }
              for (String schemaName : remainingSchemaNamesToAdd) {
                root.getSubSchema(catalogName)
                    .add(schemaName, new CatalogSchemaImpl(schemaName, catalog));
              }
              defaults.add(root.getSubSchema(catalogName));
              defaults.add(root.add(newSchema.getName(), newSchema));
            });

    // Create a type system with the correct default Timezone.
    BodoTZInfo tzInfo = catalog.getDefaultTimezone();
    RelDataTypeSystem typeSystem = new BodoSQLRelDataTypeSystem(tzInfo);
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

  public SqlNode validateQuery(String sql) throws SqlSyntaxException, SqlValidationException {
    if (this.parseNode == null) {
      parseQuery(sql);
    }
    SqlNode parseNode = this.parseNode;
    // Clear the parseNode because we will advance the planner
    this.parseNode = null;
    SqlNode validatedSqlNode;
    try {
      validatedSqlNode = planner.validate(parseNode);
    } catch (ValidationException e) {
      planner.close();
      throw new SqlValidationException(sql, e);
    }
    return validatedSqlNode;
  }

  public Pair<SqlNode, RelDataType> validateQueryAndGetType(String sql)
      throws SqlSyntaxException, SqlValidationException {
    if (this.parseNode == null) {
      parseQuery(sql);
    }
    SqlNode parseNode = this.parseNode;
    // Clear the parseNode because we will advance the planner
    this.parseNode = null;
    Pair<SqlNode, RelDataType> validatedSqlNodeAndType;
    try {
      validatedSqlNodeAndType = planner.validateAndGetType(parseNode);
    } catch (ValidationException e) {
      planner.close();
      throw new SqlValidationException(sql, e);
    }
    return validatedSqlNodeAndType;
  }

  public RelNode getNonOptimizedRelationalAlgebra(String sql, boolean closePlanner)
      throws SqlSyntaxException, SqlValidationException, RelConversionException {
    SqlNode validatedSqlNode = validateQuery(sql);
    RelNode result =
        planner.transform(0, planner.getEmptyTraitSet(), planner.rel(validatedSqlNode).project());
    if (closePlanner) {
      // TODO(jsternberg): Rework this logic because these are some incredibly leaky abstractions.
      // We won't be doing optimizations so transform the relational algebra to use the pandas nodes
      // now. This is a temporary thing while we transition to using the volcano planner.
      RelTraitSet requiredOutputTraits = planner.getEmptyTraitSet().replace(PandasRel.CONVENTION);
      result = planner.transform(2, requiredOutputTraits, result);
      planner.close();
    }
    // Close any open connections from catalogs
    if (catalog != null) {
      catalog.closeConnections();
    }
    return result;
  }

  public RelNode getOptimizedRelationalAlgebra(RelNode nonOptimizedPlan)
      throws RelConversionException {
    RelTraitSet requiredOutputTraits = planner.getEmptyTraitSet().replace(PandasRel.CONVENTION);
    RelNode optimizedPlan = planner.transform(1, requiredOutputTraits, nonOptimizedPlan);
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
  public RelNode getRelationalAlgebra(String sql, boolean performOptimizations)
      throws SqlSyntaxException, SqlValidationException, RelConversionException {
    RelNode nonOptimizedPlan = getNonOptimizedRelationalAlgebra(sql, !performOptimizations);
    LOGGER.debug("non optimized\n" + RelOptUtil.toString(nonOptimizedPlan));

    if (!performOptimizations) {
      return nonOptimizedPlan;
    }
    return getOptimizedRelationalAlgebra(nonOptimizedPlan);
  }

  public String getRelationalAlgebraString(String sql, boolean optimizePlan)
      throws SqlSyntaxException, SqlValidationException, RelConversionException {
    String response = "";

    try {
      response = RelOptUtil.toString(getRelationalAlgebra(sql, optimizePlan));
    } catch (SqlValidationException ex) {
      // System.out.println(ex.getMessage());
      // System.out.println("Found validation err!");
      throw ex;
      // return "fail: \n " + ex.getMessage();
    } catch (SqlSyntaxException ex) {
      // System.out.println(ex.getMessage());
      // System.out.println("Found syntax err!");
      throw ex;
      // return "fail: \n " + ex.getMessage();
    } catch (RelConversionException ex) {
      throw ex;
    } catch (Exception ex) {
      // System.out.println(ex.toString());
      // System.out.println(ex.getMessage());
      ex.printStackTrace();

      LOGGER.error(ex.getMessage());
      return "fail: \n " + ex.getMessage();
    }

    return response;
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
    RelNode optimizedPlan = getRelationalAlgebra(sql, true);
    return getPandasStringFromPlan(optimizedPlan, sql, debugDeltaTable);
  }

  // Default debugDeltaTable to false
  public String getPandasString(String sql) throws Exception {
    return getPandasString(sql, false);
  }

  public String getPandasStringUnoptimized(String sql, boolean debugDeltaTable) throws Exception {
    RelNode unOptimizedPlan = getNonOptimizedRelationalAlgebra(sql, true);
    return getPandasStringFromPlan(unOptimizedPlan, sql, debugDeltaTable);
  }

  // Default debugDeltaTable to false
  public String getPandasStringUnoptimized(String sql) throws Exception {
    return getPandasStringUnoptimized(sql, false);
  }

  private String getPandasStringFromPlan(RelNode plan, String originalSQL, boolean debugDeltaTable)
      throws Exception {
    /**
     * HashMap that maps a Calcite Node using a unique identifier for different "values". To do
     * this, we use two components. First, each RelNode comes with a unique id, which This is used
     * to track exprTypes before code generation.
     */
    HashMap<String, BodoSQLExprType.ExprType> exprTypes = new HashMap<>();
    // Map from search unique id to the expanded code generated
    HashMap<String, RexNode> searchMap = new HashMap<>();
    ExprTypeVisitor.determineRelNodeExprType(plan, exprTypes, searchMap);
    this.loweredGlobalVariables = new HashMap<>();
    PandasCodeGenVisitor codegen =
        new PandasCodeGenVisitor(
            exprTypes,
            searchMap,
            this.loweredGlobalVariables,
            originalSQL,
            this.typeSystem,
            debugDeltaTable,
            this.verboseLevel,
            this.streamingBatchSize);
    codegen.go(plan);
    String pandas_code = codegen.getGeneratedCode();
    return pandas_code;
  }

  public String getOptimizedPlanString(String sql) throws Exception {
    return RelOptUtil.toString(getRelationalAlgebra(sql, true));
  }

  public String getUnoptimizedPlanString(String sql) throws Exception {
    return RelOptUtil.toString(getRelationalAlgebra(sql, false));
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
    case HEURISTIC_PLANNER:
    default:
      return PlannerType.HEURISTIC;
    }
  }
}
