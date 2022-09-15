package com.bodosql.calcite.application;

import com.bodosql.calcite.application.BodoSQLOperatorTables.*;
import com.bodosql.calcite.application.BodoSQLRules.*;
import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.CatalogSchemaImpl;
import java.sql.Connection;
import java.sql.DriverManager;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import javax.annotation.Nullable;
import org.apache.calcite.avatica.util.Casing;
import org.apache.calcite.avatica.util.Quoting;
import org.apache.calcite.config.CalciteConnectionConfigImpl;
import org.apache.calcite.config.NullCollation;
import org.apache.calcite.jdbc.CalciteConnection;
import org.apache.calcite.jdbc.CalciteSchema;
import org.apache.calcite.jdbc.JavaTypeFactoryImpl;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgram;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.prepare.CalciteCatalogReader;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rex.RexExecutorImpl;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.util.ChainedSqlOperatorTable;
import org.apache.calcite.sql.validate.SqlConformanceEnum;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.sql2rel.StandardConvertletTable;
import org.apache.calcite.sql2rel.StandardConvertletTableConfig;
import org.apache.calcite.tools.*;
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
  /** Program which takes relational algebra and optimizes it */
  private HepProgram program;
  /** Stores the context for the program hep planner. E.G. it stores the schema. */
  private FrameworkConfig config;

  private List<RelOptRule> rules;

  /*
  Hashmap containing globals that need to be lowered into the resulting func_text. Used for lowering
  metadata types to improve compilation speed. Populated by the pandas visitor class.
  */
  private HashMap<String, String> loweredGlobalVariables;

  /** Store the catalog being used to close any connections after processing a query. */
  private BodoSQLCatalog catalog;

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
  public SchemaPlus setupSchema(
      CalciteConnection calciteConnection, List<BodoSqlSchema> newSchemas) {
    SchemaPlus schema = null;
    try {
      schema = calciteConnection.getRootSchema();
      for (BodoSqlSchema newSchema : newSchemas) {
        schema.add(newSchema.getName(), newSchema);
      }
    } catch (Exception e) {
      throw new RuntimeException(
          String.format("Internal Error: Unable to add schemas to database. Error message: %s", e));
    }
    return schema;
  }

  /**
   * Helper method for both RelationalAlgebraGenerator constructors to build and set the Config and
   * the Planner member variables.
   */
  public void setupPlanner(
      List<SchemaPlus> defaultSchemas, SchemaPlus schema, String namedParamTableName) {
    try {
      // Generate the schema paths for the operator table.
      List<List<String>> defaultSchemaList = new ArrayList<>();
      for (SchemaPlus defaultSchema : defaultSchemas) {
        defaultSchemaList.add(CalciteSchema.from(defaultSchema).path(null));
      }
      RelDataTypeSystem typeSystem = new BodoSQLRelDataTypeSystem();
      Properties props = new Properties();
      List<SqlOperatorTable> sqlOperatorTables = new ArrayList<>();
      // TODO: Replace this code. Deprecated?
      sqlOperatorTables.add(SqlStdOperatorTable.instance());
      sqlOperatorTables.add(
          new CalciteCatalogReader(
              CalciteSchema.from(schema),
              defaultSchemaList,
              defaultSchemaList.size(),
              new JavaTypeFactoryImpl(typeSystem),
              new CalciteConnectionConfigImpl(props)));
      sqlOperatorTables.add(DatetimeOperatorTable.instance());
      sqlOperatorTables.add(NumericOperatorTable.instance());
      sqlOperatorTables.add(StringOperatorTable.instance());
      sqlOperatorTables.add(CondOperatorTable.instance());
      sqlOperatorTables.add(SinceEpochFnTable.instance());
      sqlOperatorTables.add(ThreeOperatorStringTable.instance());
      config =
          Frameworks.newConfigBuilder()
              .defaultSchemas(defaultSchemas)
              .operatorTable(new ChainedSqlOperatorTable(sqlOperatorTables))
              .typeSystem(typeSystem)
              // Currently, Calcite only supports SOME/ANY/ALL if isExpand = false.
              // The threshold value is used to determine, when handling a SOME/ANY/ALL clause,
              // at what point the converter switches from simply creating a sequence of and's and
              // or's,
              // and instead do an inner join/filter on the values. See BS-553.
              .sqlToRelConverterConfig(
                  SqlToRelConverter.config().withInSubQueryThreshold(Integer.MAX_VALUE))
              .parserConfig(
                  SqlParser.Config.DEFAULT
                      .withCaseSensitive(false)
                      .withQuoting(Quoting.BACK_TICK)
                      .withQuotedCasing(Casing.UNCHANGED)
                      .withUnquotedCasing(Casing.UNCHANGED)
                      .withConformance(SqlConformanceEnum.LENIENT))
              .convertletTable(
                  new StandardConvertletTable(new StandardConvertletTableConfig(false, false)))
              .sqlValidatorConfig(
                  SqlValidator.Config.DEFAULT
                      .withNamedParamTableName(namedParamTableName)
                      .withDefaultNullCollation(NullCollation.LOW)
                      .withCallRewrite(
                          false)) /* setting with withCallRewrite to false disables the rewriting of
                                  "macro-like" functions. Namely:
                                  COALESCE
                                  DAYOFMONTH
                                  DAYOFWEEK
                                  DAYOFYEAR
                                  HOUR
                                  MICROSECOND
                                  MINUTE
                                  MONTH
                                  NULLIF
                                  QUARTER
                                  SECOND
                                  WEEK
                                  WEEKDAY
                                  WEEKISO
                                  WEEKOFYEAR
                                  YEAR
                                  There are likely others, but there are the ones we have encountered so far.
                                   */
              .build();
      planner = Frameworks.getPlanner(config);
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
   * {@link #config} and then set up the {@link #program} for optimizing and the {@link #planner}
   * for parsing.
   *
   * @param newSchema This is the schema which we will be using to validate our query against. This
   *     gets stored in the {@link #config}
   */
  public RelationalAlgebraGenerator(BodoSqlSchema newSchema, String namedParamTableName) {
    this.catalog = null;
    System.setProperty("calcite.default.charset", "UTF-8");
    try {
      CalciteConnection calciteConnection = setupCalciteConnection();
      List<BodoSqlSchema> newSchemas = new ArrayList<BodoSqlSchema>();
      newSchemas.add(newSchema);
      SchemaPlus schema = setupSchema(calciteConnection, newSchemas);

      List<SchemaPlus> defaultSchemas = new ArrayList();
      defaultSchemas.add(schema.getSubSchema(newSchema.getName()));
      setupPlanner(defaultSchemas, schema, namedParamTableName);
    } catch (Exception e) {
      throw new RuntimeException(
          String.format(
              "Internal Error: Unable to create Relational Algebra Generator. Error message: %s",
              e));
    }
  }

  /**
   * Constructor for the relational algebra generator class that accepts a Catalog and Schema
   * objects. It will take the schema objects in the Catalog as well as the Schema object store it
   * in the {@link #config} and then set up the {@link #program} for optimizing and the {@link
   * #planner} for parsing.
   */
  public RelationalAlgebraGenerator(
      BodoSQLCatalog catalog, BodoSqlSchema newSchema, String namedParamTableName) {
    this.catalog = catalog;
    System.setProperty("calcite.default.charset", "UTF-8");
    try {
      CalciteConnection calciteConnection = setupCalciteConnection();
      Set<String> schemaNames = catalog.getSchemaNames();
      List<BodoSqlSchema> newSchemas = new ArrayList();

      newSchemas.add(newSchema);
      for (String schemaName : schemaNames) {
        newSchemas.add(new CatalogSchemaImpl(schemaName, catalog));
      }
      SchemaPlus schema = setupSchema(calciteConnection, newSchemas);

      List<SchemaPlus> defaultSchemas = new ArrayList();
      List<BodoSqlSchema> defaultCatalogSchema = catalog.getDefaultSchema();
      for (BodoSqlSchema catalogDefaultSchema : defaultCatalogSchema) {
        // Fetch the path from the root schema.
        defaultSchemas.add(schema.getSubSchema(catalogDefaultSchema.getName()));
      }
      defaultSchemas.add(schema.getSubSchema(newSchema.getName()));
      setupPlanner(defaultSchemas, schema, namedParamTableName);
    } catch (Exception e) {
      throw new RuntimeException(
          String.format(
              "Internal Error: Unable to create Relational Algebra Generator. Error message: %s",
              e));
    }
  }

  public void setRules(List<RelOptRule> rules) {
    this.rules = rules;
  }

  public SqlNode validateQuery(String sql) throws SqlSyntaxException, SqlValidationException {
    SqlNode tempNode;
    try {
      tempNode = planner.parse(sql);
    } catch (SqlParseException e) {
      planner.close();
      throw new SqlSyntaxException(sql, e);
    }

    SqlNode validatedSqlNode;
    try {
      validatedSqlNode = planner.validate(tempNode);
    } catch (ValidationException e) {
      planner.close();
      throw new SqlValidationException(sql, e);
    }
    return validatedSqlNode;
  }

  public RelNode getNonOptimizedRelationalAlgebra(String sql, boolean closePlanner)
      throws SqlSyntaxException, SqlValidationException, RelConversionException {
    SqlNode validatedSqlNode = validateQuery(sql);
    RelNode result = planner.rel(validatedSqlNode).project();
    if (closePlanner) {
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
    if (rules == null) {
      program =
          new HepProgramBuilder()
              /*
              Planner rule that, given a Project node that merely returns its input, converts the node into its child.
              */
              .addRuleInstance(ProjectUnaliasedRemoveRule.Config.DEFAULT.toRule())
              /*
              Planner rule that combines two LogicalFilters.
              */
              .addRuleInstance(FilterMergeRule.Config.DEFAULT.toRule())
              /*
                 Planner rule that merges a Project into another Project,
                 provided the projects aren't projecting identical sets of input references.
              */
              .addRuleInstance(ProjectMergeRule.Config.DEFAULT.toRule())
              /*
              Planner rule that pushes a Filter past a Aggregate.
                 */
              .addRuleInstance(FilterAggregateTransposeRule.Config.DEFAULT.toRule())
              /*
               * Planner rule that matches an {@link org.apache.calcite.rel.core.Aggregate}
               * on a {@link org.apache.calcite.rel.core.Join} and removes the join
               * provided that the join is a left join or right join and it computes no
               * aggregate functions or all the aggregate calls have distinct.
               *
               * <p>For instance,</p>
               *
               * <blockquote>
               * <pre>select distinct s.product_id from
               * sales as s
               * left join product as p
               * on s.product_id = p.product_id</pre></blockquote>
               *
               * <p>becomes
               *
               * <blockquote>
               * <pre>select distinct s.product_id from sales as s</pre></blockquote>
               */
              .addRuleInstance(AggregateJoinRemoveRule.Config.DEFAULT.toRule())
              /*
                 Planner rule that pushes an Aggregate past a join
              */
              .addRuleInstance(AggregateJoinTransposeRule.Config.EXTENDED.toRule())
              /*
              Rule that tries to push filter expressions into a join condition and into the inputs of the join.
              */
              .addRuleInstance(
                  FilterJoinRule.FilterIntoJoinRule.FilterIntoJoinRuleConfig.DEFAULT.toRule())
              /*
              Rule that applies moves any filters that depend on a single table before the join in
              which they occur.
               */
              .addRuleInstance(
                  FilterJoinRule.JoinConditionPushRule.JoinConditionPushRuleConfig.DEFAULT.toRule())
              /*
              Filters tables for unused columns before join.
              */
              .addRuleInstance(AliasPreservingProjectJoinTransposeRule.Config.DEFAULT.toRule())
              /*
              This reduces expressions inside of the conditions of filter statements.
              Ex condition($0 = 1 and $0 = 2) ==> condition(FALSE)
              TODO(Ritwika: figure out SEARCH handling later. SARG attributes do not have public access methods.
              */
              .addRuleInstance(
                  ReduceExpressionsRule.FilterReduceExpressionsRule
                      .FilterReduceExpressionsRuleConfig.DEFAULT
                      .toRule())
              /*
              Pushes predicates that are used on one side of equality in a join to
              the other side of the join as well, enabling further filter pushdown
              and reduce the amount of data joined.

              For example, consider the query:

              select t1.a, t2.b from table1 t1, table2 t2 where t1.a = 1 AND t1.a = t2.b

              This produces a plan like

              LogicalProject(a=[$0], b=[$1])
                LogicalJoin(condition=[=($0, $1)], joinType=[inner])
                  LogicalProject(A=[$0])
                    LogicalFilter(condition=[=($0, 1)])
                      LogicalTableScan(table=[[main, table1]])
                  LogicalProject(B=[$1])
                      LogicalFilter(condition=[=($1, 1)])
                        LogicalTableScan(table=[[main, table2]])

               So both table1 and table2 filter on col = 1.
               */
              .addRuleInstance(JoinPushTransitivePredicatesRule.Config.DEFAULT.toRule())
              /*
               * Planner rule that removes
               * a {@link org.apache.calcite.rel.core.Aggregate}
               * if it computes no aggregate functions
               * (that is, it is implementing {@code SELECT DISTINCT}),
               * or all the aggregate functions are splittable,
               * and the underlying relational expression is already distinct.
               *
               */
              .addRuleInstance(AggregateRemoveRule.Config.DEFAULT.toRule())
              /*
               * Planner rule that matches an {@link org.apache.calcite.rel.core.Aggregate}
               * on a {@link org.apache.calcite.rel.core.Join} and removes the left input
               * of the join provided that the left input is also a left join if possible.
               *
               * <p>For instance,
               *
               * <blockquote>
               * <pre>select distinct s.product_id, pc.product_id
               * from sales as s
               * left join product as p
               *   on s.product_id = p.product_id
               * left join product_class pc
               *   on s.product_id = pc.product_id</pre></blockquote>
               *
               * <p>becomes
               *
               * <blockquote>
               * <pre>select distinct s.product_id, pc.product_id
               * from sales as s
               * left join product_class pc
               *   on s.product_id = pc.product_id</pre></blockquote>
               *
               * @see CoreRules#AGGREGATE_JOIN_JOIN_REMOVE
               */
              .addRuleInstance(AggregateJoinJoinRemoveRule.Config.DEFAULT.toRule()) /*
              /*
               * Planner rule that merges an Aggregate into a projection when possible,
               * maintaining any aliases.
               */
              .addRuleInstance(AliasPreservingAggregateProjectMergeRule.Config.DEFAULT.toRule())
              /*
               * Planner rule that merges a Projection into an Aggregate when possible,
               * maintaining any aliases.
               */
              .addRuleInstance(ProjectAggregateMergeRule.Config.DEFAULT.toRule())
              /*
               * Planner rule that ensures filter is always pushed into join. This is needed
               * for complex queries.
               */
              .addRuleInstance(FilterProjectTransposeRule.Config.DEFAULT.toRule())
              // Prune trivial cross-joins
              .addRuleInstance(InnerJoinRemoveRule.Config.DEFAULT.toRule())
              // Rewrite filters in either Filter or Join to convert OR with shared subexpression
              // into
              // an AND and then OR. For example
              // OR(AND(A > 1, B < 10), AND(A > 1, A < 5)) -> AND(A > 1, OR(B < 10 , A < 5))
              // Another rule pushes filters into join and we do not know if the LogicalFilter
              // optimization will get to run before its pushed into the join. As a result,
              // we write a duplicate rule that operates directly on the condition of the join.
              .addRuleInstance(JoinReorderConditionRule.Config.DEFAULT.toRule())
              .addRuleInstance(LogicalFilterReorderConditionRule.Config.DEFAULT.toRule())
              // Push a limit before a project (e.g. select col as alias from table limit 10)
              .addRuleInstance(LimitProjectTransposeRule.Config.DEFAULT.toRule())
              .build();

    } else {
      HepProgramBuilder programBuilder = new HepProgramBuilder();
      for (RelOptRule rule : rules) {
        programBuilder = programBuilder.addRuleInstance(rule);
      }
      program = programBuilder.build();
    }

    final HepPlanner hepPlanner = new HepPlanner(program, config.getContext());
    nonOptimizedPlan.getCluster().getPlanner().setExecutor(new RexExecutorImpl(null));
    hepPlanner.setRoot(nonOptimizedPlan);

    planner.close();

    return hepPlanner.findBestExp();
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

    RelNode lastOptimizedPlan = nonOptimizedPlan;
    RelNode curOptimizedPlan = getOptimizedRelationalAlgebra(nonOptimizedPlan);

    // Set an arbitrary upper bound for apply the optimization rules in case
    // for some reason a plan doesn't converge. While we should always converge,
    // its more desirable to have a suboptimal plan than an infinite loop.
    final int maxIterations = 25;

    int currIteration = 0;

    while (!curOptimizedPlan.deepEquals(lastOptimizedPlan) && (currIteration < maxIterations)) {
      lastOptimizedPlan = curOptimizedPlan;
      curOptimizedPlan = getOptimizedRelationalAlgebra(lastOptimizedPlan);
      currIteration++;
    }

    LOGGER.debug("optimized\n" + RelOptUtil.toString(curOptimizedPlan));

    return curOptimizedPlan;
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
    } catch (Exception ex) {
      // System.out.println(ex.toString());
      // System.out.println(ex.getMessage());
      ex.printStackTrace();

      LOGGER.error(ex.getMessage());
      return "fail: \n " + ex.getMessage();
    }

    return response;
  }

  public String getPandasString(String sql) throws Exception {
    RelNode optimizedPlan = getRelationalAlgebra(sql, true);
    return getPandasStringFromPlan(optimizedPlan);
  }

  public String getPandasStringUnoptimized(String sql) throws Exception {
    RelNode unOptimizedPlan = getNonOptimizedRelationalAlgebra(sql, true);
    return getPandasStringFromPlan(unOptimizedPlan);
  }

  private String getPandasStringFromPlan(RelNode plan) throws Exception {
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
        new PandasCodeGenVisitor(exprTypes, searchMap, this.loweredGlobalVariables);
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
}
