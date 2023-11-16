package com.bodosql.calcite.table;

import static java.lang.Double.min;

import com.bodosql.calcite.adapter.pandas.StreamingOptions;
import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan;
import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.application.utils.Memoizer;
import com.bodosql.calcite.catalog.BodoSQLCatalog;
import com.bodosql.calcite.catalog.SnowflakeCatalogImpl;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.rel.metadata.BodoMetadataRestrictionScan;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.CatalogSchemaImpl;
import com.bodosql.calcite.schema.InlineViewMetadata;
import com.google.common.base.Suppliers;
import java.util.*;
import java.util.function.Function;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Table;
import org.apache.calcite.schema.TranslatableTable;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.sql.type.TZAwareSqlType;
import org.apache.calcite.sql.util.SqlString;
import org.jetbrains.annotations.NotNull;

/**
 *
 *
 * <h1>Stores a table with its corresponding columns</h1>
 *
 * @author bodo
 */
public class CatalogTableImpl extends BodoSqlTable implements TranslatableTable {
  // Hold the statistics for this table.
  private final Statistic statistic = new StatisticImpl();

  /**
   * See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Table
   */

  /**
   * This constructor is used to fill in all the values for the table.
   *
   * @param name the name of the table that is being created
   * @param schema the BodoSQL schema to which this table belongs
   * @param columns list of columns to be added to the table.
   */
  public CatalogTableImpl(String name, BodoSqlSchema schema, List<BodoSQLColumn> columns) {
    super(name, schema, columns);
    if (!(schema instanceof CatalogSchemaImpl)) {
      throw new RuntimeException("Catalog table must be initialized with a Catalog Schema.");
    }
  }

  /**
   * Returns the schema for the table but cast to the required CatalogSchemaImpl. This is done when
   * actions are catalog specific, such as the read/write code.
   *
   * @return The stored schema cast to a CatalogSchemaImpl.
   */
  private CatalogSchemaImpl getCatalogSchema() {
    // This cast is safe because it's enforced on the constructor.
    return (CatalogSchemaImpl) this.getSchema();
  }

  /** Returns the catalog for the table. */
  public BodoSQLCatalog getCatalog() {
    return getCatalogSchema().getCatalog();
  }

  /**
   * Can BodoSQL write to this table. By default this is true but in the future this may be extended
   * to look at the permissions given in the catalog.
   *
   * @return Can BodoSQL write to this table.
   */
  @Override
  public boolean isWriteable() {
    // TODO: Update with the ability to check permissions from the schema/catalog
    return true;
  }

  /**
   * This is used to facilitate the indirection required for getting the correct casing.
   *
   * <p>Calcite needs to pretend that the case is lowercase for the purposes of expanding the star
   * for selects and also to fit in with the pandas convention.
   *
   * <p>At the same time, Calcite needs to know the original name of the columns for SQL generation.
   *
   * <p>Until we have conventions in place and have overridden the default behavior of star (which
   * uses the real names instead of normalized lowercase names), we need to have this little hack.
   *
   * @param name column index.
   * @return the column name.
   */
  public String getPreservedColumnName(String name) {
    for (BodoSQLColumn column : columns) {
      if (column.getColumnName().equals(name)) {
        // We found the original column so return
        // the write name as that's the original.
        return column.getWriteColumnName();
      }
    }
    // Just return the original name.
    return name;
  }

  /**
   * Generate the code needed to write the given variable to storage. This table type generates code
   * common to all tables in the catalog.
   *
   * @param varName Name of the variable to write.
   * @return The generated code to write the table.
   */
  @Override
  public Expr generateWriteCode(Variable varName) {
    return this.getCatalogSchema()
        .generateWriteCode(varName, this.getName(), BodoSQLCatalog.ifExistsBehavior.APPEND);
  }

  /**
   * Generate the code needed to write the given variable to storage.
   *
   * @param varName Name of the variable to write.
   * @param extraArgs Extra arguments to pass to the Python API. They are assume to be escaped by
   *     the calling function and are of the form "key1=value1, ..., keyN=valueN".
   * @return The generated code to write the table.
   */
  public Variable generateWriteCode(Variable varName, String extraArgs) {
    throw new UnsupportedOperationException("Catalog APIs do not support additional arguments");
  }

  /**
   * Generate the streaming code needed to initialize a writer for the given variable.
   *
   * @return The generated streaming code to write the table.
   */
  public Expr generateStreamingWriteInitCode(Expr.IntegerLiteral operatorID) {
    return this.getCatalogSchema()
        .generateStreamingWriteInitCode(
            operatorID, this.getName(), BodoSQLCatalog.ifExistsBehavior.APPEND);
  }

  public Expr generateStreamingWriteAppendCode(
      Variable stateVarName,
      Variable dfVarName,
      Variable colNamesGlobal,
      Variable isLastVarName,
      Variable iterVarName,
      Expr columnPrecisions) {
    return this.getCatalogSchema()
        .generateStreamingWriteAppendCode(
            stateVarName, dfVarName, colNamesGlobal, isLastVarName, iterVarName, columnPrecisions);
  }

  /**
   * Return the location from which the table is generated. The return value is always entirely
   * capitalized.
   *
   * @return The source DB location.
   */
  @Override
  public String getDBType() {
    return this.getCatalogSchema().getDBType().toUpperCase();
  }

  /**
   * Generate the code needed to read the table. This table type generates code common to all tables
   * in the catalog.
   *
   * @param useStreaming Should we generate code to read the table as streaming (currently only
   *     supported for snowflake tables)
   * @param streamingOptions Streaming-related options including batch size
   * @return The generated code to read the table.
   */
  @Override
  public Expr generateReadCode(boolean useStreaming, StreamingOptions streamingOptions) {
    return this.getCatalogSchema().generateReadCode(this.getName(), useStreaming, streamingOptions);
  }

  /**
   * Generate the code needed to read the table. This function is called by specialized IO
   * implementations that require passing 1 or more additional arguments.
   *
   * @param extraArgs: Extra arguments to pass to the Python API. They are assume to be escaped by
   *     the calling function and are of the form "key1=value1, ..., keyN=valueN".
   * @return The generated code to read the table.
   */
  @Override
  public Expr generateReadCode(String extraArgs) {
    throw new UnsupportedOperationException("Catalog APIs do not support additional arguments");
  }

  @Override
  public Expr generateReadCastCode(Variable varName) {
    // Snowflake catalog uses _bodo_read_date_as_dt64=True to convert date columns to datetime64
    // without astype() calls in the IR which cause issues for limit pushdown.
    // see BE-4238
    return varName;
  }

  /**
   * Generates the code necessary to submit the remote query to the catalog DB. This is not
   * supported for local tables.
   *
   * @param query Query to submit.
   * @return The generated code.
   */
  @Override
  public Expr generateRemoteQuery(String query) {
    return this.getCatalogSchema().generateRemoteQuery(query);
  }

  @Override
  public Table extend(List<RelDataTypeField> extensionFields) {
    String name = this.getName();
    BodoSqlSchema schema = this.getSchema();
    List<BodoSQLColumn> extendedColumns = new ArrayList<>();
    extendedColumns.addAll(this.columns);
    for (int i = 0; i < extensionFields.size(); i++) {
      RelDataTypeField curField = extensionFields.get(0);
      String fieldName = curField.getName();
      RelDataType colType = curField.getType();
      BodoSQLColumn.BodoSQLColumnDataType newColType =
          BodoSQLColumn.BodoSQLColumnDataType.fromSqlType(colType);
      // getTZInfo() returns null if the type is not TZAware Timestamp
      BodoTZInfo tzInfo = TZAwareSqlType.getTZInfo(colType);
      BodoSQLColumn newCol = new BodoSQLColumnImpl(fieldName, newColType, false, tzInfo);
      extendedColumns.add(newCol);
    }
    return new CatalogTableImpl(name, schema, extendedColumns);
  }

  /**
   * Returns if calling `generateReadCode()` for a table will result in an IO operation in the Bodo
   * generated code.
   *
   * @return Does the table require IO?
   */
  @Override
  public boolean readRequiresIO() {
    return true;
  }

  @Override
  public Statistic getStatistic() {
    return statistic;
  }

  /**
   * Try to inline a view. If the view cannot be inlined then return the baseRelNode instead.
   *
   * @param toRelContext The context used to expand a view.
   * @param viewDefinition The view definition.
   * @param baseRelNode The RelNode generated if inlining this view fails.
   * @return Either the new tree generated from inlining a view or the baseRelNode.
   */
  private RelNode tryInlineView(
      RelOptTable.ToRelContext toRelContext,
      @NotNull String viewDefinition,
      @NotNull RelNode baseRelNode) {
    try {
      RelRoot root =
          toRelContext.expandView(
              baseRelNode.getRowType(),
              viewDefinition,
              List.of(getCatalog().getCatalogName(), getSchema().getName()),
              List.of(getCatalog().getCatalogName(), getSchema().getName(), getName()));
      return root.project();
    } catch (Exception e) {
      return baseRelNode;
    }
  }

  @Override
  public RelNode toRel(RelOptTable.ToRelContext toRelContext, RelOptTable relOptTable) {
    // TODO(jsternberg): We should refactor the catalog table types to specific adapters (see also,
    // trySubmitIntegerMetadataQuerySnowflake).
    // This catalog is only used for snowflake though so we're going to cheat a little
    // bit before the refactor and directly create it here rather than refactor the entire
    // chain. That should reduce the scope of the code change to make it more easily reviewed
    // and separate the new feature from the refactor.
    RelNode baseRelNode = SnowflakeTableScan.create(toRelContext.getCluster(), relOptTable, this);
    // Check if this table is a view and if so attempt to inline it.
    if (RelationalAlgebraGenerator.tryInlineViews && canSafelyInlineView()) {
      String viewDefinition = getViewDefinitionString();
      if (viewDefinition != null) {
        return tryInlineView(toRelContext, viewDefinition, baseRelNode);
      }
    }
    return baseRelNode;
  }

  private final Function<Integer, Double> columnDistinctCount =
      Memoizer.memoize(this::estimateColumnDistinctCount);

  /**
   * Determine the estimated approximate number of distinct values for the column. This value is
   * memoized.
   *
   * @return Estimated distinct count for this table.
   */
  public @Nullable Double getColumnDistinctCount(int column) {
    return columnDistinctCount.apply(column);
  }

  /**
   * Estimate the distinct count for a column by submitting an APPROX_COUNT_DISTINCT call to
   * Snowflake. This value is capped to be no greater than the row count, which Snowflake can
   * violate.
   *
   * @param column The column to check.
   * @return The approximate distinct count. Returns null if there is a timeout.
   */
  private @Nullable Double estimateColumnDistinctCount(int column) {
    List<String> qualifiedName = List.of(getSchema().getName(), getName());
    String columnName = getColumnNames().get(column).toUpperCase(Locale.ROOT);
    SnowflakeCatalogImpl catalog = (SnowflakeCatalogImpl) getCatalog();
    // Do not allow the metadata to be requested if this column of this table was
    // not pre-cleared by the metadata scanning pass.
    if (!BodoMetadataRestrictionScan.Companion.canRequestColumnDistinctiveness(
        qualifiedName, columnName)) {
      return null;
    }
    // Avoid ever returning more than the row count. This can happen because
    // Snowflake returns an estimate.
    Double distinctCount = catalog.estimateColumnDistinctCount(qualifiedName, columnName);

    // If the original query failed, try again with sampling
    if (distinctCount == null) {
      distinctCount =
          catalog.estimateColumnDistinctCountWithSampling(
              qualifiedName, columnName, getStatistic().getRowCount());
    }

    // Important: We must use getStatistic() here to allow subclassing, which we use for
    // our mocking infrastructure.
    Double maxCount = getStatistic().getRowCount();
    if (distinctCount == null || maxCount == null) {
      return null;
    }
    return min(distinctCount, maxCount);
  }

  /**
   * Wrappers around submitRowCountQueryEstimateInternal that handles memoization. See
   * submitRowCountQueryEstimateInternal for documentation.
   */
  public @Nullable Long trySubmitIntegerMetadataQuerySnowflake(SqlString sql) {
    return trySubmitLongMetadataQuerySnowflakeMemoizedFn.apply(sql);
  }

  private final Function<SqlString, Long> trySubmitLongMetadataQuerySnowflakeMemoizedFn =
      Memoizer.memoize(this::trySubmitLongMetadataQuerySnowflakeInternal);

  /**
   * Submits a Submits the specified query to Snowflake for evaluation, with a timeout. See
   * SnowflakeCatalogImpl#trySubmitIntegerMetadataQuery for the full documentation.
   *
   * <p>This should only ever be used for snowflake catalog tables, it will throw an error
   * otherwise.
   *
   * <p>TODO(Keaton): refactor the catalog table types to specific adapters (see also, toRel)
   *
   * @return
   */
  public @Nullable Long trySubmitLongMetadataQuerySnowflakeInternal(
      SqlString metadataSelectQueryString) {
    SnowflakeCatalogImpl catalog = (SnowflakeCatalogImpl) getCatalog();
    return catalog.trySubmitLongMetadataQuery(metadataSelectQueryString);
  }

  /**
   * Load the view metadata information from the catalog. If the table is not a view or no
   * information can be found this should return NULL. This should be used to implement
   * isAccessibleView(), canSafelyInlineView(), and getViewDefinitionString().
   *
   * <p>This is currently only support for Snowflake catalogs.
   *
   * @return The InlineViewMetadata loaded from the catalog or null if no information is available.
   */
  private @Nullable InlineViewMetadata tryGetViewMetadata() {
    String tableName = getName();
    String schemaName = getSchema().getName();
    SnowflakeCatalogImpl catalog = (SnowflakeCatalogImpl) getCatalog();
    return catalog.tryGetViewMetadata(List.of(schemaName, tableName));
  }

  /**
   * Is this table definitely a view (meaning we can access its definition). If this returns False
   * we may have a view if we lack the permissions necessary to know it is a view.
   *
   * @return True if this is a view for which we can load metadata information.
   */
  public boolean isAccessibleView() {
    return tryGetViewMetadata() != null;
  }

  /**
   * Is this a view that can be safely inlined.
   *
   * @return Returns true if table is a view and the metadata indicates inlining is legal. If this
   *     table is not a view this return false.
   */
  private boolean canSafelyInlineView() {
    if (isAccessibleView()) {
      InlineViewMetadata metadata = tryGetViewMetadata();
      return !(metadata.getUnsafeToInline() || metadata.isMaterialized());
    }
    return false;
  }

  /**
   * Get the SQL query definition used to define this table if it is a view.
   *
   * @return The string definition that was used to create the view. Returns null if the table is
   *     not a view.
   */
  private @Nullable String getViewDefinitionString() {
    if (isAccessibleView()) {
      return tryGetViewMetadata().getViewDefinition();
    }
    return null;
  }

  private class StatisticImpl implements Statistic {
    private final Supplier<Double> rowCount = Suppliers.memoize(this::estimateRowCount);

    /**
     * Retrieves the estimated row count for this table. This value is memoized.
     *
     * @return estimated row count for this table.
     */
    @Override
    public @Nullable Double getRowCount() {
      return rowCount.get();
    }

    /**
     * Retrieves the estimated row count for this table. It performs a query every time this is
     * invoked.
     *
     * @return estimated row count for this table.
     */
    private @Nullable Double estimateRowCount() {
      List<String> qualifiedName = List.of(getSchema().getName(), getName());
      SnowflakeCatalogImpl catalog = (SnowflakeCatalogImpl) getCatalog();
      return catalog.estimateRowCount(qualifiedName);
    }
  }
}
