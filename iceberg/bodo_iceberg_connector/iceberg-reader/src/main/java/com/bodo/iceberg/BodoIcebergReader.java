package com.bodo.iceberg;

import java.util.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.DeleteFile;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.TableScan;
import org.apache.iceberg.arrow.ArrowSchemaUtil;
import org.apache.iceberg.catalog.Namespace;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.expressions.Expressions;
import org.apache.iceberg.expressions.Literal;
import org.apache.iceberg.hadoop.HadoopTables;
import org.apache.iceberg.hive.HiveCatalog;

public class BodoIcebergReader {
  /**
   * Java Class used to map Bodo's required read operations to a corresponding Iceberg table. This
   * is meant to provide 1 instance per Table and Bodo is responsible for closing it.
   */

  // Table instance for the underlying Iceberg table
  private Table table;

  public BodoIcebergReader(String warehouse_loc, String db_name, String tableName) {
    // Hive metastore case
    if (warehouse_loc.startsWith("thrift:")) {
      Map<String, String> properties = new HashMap<>();
      properties.put(CatalogProperties.URI, warehouse_loc);
      HiveCatalog catalog = new HiveCatalog();
      Configuration conf = new Configuration();
      catalog.setConf(conf);
      // TODO[BE-2833]: explore using CachingCatalog
      catalog.initialize("hive_catalog", properties);
      Namespace db_namespace = Namespace.of(db_name);
      TableIdentifier name = TableIdentifier.of(db_namespace, tableName);
      this.table = catalog.loadTable(name);
      return;
    }

    // local file system case
    HadoopTables hadoopTables = new HadoopTables();
    // Set CWD for opening the metadata files later.
    System.setProperty("user.dir", warehouse_loc);
    this.table = hadoopTables.load(warehouse_loc + "/" + db_name + "/" + tableName);
  }

  public Schema getIcebergSchema() {
    return this.table.schema();
  }

  public org.apache.arrow.vector.types.pojo.Schema getArrowSchema() {
    return ArrowSchemaUtil.convert(this.table.schema());
  }

  /** Returns a list of parquet files that construct the given Iceberg Table. */
  public List<BodoParquetInfo> getParquetInfo(LinkedList<Object> filters) {
    Expression filter = filtersToExpr(filters);
    TableScan scan = table.newScan().filter(filter);
    List<BodoParquetInfo> parquetPaths = new ArrayList<>();
    Iterable<FileScanTask> files = scan.planFiles();
    for (FileScanTask file : files) {
      // Set to null by default to save memory while we don't support deletes.
      List<String> deletes = null;
      // Check for any delete files.
      List<DeleteFile> deleteFiles = file.deletes();
      if (!deleteFiles.isEmpty()) {
        deletes = new LinkedList<>();
        for (DeleteFile deleteFile : deleteFiles) {
          deletes.add(String.valueOf(deleteFile.path()));
        }
      }
      parquetPaths.add(
          new BodoParquetInfo(file.file().path().toString(), file.start(), file.length(), deletes));
    }
    return parquetPaths;
  }

  /**
   * Parses the filters passed to java, which is a list of operators and filter components and
   * converts them to proper iceberg scalars. Each individual filter consists of column name,
   * OpEnum, IcebergLiteral.
   *
   * <p>In addition, filters are joined by other OpEnum values AND/OR. We don't have to worry about
   * operator precedence because the form is always ORing AND expressions.
   */
  public Expression filtersToExpr(LinkedList<Object> filters) {
    if (filters == null || filters.isEmpty()) {
      // If there are no predicates pass the true predicate
      return Expressions.alwaysTrue();
    }
    // We now process expressions in ANDs.
    LinkedList<Expression> expressions = new LinkedList<>();
    while (!filters.isEmpty()) {
      expressions.push(andFiltersToExpr(filters));
    }
    if (expressions.size() == 1) {
      return expressions.pop();
    }
    Expression currentExpr = expressions.removeFirst();
    while (!expressions.isEmpty()) {
      currentExpr = Expressions.or(currentExpr, expressions.removeFirst());
    }
    return currentExpr;
  }

  public Expression andFiltersToExpr(LinkedList<Object> filters) {
    Object andStart = filters.removeFirst();
    assert andStart.equals(OpEnum.AND_START);
    LinkedList<Expression> expressions = new LinkedList<>();
    while (!filters.getFirst().equals(OpEnum.AND_END)) {
      expressions.push(singleFilterToExpr(filters));
    }
    // Remove the ANDEND
    filters.removeFirst();
    if (expressions.size() == 0) {
      // If this is just start-end return TRUE.
      return Expressions.alwaysTrue();
    } else if (expressions.size() == 1) {
      return expressions.pop();
    }
    Expression currentExpr = expressions.removeFirst();
    while (!expressions.isEmpty()) {
      currentExpr = Expressions.and(currentExpr, expressions.removeFirst());
    }
    return currentExpr;
  }

  public Expression singleFilterToExpr(LinkedList<Object> filters) {
    // First value is always a field ID.
    String name = (String) filters.removeFirst();
    // Get the op.
    OpEnum op = (OpEnum) filters.removeFirst();

    Expression.Operation icebergOp = Expression.Operation.TRUE;
    // Only used by in/not in
    ArrayList<Object> lit_list = null;
    switch (op) {
      case EQ:
        icebergOp = Expression.Operation.EQ;
        break;
      case NE:
        icebergOp = Expression.Operation.NOT_EQ;
        break;
      case LT:
        icebergOp = Expression.Operation.LT;
        break;
      case GT:
        icebergOp = Expression.Operation.GT;
        break;
      case GE:
        icebergOp = Expression.Operation.GT_EQ;
        break;
      case LE:
        icebergOp = Expression.Operation.LT_EQ;
        break;
      case STARTS_WITH:
        icebergOp = Expression.Operation.STARTS_WITH;
        break;
      case NOT_STARTS_WITH:
        icebergOp = Expression.Operation.NOT_STARTS_WITH;
        break;
      case IN:
        icebergOp = Expression.Operation.IN;
        // NOTE: Iceberg takes regular Java lists in this case, not Literal lists.
        // see predicate(Expression.Operation op, java.lang.String name,
        //               java.lang.Iterable<T> values)
        // https://iceberg.apache.org/javadoc/0.13.1/index.html?org/apache/iceberg/types/package-summary.html
        lit_list = (ArrayList<Object>) filters.removeFirst();
        return Expressions.predicate(icebergOp, name, lit_list);
      case NOT_IN:
        icebergOp = Expression.Operation.NOT_IN;
        // NOTE: Iceberg takes regular Java lists in this case, not Literal lists.
        // see predicate(Expression.Operation op, java.lang.String name,
        //               java.lang.Iterable<T> values)
        // https://iceberg.apache.org/javadoc/0.13.1/index.html?org/apache/iceberg/types/package-summary.html
        lit_list = (ArrayList<Object>) filters.removeFirst();
        return Expressions.predicate(icebergOp, name, lit_list);
      case IS_NULL:
        icebergOp = Expression.Operation.IS_NULL;
        // remove "NULL" from list
        filters.removeFirst();
        return Expressions.predicate(icebergOp, name);
      case NOT_NULL:
        icebergOp = Expression.Operation.NOT_NULL;
        // remove "NULL" from list
        filters.removeFirst();
        return Expressions.predicate(icebergOp, name);
      default:
        // We should never reach this case.
        assert false;
    }
    // Get the literal
    Literal<Object> lit = (Literal<Object>) filters.removeFirst();
    return Expressions.predicate(icebergOp, name, lit);
  }
}
