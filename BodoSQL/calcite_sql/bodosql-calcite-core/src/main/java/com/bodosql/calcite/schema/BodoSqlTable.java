/*
 * Copyright 2018 Bodo, Inc.
 */

package com.bodosql.calcite.schema;

import com.bodosql.calcite.catalog.domain.CatalogColumn;
import com.bodosql.calcite.catalog.domain.CatalogColumnDataType;
import com.bodosql.calcite.catalog.domain.CatalogTable;
import java.util.List;
import org.apache.calcite.DataContext;
import org.apache.calcite.avatica.util.TimeUnit;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.linq4j.Enumerable;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.ProjectableFilterableTable;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Statistic;
import org.apache.calcite.schema.Statistics;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlIntervalQualifier;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BodoSqlTable implements ProjectableFilterableTable {
  static final Logger LOGGER = LoggerFactory.getLogger(BodoSqlTable.class);

  private final CatalogTable catalogTable;

  public BodoSqlTable(CatalogTable catalogTable) {
    this.catalogTable = catalogTable;
  }

  @Override
  public RelDataType getRowType(RelDataTypeFactory rdtf) {
    RelDataTypeFactory.FieldInfoBuilder builder = rdtf.builder();
    for (CatalogColumn column : catalogTable.getColumns()) {
      builder.add(column.getColumnName(), convertToSqlType(column.getColumnDataType(), rdtf));
      builder.nullable(true);
    }
    return builder.build();
  }

  @Override
  public Statistic getStatistic() {
    return Statistics.UNKNOWN;
  }

  @Override
  public Schema.TableType getJdbcTableType() {
    return Schema.TableType.TABLE;
  }

  public CatalogTable getCatalogTable() {
    return this.catalogTable;
  }

  private RelDataType convertToSqlType(
      CatalogColumnDataType dataType, RelDataTypeFactory typeFactory) {
    RelDataType temp;
    switch (dataType) {
      case INT8:
      case UINT8:
        temp = typeFactory.createSqlType(SqlTypeName.TINYINT);
        break;
      case INT16:
      case UINT16:
        temp = typeFactory.createSqlType(SqlTypeName.SMALLINT);
        break;
      case INT32:
      case UINT32:
        temp = typeFactory.createSqlType(SqlTypeName.INTEGER);
        break;
      case INT64:
      case UINT64:
        temp = typeFactory.createSqlType(SqlTypeName.BIGINT);
        break;
      case FLOAT32:
        temp = typeFactory.createSqlType(SqlTypeName.FLOAT);
        break;
      case FLOAT64:
        temp = typeFactory.createSqlType(SqlTypeName.DOUBLE);
        break;
      case BOOL8:
        temp = typeFactory.createSqlType(SqlTypeName.BOOLEAN);
        break;
      case DATE:
        temp = typeFactory.createSqlType(SqlTypeName.DATE);
        break;
      case DATETIME:
        temp = typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
        break;
      case TIMEDELTA:
        // TODO: Figure out SqlParserPos. Probably not relevant
        // if we aren't using timedelta directly for literals.
        // TODO: Determine how to hanlde Timeunits properly. Default
        // seems to only allow Year-Month and Day-Second.
        temp =
            typeFactory.createSqlIntervalType(
                new SqlIntervalQualifier(TimeUnit.DAY, TimeUnit.SECOND, SqlParserPos.ZERO));
        break;
      case DATEOFFSET:
        // TODO: Figure out SqlParserPos. Probably not relevant
        // if we aren't using timedelta directly for literals.
        temp =
            typeFactory.createSqlIntervalType(
                new SqlIntervalQualifier(TimeUnit.YEAR, TimeUnit.MONTH, SqlParserPos.ZERO));
        break;
      case STRING:
        temp = typeFactory.createSqlType(SqlTypeName.VARCHAR);
        break;
      case BINARY:
        temp = typeFactory.createSqlType(SqlTypeName.VARBINARY);
        break;
      default:
        temp = null;
    }
    // TODO: Raise an exception if temp is NULL
    return temp;
  }

  @Override
  public boolean isRolledUp(String string) {
    // will set to false by default
    return false;
  }

  @Override
  public boolean rolledUpColumnValidInsideAgg(
      String string, SqlCall sc, SqlNode sn, CalciteConnectionConfig ccc) {
    throw new UnsupportedOperationException("rolledUpColumnValidInsideAgg Not supported yet.");
  }

  @Override
  public Enumerable<Object[]> scan(DataContext root, List<RexNode> filters, int[] projects) {
    // TODO Auto-generated method stub

    return null;
  }
}
