package com.bodosql.calcite.schema;

import com.bodosql.calcite.catalog.SnowflakeCatalog;
import com.bodosql.calcite.table.BodoSQLColumn;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.FunctionParameter;
import org.apache.calcite.sql.type.BodoTZInfo;

public class SnowflakeUDFFunctionParameter implements FunctionParameter {

  private final String name;
  private final int ord;
  private final SnowflakeCatalog.SnowflakeTypeInfo typeInfo;

  public SnowflakeUDFFunctionParameter(
      String name, int ord, SnowflakeCatalog.SnowflakeTypeInfo typeInfo) {
    this.name = name;
    this.ord = ord;
    this.typeInfo = typeInfo;
  }

  /**
   * Zero-based ordinal of this parameter within the member's parameter list.
   *
   * @return Parameter ordinal
   */
  @Override
  public int getOrdinal() {
    return ord;
  }

  /**
   * Name of the parameter.
   *
   * @return Parameter name
   */
  @Override
  public String getName() {
    return name;
  }

  /**
   * Returns the type of this parameter.
   *
   * @param typeFactory Type factory to be used to create the type
   * @return Parameter type.
   */
  @Override
  public RelDataType getType(RelDataTypeFactory typeFactory) {
    // Assume nullable for now.
    final boolean nullable = true;
    final BodoTZInfo tzInfo = BodoTZInfo.getDefaultTZInfo(typeFactory.getTypeSystem());
    // Make a dummy BodoSQLColumn for now.
    BodoSQLColumn.BodoSQLColumnDataType type = typeInfo.columnDataType;
    BodoSQLColumn.BodoSQLColumnDataType elemType = typeInfo.elemType;
    int precision = typeInfo.precision;
    BodoSQLColumn column =
        new BodoSQLColumnImpl("", "", type, elemType, nullable, tzInfo, precision);
    return column.convertToSqlType(typeFactory, nullable, tzInfo, precision);
  }

  /** Bodo requires every argument to be required for now. */
  @Override
  public boolean isOptional() {
    return false;
  }
}
