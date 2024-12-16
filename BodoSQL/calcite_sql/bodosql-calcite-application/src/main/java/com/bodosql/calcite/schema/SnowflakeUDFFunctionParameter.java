package com.bodosql.calcite.schema;

import com.bodosql.calcite.table.ColumnDataTypeInfo;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.FunctionParameter;

public class SnowflakeUDFFunctionParameter implements FunctionParameter {

  private final String name;
  private final int ord;

  private final boolean isOptional;
  private final ColumnDataTypeInfo typeInfo;

  public SnowflakeUDFFunctionParameter(
      String name, int ord, ColumnDataTypeInfo typeInfo, boolean isOptional) {
    this.name = name;
    this.ord = ord;
    this.typeInfo = typeInfo;
    this.isOptional = isOptional;
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
    return typeInfo.convertToSqlType(typeFactory);
  }

  /** Bodo requires every argument to be required for now. */
  @Override
  public boolean isOptional() {
    return isOptional;
  }
}
