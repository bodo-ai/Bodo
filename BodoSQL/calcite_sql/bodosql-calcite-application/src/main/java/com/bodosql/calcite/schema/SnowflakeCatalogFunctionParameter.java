package com.bodosql.calcite.schema;

import java.util.function.Function;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.schema.FunctionParameter;

/**
 * Implementation of FunctionParameter designed to be used with a function that is specific to the
 * Snowflake Catalog, such as INFORMATION_SCHEMAS.EXTERNAL_TABLE_FILES. Uses passed in lambdas that
 * produce the desired RelDataType of the parameter when a type factory is provided to them.
 */
public class SnowflakeCatalogFunctionParameter implements FunctionParameter {

  private final String name;
  private final int ord;

  private final boolean isOptional;
  private final Function<RelDataTypeFactory, RelDataType> typeMaker;

  public SnowflakeCatalogFunctionParameter(
      String name,
      int ord,
      Function<RelDataTypeFactory, RelDataType> typeMaker,
      boolean isOptional) {
    this.name = name;
    this.ord = ord;
    this.typeMaker = typeMaker;
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
    return typeMaker.apply(typeFactory);
  }

  /** Bodo requires every argument to be required for now. */
  @Override
  public boolean isOptional() {
    return isOptional;
  }
}
