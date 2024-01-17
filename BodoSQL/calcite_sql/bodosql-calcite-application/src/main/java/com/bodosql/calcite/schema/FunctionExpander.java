package com.bodosql.calcite.schema;

import com.google.common.collect.ImmutableList;
import java.util.Map;
import org.apache.calcite.rel.type.RelDataType;
import org.checkerframework.checker.nullness.qual.NonNull;

/**
 * Interface that defines the API used to "inline" or "expand" the contents of a function body. This
 * is intended to be used for inlining UDFs that are defined with a known function body.
 */
public interface FunctionExpander {

  /**
   * Inline the body of a function. This API is responsible for parsing the function body as a query
   * and validating the contents query for type stability.
   *
   * <p>This API is still under active development, so the return type is not yet finalized and
   * additional arguments are likely to be added.
   *
   * @param functionBody Body of the function.
   * @param functionPath Path of the function.
   * @param paramNameToTypeMap Mapping from the name of each parameter as an identifier to its
   *     expected argument type. This is needed for validation.
   */
  void expandFunction(
      @NonNull String functionBody,
      @NonNull ImmutableList<@NonNull String> functionPath,
      @NonNull Map<@NonNull String, @NonNull RelDataType> paramNameToTypeMap);
}
