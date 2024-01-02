package com.bodosql.calcite.application.operatorTables;

import java.util.List;
import org.apache.calcite.sql.SqlBasicFunction;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * Operator table for builtin functions used for special SELECT * syntax, which should all be
 * expanded during validation.
 */
public class SelectOperatorTable implements SqlOperatorTable {

  private static @Nullable SelectOperatorTable instance;

  // "SELECT * EXCLUDING (A, B, C) FROM T" is represented as
  // "SELECT STAR_EXCLUDE(*, (A, B, C)) FROM T".
  public static SqlFunction STAR_EXCLUDING =
      SqlBasicFunction.create(
          "STAR_EXCLUDING",
          // If derive type is called on a call to this function, it means it was not expanded
          // during star expansion, which means it was not a top-level star node.
          opBinding -> {
            throw new RuntimeException("Cannot have an EXCLUDING term except for a top-level *");
          },
          // Using a dummy operand type checker that allows anything to ensure that the error
          // thrown in type inference is used instead of one during type checking.
          OperandTypes.VARIADIC);

  // Dummy overloads that don't do anything because these functions should not exist
  // after star expansion, so they are not needed for typing information.

  @Override
  public void lookupOperatorOverloads(
      SqlIdentifier opName,
      @Nullable SqlFunctionCategory category,
      SqlSyntax syntax,
      List<SqlOperator> operatorList,
      SqlNameMatcher nameMatcher) {}

  @Override
  public List<SqlOperator> getOperatorList() {
    return List.of();
  }
}
