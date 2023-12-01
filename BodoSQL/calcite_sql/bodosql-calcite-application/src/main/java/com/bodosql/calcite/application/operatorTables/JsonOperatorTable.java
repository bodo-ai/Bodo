package com.bodosql.calcite.application.operatorTables;

import java.util.Arrays;
import java.util.List;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlBasicFunction;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.fun.SqlBasicAggFunction;
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SameOperandTypeChecker;
import org.apache.calcite.sql.type.SqlSingleOperandTypeChecker;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.apache.calcite.util.Optionality;
import org.checkerframework.checker.nullness.qual.Nullable;

public final class JsonOperatorTable implements SqlOperatorTable {

  private static @Nullable JsonOperatorTable instance;

  /** Returns the JSON operator table, creating it if necessary. */
  public static synchronized JsonOperatorTable instance() {
    JsonOperatorTable instance = JsonOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new JsonOperatorTable();
      JsonOperatorTable.instance = instance;
    }
    return instance;
  }

  public static final SqlSingleOperandTypeChecker SEMI_STRUCTURED =
      SemiStructuredOperandChecker.INSTANCE;

  public static final SqlSingleOperandTypeChecker OBJECT_DELETE_TYPE_CHECKER =
      ObjectDeleteOperandChecker.INSTANCE;

  public static final SameOperandTypeChecker OPERAND_CONSTRUCT_TYPE_CHECKER =
      ObjectConstructOperandChecker.INSTANCE;

  public static final SqlAggFunction OBJECT_AGG =
      SqlBasicAggFunction.create(
              "OBJECT_AGG",
              SqlKind.OTHER_FUNCTION,
              ReturnTypes.ARG1.andThen(BodoReturnTypes.TO_MAP),
              OperandTypes.family(SqlTypeFamily.STRING, SqlTypeFamily.ANY))
          .withGroupOrder(Optionality.FORBIDDEN)
          .withFunctionType(SqlFunctionCategory.SYSTEM);

  public static final SqlFunction GET_PATH =
      new SqlFunction(
          "GET_PATH",
          SqlKind.OTHER_FUNCTION,
          // Returns null if path is invalid
          BodoReturnTypes.VARIANT_FORCE_NULLABLE,
          null,
          OperandTypes.sequence(
              "GET_PATH(SEMI_STRUCTURED, STRING_LITERAL)", SEMI_STRUCTURED, OperandTypes.CHARACTER),
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction JSON_EXTRACT_PATH_TEXT =
      new SqlFunction(
          "JSON_EXTRACT_PATH_TEXT",
          SqlKind.OTHER_FUNCTION,
          // Cannot statically determine the precision
          // returns null if path is invalid
          BodoReturnTypes.VARCHAR_UNKNOWN_PRECISION_FORCE_NULLABLE,
          null,
          OperandTypes.STRING_STRING,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction OBJECT_DELETE =
      new SqlFunction(
          "OBJECT_DELETE",
          SqlKind.OTHER_FUNCTION,
          ReturnTypes.ARG0,
          null,
          OBJECT_DELETE_TYPE_CHECKER,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  // TODO(aneesh): [BSE-2122] this is using the same type checker as OBJECT_DELETE for now, but
  // OBJECT_PICK has an additional overload where the names to pick can be passed as an array
  // instead of a variadic list.
  public static final SqlBasicFunction OBJECT_PICK =
      SqlBasicFunction.create(
          "OBJECT_PICK",
          ReturnTypes.ARG0,
          OBJECT_DELETE_TYPE_CHECKER,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlFunction OBJECT_KEYS =
      new SqlFunction(
          "OBJECT_KEYS",
          SqlKind.OTHER_FUNCTION,
          opBinding -> {
            RelDataTypeFactory typeFactory = opBinding.getTypeFactory();
            RelDataType inputType = opBinding.getOperandType(0);
            return typeFactory.createArrayType(inputType.getKeyType(), -1);
          },
          null,
          OperandTypes.family(SqlTypeFamily.MAP),
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlBasicFunction OBJECT_CONSTRUCT_KEEP_NULL =
      SqlBasicFunction.create(
          "OBJECT_CONSTRUCT_KEEP_NULL",
          BodoReturnTypes.MAP_VARIANT,
          OPERAND_CONSTRUCT_TYPE_CHECKER,
          SqlFunctionCategory.USER_DEFINED_FUNCTION);

  public static final SqlBasicFunction OBJECT_CONSTRUCT =
      OBJECT_CONSTRUCT_KEEP_NULL.withName("OBJECT_CONSTRUCT");

  public static final SqlBasicFunction IS_ARRAY =
      SqlBasicFunction.create(
          "IS_ARRAY",
          ReturnTypes.BOOLEAN_NULLABLE,
          OperandTypes.family(SqlTypeFamily.ANY),
          SqlFunctionCategory.USER_DEFINED_FUNCTION);
  public static final SqlFunction IS_OBJECT = IS_ARRAY.withName("IS_OBJECT");

  private List<SqlOperator> functionList =
      Arrays.asList(
          GET_PATH,
          OBJECT_DELETE,
          OBJECT_PICK,
          JSON_EXTRACT_PATH_TEXT,
          OBJECT_KEYS,
          OBJECT_CONSTRUCT_KEEP_NULL,
          OBJECT_CONSTRUCT,
          OBJECT_AGG,
          IS_ARRAY,
          IS_OBJECT);

  @Override
  public void lookupOperatorOverloads(
      SqlIdentifier opName,
      @Nullable SqlFunctionCategory category,
      SqlSyntax syntax,
      List<SqlOperator> operatorList,
      SqlNameMatcher nameMatcher) {
    // Heavily copied from Calcite:
    // https://github.com/apache/calcite/blob/4bc916619fd286b2c0cc4d5c653c96a68801d74e/core/src/main/java/org/apache/calcite/sql/util/ListSqlOperatorTable.java#L57
    for (SqlOperator operator : functionList) {
      // All JSON Operators added are functions so far.

      if (syntax != operator.getSyntax()) {
        continue;
      }
      // Check that the name matches the desired names.
      if (!opName.isSimple() || !nameMatcher.matches(operator.getName(), opName.getSimple())) {
        continue;
      }
      // TODO: Check the category. The Lexing currently thinks
      //  all of these functions are user defined functions.
      operatorList.add(operator);
    }
  }

  @Override
  public List<SqlOperator> getOperatorList() {
    return functionList;
  }
}
