package com.bodosql.calcite.application.operatorTables;

import com.bodosql.calcite.catalog.SnowflakeCatalog;
import com.bodosql.calcite.schema.SnowflakeCatalogFunctionParameter;
import java.math.BigDecimal;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;
import kotlin.Pair;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.FunctionParameter;
import org.apache.calcite.sql.SnowflakeNamedArgumentSqlCatalogTableFunction;
import org.apache.calcite.sql.SnowflakeNamedArgumentSqlTableFunction;
import org.apache.calcite.sql.SnowflakeSqlTableFunction;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.TableCharacteristic;
import org.apache.calcite.sql.type.BodoReturnTypes;
import org.apache.calcite.sql.type.SnowflakeNamedOperandMetadataImpl;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Operator table for builtin functions that return a table, such as FLATTEN and SPLIT_TO_TABLE. */
public class TableFunctionOperatorTable implements SqlOperatorTable {

  private static @Nullable TableFunctionOperatorTable instance;

  /** Returns the Table Function operator table, creating it if necessary. */
  public static synchronized TableFunctionOperatorTable instance() {
    TableFunctionOperatorTable instance = TableFunctionOperatorTable.instance;
    if (instance == null) {
      // Creates and initializes the standard operator table.
      // Uses two-phase construction, because we can't initialize the
      // table until the constructor of the sub-class has completed.
      instance = new TableFunctionOperatorTable();
      TableFunctionOperatorTable.instance = instance;
    }
    return instance;
  }

  private static final SnowflakeNamedOperandMetadataImpl FLATTEN_OPERAND_METADATA =
      SnowflakeNamedOperandMetadataImpl.create(
          List.of(
              SqlTypeFamily.ANY,
              SqlTypeFamily.CHARACTER,
              SqlTypeFamily.BOOLEAN,
              SqlTypeFamily.BOOLEAN,
              SqlTypeFamily.CHARACTER),
          List.of("INPUT", "PATH", "OUTER", "RECURSIVE", "MODE"),
          1,
          List.of(false, true, true, true, true),
          (SqlLiteral literal, int argNumber) -> {
            if (argNumber == 4) {
              // Must be one of 'OBJECT', 'ARRAY', or 'BOTH'.
              if (SqlTypeName.CHAR_TYPES.contains(literal.getTypeName())) {
                String value = literal.getValueAs(String.class).toUpperCase(Locale.ROOT);
                return List.of("OBJECT", "ARRAY", "BOTH").contains(value);
              } else {
                return false;
              }
            }
            // Note the default is no restrictions since this is only intended for APIs that
            // restrict
            // below type requirements.
            return true;
          },
          (RexBuilder builder, int i) -> {
            final RexNode literal;
            if (i == 1) {
              literal = builder.makeLiteral("");
            } else if (i == 2) {
              literal = builder.makeLiteral(false);
            } else if (i == 3) {
              literal = builder.makeLiteral(false);
            } else if (i == 4) {
              literal = builder.makeLiteral("BOTH");
            } else {
              throw new RuntimeException("Invalid input");
            }
            return literal;
          });
  public static final SnowflakeSqlTableFunction FLATTEN =
      SnowflakeNamedArgumentSqlTableFunction.create(
          "FLATTEN",
          BodoReturnTypes.FLATTEN_RETURN_TYPE,
          FLATTEN_OPERAND_METADATA,
          SnowflakeSqlTableFunction.FunctionType.FLATTEN,
          0,
          TableCharacteristic.Semantics.ROW);

  private static final SnowflakeNamedOperandMetadataImpl GENERATOR_OPERAND_METADATA =
      SnowflakeNamedOperandMetadataImpl.create(
          List.of(SqlTypeFamily.INTEGER, SqlTypeFamily.INTEGER),
          List.of("ROWCOUNT", "TIMELIMIT"),
          // Generator technically does not require either of its arguments to be provided, but
          // this is a simple way of helping to enforce that the ROWCOUNT argument must be provided.
          1,
          List.of(true, true),
          (SqlLiteral literal, int argNumber) -> {
            // Currently only allows ROWCOUNT
            if (argNumber != 0) {
              throw new RuntimeException(
                  "Function \"GENERATOR\" does not currently allow providing any arguments except"
                      + " \"ROWCOUNT\".");
            }
            // The literal must be a non-negative integer, or null.
            if (SqlTypeName.INT_TYPES.contains(literal.getTypeName())
                || literal.getTypeName() == SqlTypeName.DECIMAL) {
              BigDecimal value = literal.getValueAs(BigDecimal.class);
              return value.compareTo(BigDecimal.ZERO) >= 0;
            } else {
              return false;
            }
          },
          (RexBuilder builder, int i) -> {
            final RexNode literal;
            if (i == 0 || i == 1) {
              literal = builder.makeNullLiteral(SqlTypeName.INTEGER);
            } else {
              throw new RuntimeException("Invalid input");
            }
            return literal;
          });

  public static final SnowflakeSqlTableFunction GENERATOR =
      SnowflakeNamedArgumentSqlTableFunction.create(
          "GENERATOR",
          BodoReturnTypes.GENERATOR_RETURN_TYPE,
          GENERATOR_OPERAND_METADATA,
          SnowflakeSqlTableFunction.FunctionType.GENERATOR,
          0,
          TableCharacteristic.Semantics.ROW);

  private static final SnowflakeNamedOperandMetadataImpl EXTERNAL_TABLE_FILES_OPERAND_METADATA =
      SnowflakeNamedOperandMetadataImpl.create(
          List.of(SqlTypeFamily.CHARACTER),
          List.of("TABLE_NAME"),
          1,
          List.of(true),
          (SqlLiteral literal, int argNumber) -> true,
          (RexBuilder builder, int i) -> null);

  public static final String EXTERNAL_TABLE_FILES_NAME = "EXTERNAL_TABLE_FILES";

  // Returns a new EXTERNAL_TABLE_FILES function given a catalog and path definition
  public static final SnowflakeNamedArgumentSqlCatalogTableFunction makeExternalTableFiles(
      SnowflakeCatalog catalog, List<String> functionPath) {
    List<FunctionParameter> parameters =
        List.of(
            new SnowflakeCatalogFunctionParameter(
                "TABLE_NAME", 0, (factory) -> factory.createSqlType(SqlTypeName.CHAR), true));
    Function<RelDataTypeFactory, RelDataType> charFunc =
        (factory) ->
            factory.createTypeWithNullability(
                factory.createSqlType(SqlTypeName.VARCHAR, -1), false);
    Function<RelDataTypeFactory, RelDataType> intFunc =
        (factory) ->
            factory.createTypeWithNullability(factory.createSqlType(SqlTypeName.BIGINT), false);
    Function<RelDataTypeFactory, RelDataType> ltzFunc =
        (factory) ->
            factory.createTypeWithNullability(
                factory.createSqlType(SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE), false);
    List<Pair<String, Function<RelDataTypeFactory, RelDataType>>> outputs =
        List.of(
            new Pair("FILE_NAME", charFunc),
            new Pair("REGISTERED_ON", ltzFunc),
            new Pair("FILE_SIZE", intFunc),
            new Pair("LAST_MODIFIED", ltzFunc),
            new Pair("ETAG", charFunc),
            new Pair("MD5", charFunc));
    return SnowflakeNamedArgumentSqlCatalogTableFunction.create(
        EXTERNAL_TABLE_FILES_NAME,
        BodoReturnTypes.EXTERNAL_TABLE_FILES_RETURN_TYPE,
        EXTERNAL_TABLE_FILES_OPERAND_METADATA,
        SnowflakeSqlTableFunction.FunctionType.EXTERNAL_TABLE_FILES,
        0,
        TableCharacteristic.Semantics.ROW,
        parameters,
        outputs,
        catalog,
        functionPath);
  }

  private List<SqlOperator> functionList = Arrays.asList(FLATTEN, GENERATOR);

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
      // All Table Function Operators are functions so far.
      SqlFunction func = (SqlFunction) operator;
      if (syntax != func.getSyntax()) {
        continue;
      }
      // Check that the name matches the desired names.
      if (!opName.isSimple() || !nameMatcher.matches(func.getName(), opName.getSimple())) {
        continue;
      }
      // TODO: Check the category. The Lexing currently thinks
      //  all of these functions are user defined functions.
      operatorList.add(func);
    }
  }

  @Override
  public List<SqlOperator> getOperatorList() {
    return functionList;
  }
}
