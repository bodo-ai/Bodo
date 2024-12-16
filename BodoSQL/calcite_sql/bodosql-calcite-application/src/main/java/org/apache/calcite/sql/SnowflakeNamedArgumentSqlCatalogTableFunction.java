package org.apache.calcite.sql;

import com.bodosql.calcite.catalog.SnowflakeCatalog;
import com.bodosql.calcite.table.ColumnDataTypeInfo;
import kotlin.Pair;
import org.apache.calcite.linq4j.Ord;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.schema.Function;
import org.apache.calcite.schema.FunctionParameter;
import org.apache.calcite.schema.TableFunction;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SnowflakeNamedOperandMetadataImpl;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlReturnTypeInference;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.jetbrains.annotations.NotNull;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

import static org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE;
import static org.apache.calcite.util.Static.RESOURCE;


/**
 * Subclass of SnowflakeNamedArgumentSqlTableFunction for table functions with named arguments that are
 * connected to a specific Snowflake Catalog/Database.
 */
public class SnowflakeNamedArgumentSqlCatalogTableFunction extends SnowflakeNamedArgumentSqlTableFunction implements Function, TableFunction {
    private final SnowflakeCatalog catalog;
    private final List<String> functionPath;
    private final List<FunctionParameter> parameters;
    private final List<Pair<String, java.util.function.Function<RelDataTypeFactory, RelDataType>>> returnInfo;

    /**
     * Package private. Use SnowflakeNamedArgumentSqlCatalogTableFunction.create() instead.
     *
     */
    SnowflakeNamedArgumentSqlCatalogTableFunction(String name, SqlReturnTypeInference rowTypeInference, @NotNull SnowflakeNamedOperandMetadataImpl operandTypeChecker, FunctionType type, int tableOperandNum, TableCharacteristic.Semantics semantics,
                                                  List<FunctionParameter> parameters,
                                                  List<Pair<String, java.util.function.Function<RelDataTypeFactory, RelDataType>>> returnInfo,
                                                  SnowflakeCatalog catalog,
                                                  List<String> functionPath) {
        super(name, rowTypeInference, operandTypeChecker, type, tableOperandNum, semantics);
        this.catalog = catalog;
        this.functionPath = functionPath;
        this.parameters = parameters;
        this.returnInfo = returnInfo;
    }

    public SnowflakeCatalog getCatalog() {
        return catalog;
    }

    public List<String> getFunctionPath() {
        return functionPath;
    }


    public static SnowflakeNamedArgumentSqlCatalogTableFunction create(
            String name,
            SqlReturnTypeInference rowTypeInference,
            @NotNull SnowflakeNamedOperandMetadataImpl operandTypeChecker,
            FunctionType type,
            int tableOperandNum,
            TableCharacteristic.Semantics semantics,
            List<FunctionParameter> parameters,
            List<Pair<String, java.util.function.Function<RelDataTypeFactory, RelDataType>>> returnInfo,
            SnowflakeCatalog catalog,
            List<String> functionPath) {
        return new SnowflakeNamedArgumentSqlCatalogTableFunction(name, rowTypeInference, operandTypeChecker, type, tableOperandNum, semantics, parameters, returnInfo, catalog, functionPath);
    }


    public List<FunctionParameter> getParameters() {
        return parameters;
    }

    @Override
    public RelDataType getRowType(RelDataTypeFactory typeFactory, List<?> arguments) {
        // Convert the pairs into lists for generating a row type.
        List<String> fieldNames = new ArrayList();
        List<RelDataType> fieldTypes = new ArrayList();
        for (Pair<String, java.util.function.Function<RelDataTypeFactory, RelDataType>> output: returnInfo) {
            fieldNames.add(output.component1());
            fieldTypes.add(output.component2().apply(typeFactory));
        }
        return typeFactory.createStructType(fieldTypes, fieldNames);
    }

    @Override
    public Type getElementType(List<?> arguments) {
        return null;
    }
}
