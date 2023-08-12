package com.bodosql.calcite.sql.func;

import static org.apache.calcite.util.Static.RESOURCE;

import com.google.common.collect.ImmutableList;
import java.util.List;
import org.apache.calcite.jdbc.CalciteSchema;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlSpecialOperator;
import org.apache.calcite.sql.SqlWriter;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorScope;
import org.apache.calcite.sql.validate.SqlValidatorUtil;

public class SqlNamedParameterOperator extends SqlSpecialOperator {

  public static SqlNamedParameterOperator INSTANCE = new SqlNamedParameterOperator();

  public SqlNamedParameterOperator() {
    super("NAMED_PARAM", SqlKind.OTHER_FUNCTION);
  }

  @Override
  public RelDataType deriveType(SqlValidator validator, SqlValidatorScope scope, SqlCall call) {
    SqlLiteral nameNode = (SqlLiteral) call.getOperandList().get(0);
    String name = nameNode.getValueAs(String.class).substring(1);

    // Named Param is always in the default schema with the encoded table name.
    String tableName = validator.config().namedParamTableName();
    if (tableName.isEmpty()) {
      throw validator.newValidationError(call, RESOURCE.namedParamTableNotRegistered());
    }
    // TODO: Set caseSensitive?
    CalciteSchema.TableEntry entry =
        SqlValidatorUtil.getTableEntry(validator.getCatalogReader(), ImmutableList.of(tableName));
    if (entry == null) {
      throw validator.newValidationError(call, RESOURCE.namedParamTableNotFound(tableName));
    }
    Table table = entry.getTable();
    RelDataType rowStruct = table.getRowType(validator.getTypeFactory());
    // TODO: Set caseSensitive?
    RelDataTypeField typeField = rowStruct.getField(name, false, false);
    if (typeField == null) {
      throw validator.newValidationError(call, RESOURCE.namedParamParameterNotFound(name));
    }
    return typeField.getType();
  }

  @Override
  public void unparse(SqlWriter writer, SqlCall call, int leftPrec, int rightPrec) {
    List<SqlNode> operandList = call.getOperandList();
    String name = ((SqlLiteral) operandList.get(0)).getValueAs(String.class);
    writer.print(name);
  }

  @Override
  public boolean isDynamicFunction() {
    // This prevents reduce expressions rule from reducing expressions with named parameters.
    return true;
  }
}
