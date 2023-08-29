package org.apache.calcite.sql.type;

import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlTypeNameSpec;
import org.apache.calcite.sql.SqlTzAwareTypeNameSpec;
import org.apache.calcite.sql.VariantTypeNameSpec;
import org.apache.calcite.sql.parser.SqlParserPos;

public class BodoSqlTypeUtil {
  public static SqlDataTypeSpec convertTypeToSpec(RelDataType type) {
    if (type instanceof TZAwareSqlType) {
      SqlTypeNameSpec typeNameSpec = new SqlTzAwareTypeNameSpec((TZAwareSqlType) type);
      return new SqlDataTypeSpec(typeNameSpec, SqlParserPos.ZERO);
    } else if (type instanceof VariantSqlType) {
      SqlTypeNameSpec typeNameSpec = new VariantTypeNameSpec((VariantSqlType) type);
      return new SqlDataTypeSpec(typeNameSpec, SqlParserPos.ZERO);
    }
    return SqlTypeUtil.convertTypeToSpec(type);
  }
}
