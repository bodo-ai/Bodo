package org.apache.calcite.sql

import com.bodosql.calcite.rel.type.BodoRelDataTypeFactory
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.parser.SqlParserPos
import org.apache.calcite.sql.validate.SqlValidator
import org.apache.calcite.util.Litmus
import java.util.*

class VariantTypeNameSpec(identifier: SqlIdentifier) : SqlTypeNameSpec(
    identifier,
    SqlParserPos.ZERO,
) {

    override fun deriveType(validator: SqlValidator): RelDataType? {
        return BodoRelDataTypeFactory.createVariantSqlType(validator.typeFactory)
    }

    override fun unparse(writer: SqlWriter, leftPrec: Int, rightPrec: Int) {
        writer.keyword("VARIANT")
    }

    override fun equalsDeep(spec: SqlTypeNameSpec, litmus: Litmus): Boolean {
        return if (spec !is VariantTypeNameSpec) {
            litmus.fail("{} != {}", this, spec)
        } else {
            litmus.succeed()
        }
    }
}
