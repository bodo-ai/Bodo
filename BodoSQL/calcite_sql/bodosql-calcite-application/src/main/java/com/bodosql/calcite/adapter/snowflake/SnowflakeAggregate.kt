package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.application.PandasRel
import com.bodosql.calcite.catalog.SnowflakeCatalogImpl
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Module
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.table.CatalogTableImpl
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.rel2sql.RelToSqlConverter
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFieldImpl
import org.apache.calcite.rel.type.RelRecordType
import org.apache.calcite.util.ImmutableBitSet

class SnowflakeAggregate(
    cluster: RelOptCluster?,
    traitSet: RelTraitSet?,
    input: RelNode,
    groupSet: ImmutableBitSet,
    groupSets: List<ImmutableBitSet>?,
    aggCalls: List<AggregateCall>,
    val catalogTable: CatalogTableImpl,
    private val preserveCase: Boolean = false,
) :
    Aggregate(cluster, traitSet, listOf(), input, groupSet, groupSets, aggCalls),
    PandasRel
{

    override fun emit(builder: Module.Builder, inputs: () -> List<Dataframe>): Dataframe {
        // TODO(jsternberg): The catalog will specifically be SnowflakeCatalogImpl.
        // This cast is a bad idea and is particularly unsafe and unverifiable using
        // the compiler tools. It would be better if the catalog implementations were
        // refactored to not be through an interface and we had an actual class type
        // that referenced snowflake than needing to do it through a cast.
        // That's a bit too much work to refactor quite yet, so this cast gets us
        // through this time where the code is too abstract and we just need a way
        // to convert over.
        val catalog = catalogTable.catalog as SnowflakeCatalogImpl
        val expr = Expr.Call("pd.read_sql",
            args = listOf(
                // First argument is the sql. This should escape itself.
                Expr.StringLiteral(toSql()),
                // Second argument is the connection string.
                // We don't use a schema name because we've already fully qualified
                // all table references and it's better if this doesn't have any
                // potentially unexpected behavior.
                Expr.StringLiteral(catalog.generatePythonConnStr("")),
            ),
            namedArgs = listOf(
                // Special parameter to read date as dt64.
                "_bodo_read_date_as_dt64" to Expr.BooleanLiteral(builder.useDateRuntime),
                // We do not include _bodo_is_table_input because this is not a table input.
            )
        )

        val df = builder.genDataframe(this)
        builder.add(Op.Assign(df.variable, expr))
        return df
    }

    override fun deriveRowType(): RelDataType {
        if (preserveCase) {
            // Preserve case is only set when all inputs are also preserved.
            // We get the correct value by just going down the tree.
            return super.deriveRowType()
        }

        // Preserve case is false only when the inputs are preserved.
        // We need to convert the input's field list back to the read names
        // just so we can derive the correct type.
        val inputRowType = RelRecordType(input.rowType.fieldList.map { field ->
            val name = if (field.name.equals(field.name.uppercase())) {
                field.name.lowercase()
            } else field.name
            RelDataTypeFieldImpl(name, field.index, field.type)
        })
        return deriveRowType(cluster.typeFactory, inputRowType, false, groupSet, groupSets, aggCalls)
    }

    private fun withPreserveCase(preserveCase: Boolean): SnowflakeAggregate {
        return SnowflakeAggregate(this.cluster, this.traitSet, this.input,
            this.groupSet, this.groupSets, this.aggCalls,
            catalogTable, preserveCase = preserveCase)
    }

    override fun explainTerms(pw: RelWriter?): RelWriter {
        // Necessary for the digest to be different.
        // Remove when we have proper converters.
        return super.explainTerms(pw)
            .item("preserveCase", preserveCase)
    }

    override fun copy(
        traitSet: RelTraitSet?,
        input: RelNode,
        groupSet: ImmutableBitSet,
        groupSets: List<ImmutableBitSet>?,
        aggCalls: List<AggregateCall>
    ): Aggregate {
        return SnowflakeAggregate(cluster, traitSet, input, groupSet, groupSets, aggCalls, catalogTable, preserveCase)
    }

    private fun toSql(): String {
        // Use the snowflake dialect for generating the sql string.
        val rel2sql = RelToSqlConverter(SnowflakeSqlDialect.DEFAULT)
        val result = rel2sql.visitRoot(this.withPreserveCase(true))
        return result.asSelect()
            .toSqlString { c ->
                c.withClauseStartsLine(false)
                    .withDialect(SnowflakeSqlDialect.DEFAULT) }
            .toString()
    }
}
