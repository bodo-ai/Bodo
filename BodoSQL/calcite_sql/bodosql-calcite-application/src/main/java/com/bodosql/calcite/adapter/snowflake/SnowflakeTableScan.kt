package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.table.CatalogTableImpl
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.core.TableScan
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFieldImpl
import org.apache.calcite.rel.type.RelRecordType

class SnowflakeTableScan(cluster: RelOptCluster?, traitSet: RelTraitSet?, table: RelOptTable?, val catalogTable: CatalogTableImpl, private val preserveCase: Boolean) :
    TableScan(cluster, traitSet, ImmutableList.of(), table) {

    constructor(cluster: RelOptCluster?, table: RelOptTable?, catalogTable: CatalogTableImpl) : this(cluster, cluster?.traitSet(), table, catalogTable, false)

    override fun deriveRowType(): RelDataType {
        return table.rowType.let { rowType ->
            if (preserveCase) {
                RelRecordType(rowType.fieldList.map { field ->
                    val name = catalogTable.getPreservedColumnName(field.name)
                    RelDataTypeFieldImpl(name, field.index, field.type)
                })
            } else rowType
        }
    }

    override fun copy(traitSet: RelTraitSet?, inputs: MutableList<RelNode>?): RelNode {
        return SnowflakeTableScan(cluster, traitSet, table, catalogTable, preserveCase)
    }

    // This is kind of a nasty hack to preserve behavior on the Python side
    // while updating the Java code to more formally handle pushdown operations.
    // The Python code does not preserve casing. Instead, it automatically
    // maps the default uppercase schema of snowflake to python's sqlalchemy
    // default of lowercase. This creates a user interface that results in
    // the column names being lowercase.
    //
    // On the other hand, we need to preserve the original casing for our own
    // purposes. Mostly, SQL generation. The proper way to do this is to create
    // a calcite convention and have the converter implementation deal with this
    // logic. That requires a bit too much code during the refactoring of the
    // Java code. So, this exists to selectively preserve the case.
    // The Aggregate pushdown turns this off and this defaults to on.
    //
    // There is another way which is to automatically generate a projection
    // that forces the columns to lowercase when we create the table scan,
    // but the AliasPreservingProjectJoinTransposeRule gets confused and
    // thinks the aliases are meaningful in areas where they aren't which prevents
    // the pushdown from happening.
    fun withPreserveCase(preserveCase: Boolean): SnowflakeTableScan {
        return SnowflakeTableScan(this.cluster, this.traitSet, this.table, this.catalogTable, preserveCase)
    }

    override fun explainTerms(pw: RelWriter?): RelWriter {
        // Necessary for the digest to be different.
        // Remove when we have proper converters.
        return super.explainTerms(pw)
            .item("preserveCase", preserveCase)
    }

    override fun register(planner: RelOptPlanner?) {
        planner?.addRule(SnowflakeAggregateRule.Config.DEFAULT.toRule())
        planner?.addRule(SnowflakeAggregateRule.Config.WITH_FILTER.toRule())
    }
}
