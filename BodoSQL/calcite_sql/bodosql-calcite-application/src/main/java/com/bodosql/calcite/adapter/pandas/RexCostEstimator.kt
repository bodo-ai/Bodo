package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.plan.Cost
import org.apache.calcite.rel.metadata.RelMdSize
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.*
import org.apache.calcite.sql.type.SqlTypeName

object RexCostEstimator : RexVisitor<Cost> {
    // Input refs have no cost to either use or materialize.
    override fun visitInputRef(inputRef: RexInputRef): Cost = Cost()

    override fun visitLocalRef(localRef: RexLocalRef): Cost =
        throw UnsupportedOperationException()

    override fun visitLiteral(literal: RexLiteral): Cost {
        // Literals are a bit strange. They can produce a memory cost,
        // but only if they're materialized and not just part of another
        // operation.
        //
        // We're not really going to consider that case though at this stage
        // so just assign is a zero cost since that's probably what
        // it will be anyway.
        return Cost()
    }

    override fun visitCall(call: RexCall): Cost {
        // TODO(jsternberg): More complete usage but just doing something
        // very basic for now.
        // Base cost for this operation.
        var cost = Cost(
            cpu = if (call is RexOver) 10.0 else 1.0,
            mem = averageTypeValueSize(call.type) ?: 8.0,
        )
        // If there are operands, include them in the cost.
        if (call.operands.isNotEmpty()) {
            cost = cost.plus(
                call.operands.asSequence()
                    .map { op -> op.accept(this) }
                    .reduce { l, r -> l.plus(r) as Cost }
            ) as Cost
        }
        return cost
    }

    override fun visitOver(over: RexOver): Cost {
        // Same as call but use a base cost of 10 instead of 1.
        // The cost difference is represented in visitCall.
        return visitCall(over)
    }

    override fun visitCorrelVariable(correlVariable: RexCorrelVariable): Cost =
        // We don't support correlation variables.
        throw UnsupportedOperationException()

    override fun visitDynamicParam(param: RexDynamicParam): Cost =
        Cost(mem = averageTypeValueSize(param.type) ?: 0.0)

    override fun visitNamedParam(param: RexNamedParam): Cost =
        Cost(mem = averageTypeValueSize(param.type) ?: 0.0)

    override fun visitRangeRef(rangeRef: RexRangeRef): Cost =
        throw UnsupportedOperationException()

    override fun visitFieldAccess(fieldAccess: RexFieldAccess): Cost =
        throw UnsupportedOperationException()

    override fun visitSubQuery(subQuery: RexSubQuery): Cost =
        // Subqueries should have been removed by now.
        // We don't support them within any pandas operation.
        throw UnsupportedOperationException()

    override fun visitTableInputRef(tableInputRef: RexTableInputRef): Cost =
        visitInputRef(tableInputRef)

    override fun visitPatternFieldRef(patternFieldRef: RexPatternFieldRef): Cost =
        visitInputRef(patternFieldRef)

    /**
     * This is copied from RelMdSize in Calcite with some suitable defaults for
     * value sizes based on the sql type.
     */
    private fun averageTypeValueSize(type: RelDataType): Double? =
        when (type.sqlTypeName) {
            SqlTypeName.BOOLEAN, SqlTypeName.TINYINT -> 1.0
            SqlTypeName.SMALLINT -> 2.0
            SqlTypeName.INTEGER, SqlTypeName.REAL, SqlTypeName.DECIMAL, SqlTypeName.DATE, SqlTypeName.TIME, SqlTypeName.TIME_WITH_LOCAL_TIME_ZONE, SqlTypeName.INTERVAL_YEAR, SqlTypeName.INTERVAL_YEAR_MONTH, SqlTypeName.INTERVAL_MONTH -> 4.0
            SqlTypeName.BIGINT, SqlTypeName.DOUBLE, SqlTypeName.FLOAT, SqlTypeName.TIMESTAMP, SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE, SqlTypeName.INTERVAL_WEEK, SqlTypeName.INTERVAL_DAY, SqlTypeName.INTERVAL_DAY_HOUR, SqlTypeName.INTERVAL_DAY_MINUTE, SqlTypeName.INTERVAL_DAY_SECOND, SqlTypeName.INTERVAL_HOUR, SqlTypeName.INTERVAL_HOUR_MINUTE, SqlTypeName.INTERVAL_HOUR_SECOND, SqlTypeName.INTERVAL_MINUTE, SqlTypeName.INTERVAL_MINUTE_SECOND, SqlTypeName.INTERVAL_SECOND -> 8.0
            SqlTypeName.BINARY -> type.precision.coerceAtLeast(1).toDouble()
            SqlTypeName.VARBINARY -> type.precision.coerceAtLeast(1).toDouble().coerceAtMost(100.0)
            SqlTypeName.CHAR -> type.precision.coerceAtLeast(1).toDouble() * RelMdSize.BYTES_PER_CHARACTER
            SqlTypeName.VARCHAR ->       // Even in large (say VARCHAR(2000)) columns most strings are small
                (type.precision.coerceAtLeast(1).toDouble() * RelMdSize.BYTES_PER_CHARACTER).coerceAtMost(100.0)

            SqlTypeName.ROW -> {
                var average = 0.0
                for (field in type.fieldList) {
                    val size = averageTypeValueSize(field.type)
                    if (size != null) {
                        average += size
                    }
                }
                average
            }

            else -> null
        }
}
