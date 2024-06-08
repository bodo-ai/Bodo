package com.bodosql.calcite.plan

import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelOptUtil
import java.text.DecimalFormat
import kotlin.math.abs

/**
 * Represents the cost of a relational operation.
 *
 * The cost is composed of the cpu, io, and mem used to complete an operation.
 * If a resource is divided between multiple processing units, the cost would be
 * the usage of each processing unit. As an example, a parallel operation
 * might be divided by the number of ranks it can effectively utilize so an operation
 * that uses two ranks effectively costs half for the same operation that uses
 * one rank.
 *
 * The row count for an operation is included as part of the cost, but is
 * not used for cost comparisons.
 */
class Cost private constructor(
    private val rows: Double,
    private val cpu: Double,
    private val io: Double,
    val mem: Double,
    cost: Double?,
) : RelOptCost {
    // Assign a fixed cost, so we prioritize fewer operators
    // in the case of a tie.
    private val fixedOperatorCost = 0.1

    /**
     * Initializes a cost with the given resources.
     *
     * @param rows the number of rows produced by this operation.
     * @param cpu amount of cpu resource utilized by this operation.
     * @param io amount of io resource utilized by this operation.
     * @param mem amount of mem resource utilized by this operation.
     */
    constructor(rows: Double = 0.0, cpu: Double = 0.0, io: Double = 0.0, mem: Double = 0.0) : this(
        rows,
        cpu,
        io,
        mem,
        null,
    )

    /**
     * The cost represents the single numeric value that determines
     * how expensive this operation is.
     *
     * While costs include information about rows, cpu, io, etc
     * for debugging, the cost is the only thing used for comparisons.
     *
     * We can use whichever method we find most appropriate for this
     * cost calculation. For now, I'm just equally dividing it between
     * the cpu, io, and memory components. Row count is not considered.
     * This is because row count is mostly represented in the other
     * metrics.
     *
     * For actual comparisons we wrap this behind totalCost so
     * we can assign a fixed value to each operator.
     */
    val value = cost ?: cpu + io + mem

    fun totalCost(): Double = value + fixedOperatorCost

    /**
     * Visually shows the cost value as a formatted string.
     */
    val valueString: String get() = df(totalCost())

    override fun equals(other: RelOptCost): Boolean {
        return if (other is Cost) {
            return rows == other.rows &&
                cpu == other.cpu &&
                io == other.io &&
                mem == other.mem
        } else {
            false
        }
    }

    override fun getRows(): Double = rows

    override fun getCpu(): Double = cpu

    override fun getIo(): Double = io

    override fun isInfinite(): Boolean = totalCost().isInfinite()

    override fun isEqWithEpsilon(other: RelOptCost): Boolean = isEqWithEpsilon(convert(other))

    private fun isEqWithEpsilon(other: Cost): Boolean = abs(other.totalCost() - this.totalCost()) < RelOptUtil.EPSILON

    override fun isLe(other: RelOptCost): Boolean = isLe(convert(other))

    private fun isLe(other: Cost): Boolean = this.totalCost() <= other.totalCost()

    override fun isLt(other: RelOptCost): Boolean = isLt(convert(other))

    private fun isLt(other: Cost): Boolean = this.totalCost() < other.totalCost()

    override fun plus(other: RelOptCost): RelOptCost {
        return convert(other).let { c ->
            Cost(
                rows = rows + c.rows,
                cpu = cpu + c.cpu,
                io = io + c.io,
                mem = mem + c.mem,
                cost = value + c.value,
            )
        }
    }

    override fun minus(other: RelOptCost): RelOptCost {
        return convert(other).let { c ->
            Cost(
                rows = rows - c.rows,
                cpu = cpu - c.cpu,
                io = io - c.io,
                mem = mem - c.mem,
                cost = value - c.value,
            )
        }
    }

    override fun multiplyBy(factor: Double): RelOptCost {
        return Cost(
            rows = rows * factor,
            cpu = cpu * factor,
            io = io * factor,
            mem = mem * factor,
            cost = value * factor,
        )
    }

    override fun divideBy(other: RelOptCost): Double {
        return convert(other).let { c -> value / c.value }
    }

    override fun toString(): String = "{${df(rows)} rows, ${df(cpu)} cpu, ${df(io)} io, ${df(mem)} mem}".format(rows, cpu, io, mem)

    companion object {
        private val DECIMAL_FORMAT = DecimalFormat("0.######E0")

        private fun df(decimal: Double): String =
            DECIMAL_FORMAT
                // Format in engineering notation (exponent is always a multiple of thousands).
                .format(decimal)
                // Remove extraneous E0.
                .replace("E0", "")
                // E to e because I find it easier to read.
                .replace("E", "e")

        @JvmField
        val INFINITY: Cost = Cost(0.0, 0.0, 0.0, 0.0, Double.POSITIVE_INFINITY)

        private fun convert(cost: RelOptCost): Cost =
            if (cost is Cost) {
                cost
            } else {
                Cost(cost.rows, cost.cpu, cost.io, 0.0)
            }
    }
}

/**
 * Convenience method for producing a cost.
 *
 * This function will determine if the cost factory supports the extra parameters such as memory
 * and will use those if available. Otherwise, it will pass the row count, cpu, and io as they are
 * and ignore the other parameters.
 *
 * All parameters are optional. If row count is not specified, then this cost is assumed to be
 * for a single row. This can be useful if the cost scales linearly based on row count since
 * you can make a cost for a single row and then multiply by the number of rows.
 *
 * @param rows the number of rows produced by this operation.
 * @param cpu amount of cpu resource utilized by this operation.
 * @param io amount of io resource utilized by this operation.
 * @param mem amount of mem resource utilized by this operation.
 */
fun RelOptPlanner.makeCost(
    rows: Double = 1.0,
    cpu: Double = 0.0,
    io: Double = 0.0,
    mem: Double = 0.0,
): RelOptCost {
    return when (val factory = costFactory) {
        is CostFactory -> factory.makeCost(rows, cpu, io, mem)
        else -> factory.makeCost(rows, cpu, io)
    }
}

/**
 * Convenience method for producing a cost from another cost.
 *
 * This uses the parameters from the passed in cost but replaces
 * the number of rows.
 */
fun RelOptPlanner.makeCost(
    rows: Double = 1.0,
    from: RelOptCost,
): RelOptCost =
    when (from) {
        is Cost -> makeCost(rows, from = from)
        else -> makeCost(rows, cpu = from.cpu, io = from.io)
    }

/**
 * Convenience method for producing a cost from another cost.
 *
 * This uses the parameters from the passed in cost but replaces
 * the number of rows.
 */
fun RelOptPlanner.makeCost(
    rows: Double = 1.0,
    from: Cost,
): RelOptCost = makeCost(rows, cpu = from.cpu, io = from.io, mem = from.mem)
