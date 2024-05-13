package com.bodosql.calcite.adapter.common

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.adapter.bodo.RexToBodoTranslator
import com.bodosql.calcite.application.utils.BodoArrayHelpers
import com.bodosql.calcite.application.utils.IsScalar
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.Variable
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLocalRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexSlot

/**
 * Shared utils for rules generation or code generation for projections
 * in various adapters.
 */
class ProjectUtils {
    companion object {
        /**
         * Generates code to coerce a scalar value into an array.
         *
         * @param ctx The context for code generation.
         * @param dataType the sql data type for the array type.
         * @param scalar the expression that refers to the scalar value.
         * @param input the input dataframe.
         */
        private fun coerceScalarToArray(
            ctx: BodoPhysicalRel.BuildContext,
            dataType: RelDataType,
            scalar: Expr,
            input: BodoEngineTable,
        ): Expr {
            val global = ctx.lowerAsGlobal(BodoArrayHelpers.sqlTypeToBodoArrayType(dataType, true, ctx.getDefaultTZ().zoneExpr))
            return Expr.Call(
                "bodo.utils.conversion.coerce_scalar_to_array",
                scalar,
                Expr.Call("len", input),
                global,
            )
        }

        /**
         * This method constructs a new table from a logical table and additional series.
         *
         * The logical table is constructed from a set of indices. The indices refer to
         * either the input dataframe or one of the additional series provided in the series list.
         *
         * @param ctx The context for code generation.
         * @param input input table.
         * @param indices list of indices to initialize the table with.
         * @param seriesList additional series that should be included in the list of indices.
         * @param colsBeforeProject number of columns in the input table before any projection occurs.
         */
        private fun generateLogicalTableCode(
            ctx: BodoPhysicalRel.BuildContext,
            input: BodoEngineTable,
            indices: List<Int>,
            seriesList: List<Variable>,
            colsBeforeProject: Int,
        ): Variable {
            // Use the list of indices to generate a meta type with the column numbers.
            val metaType =
                ctx.lowerAsGlobal(
                    Expr.Call(
                        "MetaType",
                        Expr.Tuple(indices.map { Expr.IntegerLiteral(it) }),
                    ),
                )

            // Generate the output table with logical_table_to_table.
            val logicalTableExpr =
                Expr.Call(
                    "bodo.hiframes.table.logical_table_to_table",
                    input,
                    Expr.Tuple(seriesList),
                    metaType,
                    Expr.IntegerLiteral(colsBeforeProject),
                )
            val builder = ctx.builder()
            val logicalTableVar = builder.symbolTable.genTableVar()
            builder.add(Op.Assign(logicalTableVar, logicalTableExpr))
            return logicalTableVar
        }

        /**
         * Uses table_subset to create a projection from [RexInputRef] values.
         *
         * This function assumes the projection only contains [RexInputRef] values
         * and those values do not have duplicates. The method [canUseLoc]
         * should be invoked before calling this function.
         */
        private fun generateLocCode(
            ctx: BodoPhysicalRel.BuildContext,
            input: BodoEngineTable,
            projects: List<RexNode>,
        ): BodoEngineTable {
            val colIndices = projects.map { r -> Expr.IntegerLiteral((r as RexInputRef).index) }
            val typeCall = Expr.Call("MetaType", Expr.Tuple(colIndices))
            val colNamesMeta = ctx.lowerAsGlobal(typeCall)
            val resultExpr = Expr.Call("bodo.hiframes.table.table_subset", input, colNamesMeta, Expr.False)
            return ctx.returns(resultExpr)
        }

        /**
         * Generate the standard projection code. This is in contrast [generateLocCode]
         * which acts as just an index/rename operation.
         *
         * This is the general catch-all for most projections.
         */
        private fun generateProject(
            ctx: BodoPhysicalRel.BuildContext,
            inputVar: BodoEngineTable,
            translator: RexToBodoTranslator,
            projectExprs: List<RexNode>,
            localRefs: MutableList<Variable>,
            input: RelNode,
        ): BodoEngineTable {
            // Evaluate projections into new series.
            // In order to optimize this, we only generate new series
            // for projections that are non-trivial (aka not a RexInputRef)
            // or ones that haven't already been computed (aka not a RexLocalRef).
            // Similar to over expressions, we replace non-trivial projections
            // with a RexLocalRef that reference our computed set of local variables.
            val builder = ctx.builder()

            // newProjectRefs will be a list of RexSlot values (either RexInputRef or RexLocalRef).
            val newProjectRefs =
                projectExprs.map { proj ->
                    if (proj is RexSlot) {
                        return@map proj
                    }

                    val expr =
                        proj.accept(translator).let {
                            if (IsScalar.isScalar(proj)) {
                                coerceScalarToArray(ctx, proj.type, it, inputVar)
                            } else {
                                it
                            }
                        }
                    val arr = builder.symbolTable.genArrayVar()
                    builder.add(Op.Assign(arr, expr))
                    localRefs.add(arr)
                    RexLocalRef(localRefs.lastIndex, proj.type)
                }

            // Generate the indices we will reference when creating the table.
            val indices =
                newProjectRefs.map { proj ->
                    when (proj) {
                        is RexInputRef -> proj.index
                        is RexLocalRef -> proj.index + input.rowType.fieldCount
                        else -> throw AssertionError("Internal Error: Projection must be InputRef or LocalRef")
                    }
                }
            val logicalTableVar = generateLogicalTableCode(ctx, inputVar, indices, localRefs, input.rowType.fieldCount)
            return ctx.returns(logicalTableVar)
        }

        /**
         * Determines if we can use loc when generating code output.
         */
        private fun canUseLoc(projects: List<RexNode>): Boolean {
            val seen = hashSetOf<Int>()
            projects.forEach { r ->
                if (r !is RexInputRef) {
                    // If we have a non input ref we can't use the loc path
                    return false
                }

                if (r.index in seen) {
                    // When we have a situation with common subexpressions like "sum(A) as alias2, sum(A) as
                    // alias from table1 groupby D" Calcite generates a plan like: LogicalProject(alias2=[$1],
                    // alias=[$1]) LogicalAggregate(group=[{0}], alias=[SUM($1)]) In this case, we can't use
                    // loc, as it would lead to duplicate column names in the output dataframe See
                    // test_repeat_columns in BodoSQL/bodosql/tests/test_agg_groupby.py
                    return false
                }
                seen.add(r.index)
            }
            return true
        }

        @JvmStatic
        fun generateDataFrame(
            ctx: BodoPhysicalRel.BuildContext,
            inputVar: BodoEngineTable,
            translator: RexToBodoTranslator,
            projectExprs: List<RexNode>,
            localRefs: MutableList<Variable>,
            origProjects: List<RexNode>,
            input: RelNode,
        ): BodoEngineTable {
            return if (canUseLoc(origProjects)) {
                generateLocCode(ctx, inputVar, origProjects)
            } else {
                generateProject(ctx, inputVar, translator, projectExprs, localRefs, input)
            }
        }
    }
}
