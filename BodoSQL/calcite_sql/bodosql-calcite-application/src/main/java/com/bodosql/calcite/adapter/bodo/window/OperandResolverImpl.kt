package com.bodosql.calcite.adapter.bodo.window

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexLocalRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexWindowBound
import java.math.BigDecimal

internal class OperandResolverImpl(
    ctx: BodoPhysicalRel.BuildContext,
    input: BodoEngineTable,
    val fields: List<Field>,
) : OperandResolver {
    private val rexTranslator = ctx.rexTranslator(input)

    // Translator for args that must always be arrays.
    private val arrayRexTranslator = ctx.arrayRexTranslator(input)
    private val _extraFields: MutableList<Pair<String, Expr>> = mutableListOf()
    val extraFields: List<Pair<String, Expr>> get() = _extraFields.toList()

    override fun series(node: RexNode): Expr {
        // In general, field construction will usually result
        // in the series inputs being wrapped in a RexLocalRef since
        // series results usually reference a column and columns
        // get processed into the list of fields.
        if (node is RexLocalRef) {
            // The window function arguments reference series through
            // their name in the dataframe as a string literal.
            return Expr.StringLiteral(fields[node.index].name)
        }

        // We have an expression that has not been processed
        // into a series-compatible argument. Evaluate it using
        // the ArrayRexTranslator and then insert it as an additional
        // column we can reference by name.
        val argExpr = arrayRexTranslator.apply(node)

        // Store this expression with a generated name so it
        // can be embedded within the passed in Table.
        val name = "ARG_COL_${extraFields.size}"
        _extraFields.add(Pair(name, argExpr))
        return Expr.StringLiteral(name)
    }

    override fun scalar(node: RexNode): Expr {
        // Scalars are evaluated directly by the rex translator
        // and then inserted as string literals.
        val arg = node.accept(rexTranslator)
        return Expr.StringLiteral(arg.emit())
    }

    override fun bound(node: RexWindowBound?): Expr =
        node?.let {
            when {
                node.isUnbounded -> Expr.StringLiteral("None")
                node.isPreceding ->
                    Expr.Unary(
                        "-",
                        Expr.IntegerLiteral(
                            (node.offset as RexLiteral)
                                .getValueAs(BigDecimal::class.java)!!
                                .intValueExact(),
                        ),
                    )
                node.isFollowing -> Expr.IntegerLiteral((node.offset as RexLiteral).getValueAs(BigDecimal::class.java)!!.intValueExact())
                node.isCurrentRow -> Expr.Zero
                else -> throw AssertionError("invalid window bound")
            }
        } ?: Expr.StringLiteral("None")
}
