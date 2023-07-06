package com.bodosql.calcite.adapter.pandas.window

import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Expr
import org.apache.calcite.rex.RexLocalRef
import org.apache.calcite.rex.RexNode

internal class OperandResolverImpl(ctx: PandasRel.BuildContext, input: Dataframe, val fields: List<Field>) :
    OperandResolver {
    private val rexTranslator = ctx.rexTranslator(input)
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
        // the RexTranslator and then insert it as an additional
        // column we can reference by name.
        val argExpr = node.accept(rexTranslator)

        // Store this expression with a generated name so it
        // can be embedded within the passed in Dataframe.
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
}
