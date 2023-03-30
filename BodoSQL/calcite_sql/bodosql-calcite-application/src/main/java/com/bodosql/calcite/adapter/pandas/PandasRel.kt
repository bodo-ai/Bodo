package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Module
import org.apache.calcite.plan.Convention
import org.apache.calcite.rel.RelNode

interface PandasRel : RelNode {
    companion object {
        @JvmField
        val CONVENTION = Convention.Impl("PANDAS", PandasRel::class.java)
    }

    /**
     * Emits the code necessary for implementing this relational operator.
     *
     * @param builder the module builder to emit code into.
     * @param inputs function to emit the code necessary for the inputs and the variables for their dataframes.
     * @return the variable that represents this relational expression.
     */
    fun emit(builder: Module.Builder, inputs: () -> List<Dataframe>): Dataframe
}
