package com.bodosql.calcite.adapter.pandas.window

import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.ir.Variable

class WindowAggregate internal constructor(private val groups: List<Group>, private val index: List<GroupIndex>) {
    fun emit(ctx: PandasRel.BuildContext): List<Variable> {
        // Emit the groups to the build context which should
        // create the variables we will access.
        val variables =
            groups.map { group ->
                group.emit(ctx)
            }

        // Now organize the variables from above to the output
        // format using the index.
        return index.map { (group, index) ->
            variables[group][index]
        }
    }
}
