package com.bodosql.calcite.ir

/**
 * Symbol table holds information about the defined variables
 * and allows programs to define new variables.
 *
 * NOTE(jsternberg): The intended use of this structure is defined
 * by the comment above, but the current use of it is just a class
 * dedicated to generating unique variables for code generation.
 * It'll likely be expanded in the future, but this is the
 * functionality that's presently needed at the time that I wrote this.
 */
class SymbolTable {
    private var dfVarId: Int = 1
    private var colVarId: Int = 1
    private var groupByApplyFnId: Int = 1
    private var globalVarId: Int = 1
    private var idxVarId: Int = 1

    companion object {
        private const val dummyColNameBase = "__bodo_dummy__"
    }

    fun genDfVar(): Variable {
        return Variable("df${dfVarId++}")
    }

    fun genTableVar(): Variable {
        return Variable("T${dfVarId++}")
    }

    fun genSeriesVar(): Variable {
        return Variable("S${dfVarId++}")
    }
    fun genIndexVar(): Variable {
        return Variable("index_${idxVarId++}")
    }

    fun genGenericTempVar(): Variable {
        return Variable("_temp${dfVarId++}")
    }

    fun genTempColumnVar(): Variable {
        return Variable("__bodo_generated_column__${colVarId++}")
    }

    fun genWindowedAggDfName(): Variable {
        return Variable("__bodo_windowfn_generated_df_${dfVarId++}")
    }

    fun genWindowedAggFnName(): Variable {
        return Variable("${dummyColNameBase}_sql_windowed_apply_fn_${groupByApplyFnId++}")
    }

    fun genGroupbyApplyAggFnName(): Variable {
        return Variable("${dummyColNameBase}_sql_groupby_apply_fn_${groupByApplyFnId++}")
    }

    fun genGlobalVar(): Variable {
        return Variable("global_${globalVarId++}")
    }
}
