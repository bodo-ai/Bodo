package com.bodosql.calcite.ir

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class SymbolTableTest {
    @Test
    fun genDfVar() {
        genTestHelper("df") { genDfVar() }
    }

    @Test
    fun genTableVar() {
        genTestHelper("T") { genTableVar() }
    }

    @Test
    fun genSeriesVar() {
        genTestHelper("S") { genSeriesVar() }
    }

    @Test
    fun genGenericTempVar() {
        genTestHelper("_temp") { genGenericTempVar() }
    }

    @Test
    fun genTempColumnVar() {
        genTestHelper("__bodo_generated_column__") { genTempColumnVar() }
    }

    @Test
    fun genWindowedAggDf() {
        genTestHelper("__bodo_windowfn_generated_df_") { genWindowedAggDf() }
    }

    @Test
    fun genWindowedAggFnName() {
        genTestHelper("__bodo_dummy___sql_windowed_apply_fn_") { genWindowedAggFnName() }
    }

    @Test
    fun genGroupbyApplyAggFnName() {
        genTestHelper("__bodo_dummy___sql_groupby_apply_fn_") { genGroupbyApplyAggFnName() }
    }

    @Test
    fun genGlobalVar() {
        genTestHelper("global_") { genGlobalVar() }
    }

    @Test
    fun genIterVar() {
        genTestHelper("_iter_") { genIterVar() }
    }

    @Test
    fun genOutputControlVar() {
        genTestHelper("_produce_output_") { genOutputControlVar() }
    }

    @Test
    fun genInputRequestVar() {
        genTestHelper("_input_request_") { genInputRequestVar() }
    }

    @Test
    fun genNoConflicts() {
        // Invoke the generation functions repeatedly
        // and just ensure we never have a conflict.
        // This should be impossible because every function uses
        // a different prefix, but just being safe.
        val symbolTable = SymbolTable()
        val vars = mutableSetOf<String>()

        // Create a lot of variables of mixed types and just keep doing that
        // with some small variation. We should always create 200 different variables.
        val count = 50 * 19
        for (i in 0 until count) {
            val v =
                when (i % 19) {
                    0, 7, 9 -> symbolTable.genDfVar()
                    1, 8 -> symbolTable.genTableVar()
                    2, 6, 13 -> symbolTable.genSeriesVar()
                    3, 10 -> symbolTable.genGenericTempVar()
                    4, 12 -> symbolTable.genTempColumnVar()
                    5 -> symbolTable.genWindowedAggDf()
                    11 -> symbolTable.genWindowedAggFnName()
                    14 -> symbolTable.genGroupbyApplyAggFnName()
                    15 -> symbolTable.genGlobalVar()
                    16 -> symbolTable.genIterVar()
                    17 -> symbolTable.genOutputControlVar()
                    18 -> symbolTable.genInputRequestVar()
                    else -> throw IllegalStateException("${i % 19}")
                }
            vars.add(v.name)
        }
        assertEquals(count, vars.size)
    }

    private fun genTestHelper(
        prefix: String,
        genVar: SymbolTable.() -> Variable,
    ) {
        val symbolTable = SymbolTable()

        // Store the generated variables.
        val vars = mutableSetOf<String>()

        // First variable should be prefix1.
        val v1 = symbolTable.genVar()
        assertEquals("${prefix}1", v1.name)
        vars.add(v1.name)

        // Second variable should be prefix2.
        val v2 = symbolTable.genVar()
        assertEquals("${prefix}2", v2.name)
        vars.add(v2.name)

        // Generate 98 more variables and check that
        // we have 100 unique variable names.
        for (i in 1..98) {
            vars.add(symbolTable.genVar().name)
        }
        assertEquals(100, vars.size)
    }
}
