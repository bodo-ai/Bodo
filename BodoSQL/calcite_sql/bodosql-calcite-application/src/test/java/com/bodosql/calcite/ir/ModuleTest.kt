package com.bodosql.calcite.ir

import com.bodosql.calcite.application.BodoSQLCodegenException
import kotlin.test.Test
import kotlin.test.assertEquals

class ModuleTest {
    @Test
    fun emit() {
        val frame = CodegenFrame()
        frame.add(Op.Assign(Variable("v1"), Expr.Raw("1")))
        frame.add(Op.Assign(Variable("v2"), Expr.Raw("v1 + 3")))
        val module = Module(frame)
        val actual = module.emit()
        val expected =
            """
            v1 = 1
            v2 = v1 + 3
            
            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun append() {
        val module = Module.Builder()
        // Single line.
        module.append("a = 1\n")
        // Chain append calls.
        module
            .append("b = [\n")
            .append("  1,\n")
        // Continuing the previous append from the module is fine.
        module.append("  2,\n")
        module.append("]\n")

        val actual = module.build().emit()
        val expected =
            """
            a = 1
            b = [
              1,
              2,
            ]

            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun activeFrames() {
        /**
         * Test that generated code writes only to the active frame
         * for a builder.
         */
        val builder = Module.Builder()
        val assign1 = Op.Assign(Variable("v1"), Expr.Raw("1"))
        val assign2 = Op.Assign(Variable("x"), Expr.Raw("27"))
        val assign3 = Op.Assign(Variable("y"), Expr.Raw("x - 2"))
        val assign4 = Op.Assign(Variable("v2"), Expr.Raw("v1 + 3"))
        // Add to the default frame
        builder.add(assign1)
        // Create a new frame
        builder.startCodegenFrame()
        builder.add(assign2)
        builder.add(assign3)
        // Pop the frame
        val otherFrame = builder.endFrame()
        // Add to the original frame
        builder.add(assign4)
        // Check the other frame
        val doc = Doc()
        otherFrame.emit(doc)
        val actualOtherFrame = doc.toString()
        val expectedOtherFrame =
            """
            x = 27
            y = x - 2

            """.trimIndent()
        assertEquals(expectedOtherFrame, actualOtherFrame)
        // check what was built
        val actual = builder.build().emit()
        val expected =
            """
            v1 = 1
            v2 = v1 + 3

            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun startStreamingPipeline() {
        /**
         * Test that startStreamingPipeline returns a streaming pipeline frame.
         */
        val builder = Module.Builder()
        builder.startStreamingPipelineFrame(Variable("a"), Variable("b"))
        val frame = builder.endFrame()
        assert(frame is StreamingPipelineFrame)
    }

    @Test
    fun endCurrentStreamingPipeline() {
        /**
         * Test that endCurrentStreamingPipeline works in a streaming
         * context but not otherwise.
         */
        val builder = Module.Builder()
        builder.startStreamingPipelineFrame(Variable("a"), Variable("b"))
        builder.endCurrentStreamingPipeline()
        builder.startCodegenFrame()
        try {
            builder.endCurrentStreamingPipeline()
            // This should trigger an exception
            assert(false)
        } catch (e: BodoSQLCodegenException) {
        }
    }

    @Test
    fun getCurrentStreamingPipeline() {
        /**
         * Test that getCurrentStreamingPipeline works in a streaming
         * context but not otherwise.
         */
        val builder = Module.Builder()
        builder.startStreamingPipelineFrame(Variable("a"), Variable("b"))
        builder.getCurrentStreamingPipeline()
        // We shouldn't alter the state
        builder.getCurrentStreamingPipeline()
        builder.endFrame()
        try {
            builder.getCurrentStreamingPipeline()
            // This should trigger an exception
            assert(false)
        } catch (e: BodoSQLCodegenException) {
        }
    }
}
