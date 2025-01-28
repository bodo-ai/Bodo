package com.bodosql.calcite.ir

import kotlin.test.Test
import kotlin.test.assertEquals

class CodegenFrameTest {
    @Test
    fun testEmit() {
        val frame = CodegenFrame()
        frame.add(Op.Assign(Variable("v1"), Expr.Raw("1")))
        frame.add(Op.Assign(Variable("v2"), Expr.Raw("v1 + 3")))
        val doc = Doc()
        // Append the frame result
        frame.emit(doc)
        val actual = doc.toString()
        val expected =
            """
            v1 = 1
            v2 = v1 + 3

            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testAppend() {
        val frame = CodegenFrame()
        // Single line.
        frame.append("a = 1\n")
        // Test multiple lines being explicitly appended
        frame.append("b = [\n")
        frame.append("  1,\n")
        frame.append("  2,\n")
        frame.append("]\n")

        val doc = Doc()
        // Append the frame result
        frame.emit(doc)
        val actual = doc.toString()
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
    fun testPrependAll() {
        val frame = CodegenFrame()
        val op1 = Op.Assign(Variable("first"), Expr.IntegerLiteral(1))
        val op2 = Op.Assign(Variable("second"), Expr.IntegerLiteral(2))
        val op3 = Op.Assign(Variable("third"), Expr.IntegerLiteral(3))
        frame.add(op3)
        frame.prependAll(listOf(op1, op2))
        val doc = Doc()
        // Append the frame result
        frame.emit(doc)
        val actual = doc.toString()
        val expected =
            """
            first = 1
            second = 2
            third = 3

            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testPrependAllEmpty() {
        val frame = CodegenFrame()
        val op1 = Op.Assign(Variable("first"), Expr.IntegerLiteral(1))
        frame.add(op1)
        frame.prependAll(listOf())
        val doc = Doc()
        // Append the frame result
        frame.emit(doc)
        val actual = doc.toString()
        val expected =
            """
            first = 1

            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testAddBeforeReturn() {
        val frame = CodegenFrame()
        val op1 = Op.Assign(Variable("first"), Expr.IntegerLiteral(1))
        val ret = Op.ReturnStatement(Variable("first"))
        val op2 = Op.Assign(Variable("second"), Expr.IntegerLiteral(2))
        frame.add(op1)
        frame.add(ret)
        frame.addBeforeReturn(op2)
        val doc = Doc()
        // Append the frame result
        frame.emit(doc)
        val actual = doc.toString()
        val expected =
            """
            first = 1
            second = 2
            return first
            
            """.trimIndent()
        assertEquals(expected, actual)
    }

    @Test
    fun testAddBeforeReturnNoReturn() {
        val frame = CodegenFrame()
        val op1 = Op.Assign(Variable("first"), Expr.IntegerLiteral(1))
        val op2 = Op.Assign(Variable("second"), Expr.IntegerLiteral(2))
        frame.add(op1)
        frame.addBeforeReturn(op2)
        val doc = Doc()
        // Append the frame result
        frame.emit(doc)
        val actual = doc.toString()
        val expected =
            """
            first = 1
            second = 2
            
            """.trimIndent()
        assertEquals(expected, actual)
    }
}
