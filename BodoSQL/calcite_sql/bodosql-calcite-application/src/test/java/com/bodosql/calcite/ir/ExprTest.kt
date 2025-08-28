package com.bodosql.calcite.ir

import java.math.BigDecimal
import kotlin.test.Test
import kotlin.test.assertEquals

class ExprTest {
    @Test
    fun emitRaw() {
        val expr = Expr.Raw("1 + 2")
        assertEquals("1 + 2", expr.emit())
    }

    @Test
    fun emitCall() {
        // Call with a single argument.
        val call1 = Expr.Call("foo", listOf(Variable("a")))
        assertEquals("foo(a)", call1.emit())

        // No arguments works fine.
        val call2 = Expr.Call("noargs")
        assertEquals("noargs()", call2.emit())

        // Multiple parameters produce commas.
        val call3 =
            Expr.Call(
                "manyparams",
                listOf(
                    Variable("a"),
                    Variable("b"),
                ),
            )
        assertEquals("manyparams(a, b)", call3.emit())

        // Nested calls work fine.
        val call4 =
            Expr.Call(
                "nested",
                listOf(
                    Variable("a"),
                    Expr.Call(
                        "call",
                        listOf(Variable("b")),
                    ),
                ),
            )
        assertEquals("nested(a, call(b))", call4.emit())

        // Alternate constructor using varargs.
        val call5 = Expr.Call("foo", Variable("a"))
        assertEquals("foo(a)", call5.emit())

        // Call with an expression works fine.
        val call6 =
            Expr.Call(
                Expr.Attribute(Variable("a"), "foo"),
                Expr.StringLiteral("bar"),
            )
        assertEquals("a.foo(\"bar\")", call6.emit())
    }

    @Test
    fun emitAttribute() {
        // Attribute access uses dot notation.
        val attribute1 =
            Expr.Attribute(
                Variable("a"),
                "foo",
            )
        assertEquals("a.foo", attribute1.emit())

        val attribute2 =
            Expr.Attribute(
                Variable("bodo.types"),
                "boolean_array_type",
            )
        assertEquals("bodo.types.boolean_array_type", attribute2.emit())
    }

    @Test
    fun emitBodoSQLKernel() {
        val kernelName = "upper"
        val args = listOf(Expr.StringLiteral("test_string"))
        val kernel1 = bodoSQLKernel(kernelName, args)
        assertEquals("bodosql.kernels.upper(\"test_string\")", kernel1.emit())
    }

    @Test
    fun emitInitRangeIndex() {
        val args = listOf(Expr.IntegerLiteral(0), Expr.IntegerLiteral(1), Expr.IntegerLiteral(2))
        val initRangeIndexExpr = initRangeIndex(args)
        assertEquals("bodo.hiframes.pd_index_ext.init_range_index(0, 1, 2)", initRangeIndexExpr.emit())
    }

    @Test
    fun emitMethod() {
        // Call with a single argument.
        val inputVar = Variable("df")
        val call1 = Expr.Method(inputVar, "foo", listOf(Variable("a")))
        assertEquals("df.foo(a)", call1.emit())

        // No arguments works fine.
        val call2 = Expr.Method(inputVar, "foo")
        assertEquals("df.foo()", call2.emit())

        // Only keyword args
        val call3 = Expr.Method(inputVar, "foo", listOf(), listOf(Pair("argname", Expr.Call("bar"))))
        assertEquals("df.foo(argname=bar())", call3.emit())

        // Expr input, args, and kwargs
        val call4 =
            Expr.Method(
                Expr.Call("genDf", Variable("k")),
                "foo",
                listOf(Variable("q"), Expr.BooleanLiteral(false)),
                listOf(Pair("argname", Variable("L"))),
            )
        assertEquals("genDf(k).foo(q, False, argname=L)", call4.emit())
    }

    @Test
    fun emitGroupby() {
        val inputVar = Variable("df")
        val keys1 = Expr.List(listOf(Expr.StringLiteral("A")))
        val keys2 = Expr.List(listOf(Expr.StringLiteral("A"), Expr.StringLiteral("B")))
        val groupby1 = Expr.Groupby(inputVar, keys1, false, false)
        assertEquals("df.groupby([\"A\"], as_index=False, dropna=False, _is_bodosql=True)", groupby1.emit())
        val groupby2 = Expr.Groupby(inputVar, keys2, true, true)
        assertEquals("df.groupby([\"A\", \"B\"], as_index=True, dropna=True, _is_bodosql=True)", groupby2.emit())
    }

    @Test
    fun emitSortValues() {
        val inputVar = Variable("df")
        val ascending = Expr.List(listOf(Expr.BooleanLiteral(true)))
        val naPosition = Expr.List(listOf(Expr.StringLiteral("last")))

        val byColumns = Expr.List(listOf(Expr.StringLiteral("A"), Expr.StringLiteral("B")))
        val sortValues1 = Expr.SortValues(inputVar, byColumns, ascending, naPosition)
        assertEquals(
            "df.sort_values(by=[\"A\", \"B\"], ascending=[True], na_position=[\"last\"])",
            sortValues1.emit(),
        )

        val noAscending = Expr.List(listOf())
        val noNaPosition = Expr.List(listOf())
        val noByColumns = Expr.List(listOf())

        val sortValues2 = Expr.SortValues(inputVar, noByColumns, noAscending, noNaPosition)

        assertEquals("df", sortValues2.emit())
    }

    @Test
    fun emitIndex() {
        // Single index value.
        val index1 =
            Expr.Index(
                Variable("a"),
                Expr.IntegerLiteral(1),
            )
        assertEquals("a[1]", index1.emit())

        // Multiple indices can be passed to the index operator in python.
        val index2 =
            Expr.Index(
                Variable("a"),
                Expr.IntegerLiteral(1),
                Expr.IntegerLiteral(2),
            )
        assertEquals("a[1, 2]", index2.emit())

        // Nested expressions like lists are fine.
        val index3 =
            Expr.Index(
                Variable("a"),
                Expr.List(
                    Expr.IntegerLiteral(1),
                    Expr.IntegerLiteral(2),
                ),
            )
        assertEquals("a[[1, 2]]", index3.emit())
    }

    @Test
    fun emitGetItem() {
        val df = Variable("df")
        // Test integer index
        assertEquals("df[1]", Expr.GetItem(df, Expr.IntegerLiteral(1)).emit())
        // Test string index
        assertEquals("df[\"col_name\"]", Expr.GetItem(df, Expr.StringLiteral("col_name")).emit())
    }

    @Test
    fun emitUnary() {
        // Test -
        val unary1 = Expr.Unary("-", Expr.IntegerLiteral(100))
        assertEquals("-(100)", unary1.emit())

        // Test not
        val unary2 = Expr.Unary("not", Variable("var"))
        assertEquals("not(var)", unary2.emit())
    }

    @Test
    fun emitBinary() {
        // Test ==
        val binary1 = Expr.Binary("==", Expr.IntegerLiteral(100), Variable("var"))
        assertEquals("(100 == var)", binary1.emit())

        // Test and
        val binary2 =
            Expr.Binary(
                "and",
                Variable("var"),
                Expr.Call("bodo.libs.array_kernels.isna", listOf(Variable("arr"), Expr.IntegerLiteral(1))),
            )
        assertEquals("(var and bodo.libs.array_kernels.isna(arr, 1))", binary2.emit())
    }

    @Test
    fun emitRange() {
        // Test start and end
        val range1 = Expr.Range(Variable("start"), Expr.IntegerLiteral(100))
        assertEquals("range(start, 100)", range1.emit())

        // Test Step
        val range2 = Expr.Range(Expr.IntegerLiteral(0), Variable("end"), Expr.IntegerLiteral(2))
        assertEquals("range(0, end, 2)", range2.emit())
    }

    @Test
    fun emitLen() {
        val var1 = Variable("df")
        val lenExpr = Expr.Len(var1)
        assertEquals("len(df)", lenExpr.emit())
    }

    @Test
    fun emitTuple() {
        // Test for empty Python tuples
        val tuple1 = Expr.Tuple(listOf())
        assertEquals("()", tuple1.emit())

        // Test for 1 element Python tuples
        val tuple2 = Expr.Tuple(listOf(Expr.Raw("1")))
        assertEquals("(1,)", tuple2.emit())

        // Test for 3 element Python tuples
        val tuple3 = Expr.Tuple(listOf(Expr.Raw("1"), Expr.Raw("2"), Expr.Raw("3")))
        assertEquals("(1, 2, 3)", tuple3.emit())
    }

    @Test
    fun emitTripleQuotedString() {
        val test = { arg: String, expected: String ->
            val literal = Expr.TripleQuotedString(arg)
            assertEquals(expected, literal.emit())
        }

        // Standard text that has no escapes but would have an escape if it was a standard string.
        test(
            "\"There may be a reason this string \\\" contains anything 1.",
            "\"\"\"" + """"There may be a reason this string \" contains anything 1.""" + "\"\"\"",
        )
        // String contains triple quotes itself and would cause a problem if emitted.
        test(
            "This is a string \"\"\" with triple quotes in the middle.",
            "\"\"\"" + "This is a string \\\"\\\"\\\" with triple quotes in the middle." + "\"\"\"",
        )
    }

    @Test
    fun emitStringLiteral() {
        val test = { arg: String, expected: String ->
            val literal = Expr.StringLiteral(arg)
            assertEquals(expected, literal.emit())
        }
        // Surrounded with quotes for a normal variable.
        test("""COLNAME""", """"COLNAME"""")
        // Literal escape should be escaped itself.
        test("""\""", """"\\"""")
        // Double quotes should also be escaped and surrounded by quotes.
        test(""""""", """"\""""")
        // Special characters like newlines should be output with their correct escape sequence.
        test("\n\r\t\b\u000c", """"\n\r\t\b\f"""")
        // TODO(jsternberg): Test octal and hex escape sequences and ensure those are supported.
    }

    @Test
    fun emitBooleanLiteral() {
        assertEquals("True", Expr.BooleanLiteral(true).emit())
        assertEquals("False", Expr.BooleanLiteral(false).emit())
    }

    @Test
    fun emitIntegerLiteral() {
        assertEquals("0", Expr.IntegerLiteral(0).emit())
        assertEquals("1", Expr.IntegerLiteral(1).emit())
        assertEquals("-1", Expr.IntegerLiteral(-1).emit())
    }

    @Test
    fun emitDoubleLiteral() {
        assertEquals("0.0", Expr.DoubleLiteral(0.0).emit())
        assertEquals("1.23", Expr.DoubleLiteral(1.23).emit())
        assertEquals("-1.00001", Expr.DoubleLiteral(-1.00001).emit())
    }

    @Test
    fun emitNone() {
        assertEquals("None", Expr.None.emit())
    }

    @Test
    fun emitSlice() {
        // Slice with no parameters should just be a colon.
        val slice1 = Expr.Slice()
        assertEquals(":", slice1.emit())

        // Slice with only a start or end parameter.
        val slice2 = Expr.Slice(start = Expr.IntegerLiteral(1))
        assertEquals("1:", slice2.emit())

        val slice3 = Expr.Slice(stop = Expr.IntegerLiteral(2))
        assertEquals(":2", slice3.emit())

        // Now both.
        val slice4 =
            Expr.Slice(
                Expr.IntegerLiteral(1),
                Expr.IntegerLiteral(2),
            )
        assertEquals("1:2", slice4.emit())

        // Include the step value.
        val slice5 =
            Expr.Slice(
                Expr.IntegerLiteral(1),
                Expr.IntegerLiteral(2),
                Expr.IntegerLiteral(3),
            )
        assertEquals("1:2:3", slice5.emit())

        // Step with only start value.
        val slice6 =
            Expr.Slice(
                start = Expr.IntegerLiteral(1),
                step = Expr.IntegerLiteral(3),
            )
        assertEquals("1::3", slice6.emit())

        // Only the stop value.
        val slice7 =
            Expr.Slice(
                stop = Expr.IntegerLiteral(2),
                step = Expr.IntegerLiteral(3),
            )
        assertEquals(":2:3", slice7.emit())

        // Neither start nor stop.
        val slice8 =
            Expr.Slice(
                step = Expr.IntegerLiteral(3),
            )
        assertEquals("::3", slice8.emit())
    }

    @Test
    fun emitList() {
        // Test for empty Python lists
        val list1 = Expr.List(listOf())
        assertEquals("[]", list1.emit())

        // Test for 1 element Python lists
        val list2 = Expr.List(listOf(Expr.Raw("1")))
        assertEquals("[1]", list2.emit())

        // Test for many element Python lists
        val list3 = Expr.List(listOf(Expr.Raw("1"), Expr.Raw("2"), Expr.Raw("3")))
        assertEquals("[1, 2, 3]", list3.emit())
    }

    @Test
    fun emitDict() {
        // Test emitting a dictionary

        // Test an empty dictionary
        val dict1 = Expr.Dict()
        assertEquals("{}", dict1.emit())

        // Test 1 element dictionaries
        val dict2 =
            Expr.Dict(
                Expr.StringLiteral("A") to Variable("a"),
            )
        assertEquals("{\"A\": a}", dict2.emit())

        // Test multi-element dictionaries
        val dict3 =
            Expr.Dict(
                Expr.StringLiteral("A") to Variable("a"),
                Expr.StringLiteral("B") to Expr.Raw("x + y"),
            )
        assertEquals("{\"A\": a, \"B\": x + y}", dict3.emit())
    }

    @Test
    fun emitDecimalLiteral() {
        // Test emitting a Decimal Literal

        // Test with fractional component. Note this must be a value
        // BigDecimal can represent exactly
        val decimal1 = Expr.DecimalLiteral(BigDecimal(1.5))
        assertEquals("1.5", decimal1.emit())

        // Test no fraction
        val decimal2 = Expr.DecimalLiteral(BigDecimal(-14))
        assertEquals("-14", decimal2.emit())

        // Test 0
        val decimal3 = Expr.DecimalLiteral(BigDecimal(0))
        assertEquals("0", decimal3.emit())
    }

    @Test
    fun emitFrameTripleQuotedString() {
        // Test emitting a triple quoted string whose body is a Frame

        // Generate the frame
        val f = CodegenFrame()
        f.add(Op.Assign(Variable("x"), Expr.IntegerLiteral(7)))
        f.add(Op.Assign(Variable("y"), Expr.Call("f", listOf(Variable("x")))))

        // Test with nesting level 0
        val str1 = Expr.FrameTripleQuotedString(f, 0)
        val expected1 = "\"\"\"x = 7\ny = f(x)\n\"\"\""
        assertEquals(expected1, str1.emit())

        // Test with nesting level 1
        val str2 = Expr.FrameTripleQuotedString(f, 1)
        val expected2 = "\"\"\"  x = 7\n  y = f(x)\n\"\"\""
        assertEquals(expected2, str2.emit())

        // Test with nesting level 2
        val str3 = Expr.FrameTripleQuotedString(f, 2)
        val expected3 = "\"\"\"    x = 7\n    y = f(x)\n\"\"\""
        assertEquals(expected3, str3.emit())
    }

    @Test
    fun emitFrameTripleQuotedStringNested() {
        // Test emitting a triple quoted string whose body is a Frame
        // but contains nested frames via If/Else.
        // Generate the frame
        val mainFrame = CodegenFrame()
        val ifFrame = CodegenFrame()
        ifFrame.add(Op.Assign(Variable("x"), Expr.IntegerLiteral(7)))
        val elseFrame = CodegenFrame()
        elseFrame.add(Op.Assign(Variable("x"), Expr.Call("f", listOf(Expr.StringLiteral("abc")))))

        val ifOp = Op.If(Variable("cond"), ifFrame, elseFrame)
        mainFrame.add(ifOp)

        // Test with nesting level 0
        val str1 = Expr.FrameTripleQuotedString(mainFrame, 0)
        val expected1 = "\"\"\"if cond:\n  x = 7\nelse:\n  x = f(\"abc\")\n\"\"\""
        assertEquals(expected1, str1.emit())

        // Test with nesting level 1
        val str2 = Expr.FrameTripleQuotedString(mainFrame, 1)
        val expected2 = "\"\"\"  if cond:\n    x = 7\n  else:\n    x = f(\"abc\")\n\"\"\""
        assertEquals(expected2, str2.emit())

        // Test with nesting level 2
        val str3 = Expr.FrameTripleQuotedString(mainFrame, 2)
        val expected3 = "\"\"\"    if cond:\n      x = 7\n    else:\n      x = f(\"abc\")\n\"\"\""
        assertEquals(expected3, str3.emit())
    }

    @Test
    fun emitDataFrameSlice() {
        val dataframeExpr = Expr.Raw("df3")
        val lowerBound = Expr.IntegerLiteral(1)
        val upperBound = Expr.IntegerLiteral(10)
        // Test with simple integers
        val out1 = Expr.DataFrameSlice(dataframeExpr, lowerBound, upperBound)
        val expected1 = "df3.iloc[1:10]"
        assertEquals(expected1, out1.emit())

        // Test with more complex expressions
        val dataframeExpr2 = Expr.Raw("df3")
        val lowerBound2 = Expr.Raw("10*3 + 1")
        val upperBound2 = Expr.Raw("random.randint(1, 3)")
        val out2 = Expr.DataFrameSlice(dataframeExpr2, lowerBound2, upperBound2)
        val expected2 = "df3.iloc[10*3 + 1:random.randint(1, 3)]"
        assertEquals(expected2, out2.emit())
    }
}
