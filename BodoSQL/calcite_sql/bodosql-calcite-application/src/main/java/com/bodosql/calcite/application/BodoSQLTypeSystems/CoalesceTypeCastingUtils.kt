package com.bodosql.calcite.application.BodoSQLTypeSystems // ktlint-disable

import com.bodosql.calcite.application.operatorTables.ArrayOperatorTable
import com.bodosql.calcite.application.operatorTables.CastingOperatorTable
import com.bodosql.calcite.application.operatorTables.DatetimeOperatorTable
import com.bodosql.calcite.application.operatorTables.NumericOperatorTable
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.sql.SqlOperator
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.sql.type.TZAwareSqlType
import org.apache.calcite.sql.type.VariantSqlType
import java.util.*
import kotlin.random.Random

class CoalesceTypeCastingUtils {

    enum class SF_TYPE {
        VARCHAR,
        BINARY,
        TIMESTAMP_NTZ,
        DATE,
        BOOLEAN,
        TIME,
        FLOAT,
        NUMBER,
        TIMESTAMP_LTZ,
        TIMESTAMP_TZ,

        // Semi-structured
        VARIANT,
        OBJECT,
        ARRAY,
        ;
        // Geospatial (unsupported)
//        GEOGRAPHY,
//        GEOMETRY,

        fun is_numeric(): Boolean {
            return this.equals(SF_TYPE.NUMBER) || this.equals(SF_TYPE.FLOAT)
        }
    }

    companion object {

        private val pairTypeMap: Map<Pair<SF_TYPE, SF_TYPE>, Pair<SF_TYPE, SqlOperator>?> = mapOf(
            // identity stuff
            Pair(SF_TYPE.VARCHAR, SF_TYPE.VARCHAR) to null,
            Pair(SF_TYPE.BINARY, SF_TYPE.BINARY) to null,
            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.TIMESTAMP_NTZ) to null,
            Pair(SF_TYPE.DATE, SF_TYPE.DATE) to null,
            Pair(SF_TYPE.BOOLEAN, SF_TYPE.BOOLEAN) to null,
            Pair(SF_TYPE.TIME, SF_TYPE.TIME) to null,
            Pair(SF_TYPE.FLOAT, SF_TYPE.FLOAT) to null,
            Pair(SF_TYPE.NUMBER, SF_TYPE.NUMBER) to null,
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.TIMESTAMP_LTZ) to null,
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.TIMESTAMP_TZ) to null,
            Pair(SF_TYPE.VARIANT, SF_TYPE.VARIANT) to null,
            Pair(SF_TYPE.OBJECT, SF_TYPE.OBJECT) to null,
            Pair(SF_TYPE.ARRAY, SF_TYPE.ARRAY) to null,

            // All combinations of simple types
            Pair(SF_TYPE.VARCHAR, SF_TYPE.BINARY) to null,
            Pair(SF_TYPE.VARCHAR, SF_TYPE.TIMESTAMP_NTZ) to Pair(SF_TYPE.TIMESTAMP_NTZ, CastingOperatorTable.TO_TIMESTAMP_NTZ),
            Pair(SF_TYPE.VARCHAR, SF_TYPE.DATE) to Pair(SF_TYPE.DATE, CastingOperatorTable.TO_DATE),
            // NOTE: Based on the explain from snowflake, this should be using BOOLEAN_TO_TEXT. However, empirically, this is the same as calling TO_CHAR
            Pair(SF_TYPE.VARCHAR, SF_TYPE.BOOLEAN) to Pair(SF_TYPE.VARCHAR, CastingOperatorTable.TO_VARCHAR),
            Pair(SF_TYPE.VARCHAR, SF_TYPE.TIME) to Pair(SF_TYPE.TIME, CastingOperatorTable.TO_TIME),
            Pair(SF_TYPE.VARCHAR, SF_TYPE.FLOAT) to Pair(SF_TYPE.FLOAT, CastingOperatorTable.TO_DOUBLE),
            Pair(SF_TYPE.VARCHAR, SF_TYPE.NUMBER) to Pair(SF_TYPE.NUMBER, CastingOperatorTable.TO_NUMBER),

            Pair(SF_TYPE.BINARY, SF_TYPE.TIMESTAMP_NTZ) to null,
            Pair(SF_TYPE.BINARY, SF_TYPE.DATE) to null,
            Pair(SF_TYPE.BINARY, SF_TYPE.BOOLEAN) to null,
            Pair(SF_TYPE.BINARY, SF_TYPE.TIME) to null,
            Pair(SF_TYPE.BINARY, SF_TYPE.FLOAT) to null,
            Pair(SF_TYPE.BINARY, SF_TYPE.NUMBER) to null,

            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.DATE) to Pair(SF_TYPE.TIMESTAMP_NTZ, CastingOperatorTable.TO_TIMESTAMP_NTZ),
            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.BOOLEAN) to null,
            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.TIME) to null,

            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.FLOAT) to null,
            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.NUMBER) to null,
            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.BOOLEAN) to null,

            Pair(SF_TYPE.DATE, SF_TYPE.BOOLEAN) to null,
            Pair(SF_TYPE.DATE, SF_TYPE.TIME) to null,
            Pair(SF_TYPE.DATE, SF_TYPE.FLOAT) to null,
            Pair(SF_TYPE.DATE, SF_TYPE.NUMBER) to null,

            Pair(SF_TYPE.BOOLEAN, SF_TYPE.TIME) to null,
            Pair(SF_TYPE.BOOLEAN, SF_TYPE.FLOAT) to Pair(SF_TYPE.BOOLEAN, CastingOperatorTable.TO_BOOLEAN),
            Pair(SF_TYPE.BOOLEAN, SF_TYPE.NUMBER) to Pair(SF_TYPE.BOOLEAN, CastingOperatorTable.TO_BOOLEAN),

            Pair(SF_TYPE.TIME, SF_TYPE.FLOAT) to null,
            Pair(SF_TYPE.TIME, SF_TYPE.NUMBER) to null,

            Pair(SF_TYPE.FLOAT, SF_TYPE.NUMBER) to Pair(SF_TYPE.FLOAT, CastingOperatorTable.TO_DOUBLE),

            Pair(SF_TYPE.BINARY, SF_TYPE.VARCHAR) to null,
            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.VARCHAR) to Pair(SF_TYPE.TIMESTAMP_NTZ, CastingOperatorTable.TO_TIMESTAMP_NTZ),
            Pair(SF_TYPE.DATE, SF_TYPE.VARCHAR) to Pair(SF_TYPE.DATE, CastingOperatorTable.TO_DATE),
            Pair(SF_TYPE.BOOLEAN, SF_TYPE.VARCHAR) to Pair(SF_TYPE.BOOLEAN, CastingOperatorTable.TO_BOOLEAN),
            Pair(SF_TYPE.TIME, SF_TYPE.VARCHAR) to Pair(SF_TYPE.TIME, CastingOperatorTable.TO_TIME),
            Pair(SF_TYPE.FLOAT, SF_TYPE.VARCHAR) to Pair(SF_TYPE.FLOAT, CastingOperatorTable.TO_DOUBLE),
            Pair(SF_TYPE.NUMBER, SF_TYPE.VARCHAR) to Pair(SF_TYPE.NUMBER, CastingOperatorTable.TO_NUMBER),

            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.BINARY) to null,
            Pair(SF_TYPE.DATE, SF_TYPE.BINARY) to null,
            Pair(SF_TYPE.BOOLEAN, SF_TYPE.BINARY) to null,
            Pair(SF_TYPE.TIME, SF_TYPE.BINARY) to null,
            Pair(SF_TYPE.FLOAT, SF_TYPE.BINARY) to null,
            Pair(SF_TYPE.NUMBER, SF_TYPE.BINARY) to null,

            Pair(SF_TYPE.DATE, SF_TYPE.TIMESTAMP_NTZ) to Pair(SF_TYPE.TIMESTAMP_NTZ, CastingOperatorTable.TO_TIMESTAMP_NTZ),
            Pair(SF_TYPE.BOOLEAN, SF_TYPE.TIMESTAMP_NTZ) to null,
            Pair(SF_TYPE.TIME, SF_TYPE.TIMESTAMP_NTZ) to null,
            Pair(SF_TYPE.FLOAT, SF_TYPE.TIMESTAMP_NTZ) to null,
            Pair(SF_TYPE.NUMBER, SF_TYPE.TIMESTAMP_NTZ) to null,

            Pair(SF_TYPE.BOOLEAN, SF_TYPE.DATE) to null,
            Pair(SF_TYPE.TIME, SF_TYPE.DATE) to null,
            Pair(SF_TYPE.FLOAT, SF_TYPE.DATE) to null,
            Pair(SF_TYPE.NUMBER, SF_TYPE.DATE) to null,

            Pair(SF_TYPE.TIME, SF_TYPE.BOOLEAN) to null,
            Pair(SF_TYPE.FLOAT, SF_TYPE.BOOLEAN) to Pair(SF_TYPE.BOOLEAN, CastingOperatorTable.TO_BOOLEAN),
            Pair(SF_TYPE.NUMBER, SF_TYPE.BOOLEAN) to Pair(SF_TYPE.BOOLEAN, CastingOperatorTable.TO_BOOLEAN),

            Pair(SF_TYPE.FLOAT, SF_TYPE.TIME) to null,
            Pair(SF_TYPE.NUMBER, SF_TYPE.TIME) to null,

            Pair(SF_TYPE.NUMBER, SF_TYPE.FLOAT) to Pair(SF_TYPE.FLOAT, CastingOperatorTable.TO_DOUBLE),

            // TZ Types
            Pair(SF_TYPE.VARCHAR, SF_TYPE.TIMESTAMP_LTZ) to Pair(SF_TYPE.TIMESTAMP_LTZ, CastingOperatorTable.TO_TIMESTAMP_LTZ),
            Pair(SF_TYPE.VARCHAR, SF_TYPE.TIMESTAMP_TZ) to Pair(SF_TYPE.TIMESTAMP_TZ, CastingOperatorTable.TO_TIMESTAMP_TZ),
            Pair(SF_TYPE.BINARY, SF_TYPE.TIMESTAMP_LTZ) to null,
            Pair(SF_TYPE.BINARY, SF_TYPE.TIMESTAMP_TZ) to null,
            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.TIMESTAMP_LTZ) to Pair(SF_TYPE.TIMESTAMP_LTZ, CastingOperatorTable.TO_TIMESTAMP_LTZ),
            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.TIMESTAMP_TZ) to Pair(SF_TYPE.TIMESTAMP_TZ, CastingOperatorTable.TO_TIMESTAMP_TZ),
            Pair(SF_TYPE.DATE, SF_TYPE.TIMESTAMP_LTZ) to Pair(SF_TYPE.TIMESTAMP_LTZ, CastingOperatorTable.TO_TIMESTAMP_LTZ),
            Pair(SF_TYPE.DATE, SF_TYPE.TIMESTAMP_TZ) to Pair(SF_TYPE.TIMESTAMP_TZ, CastingOperatorTable.TO_TIMESTAMP_TZ),
            Pair(SF_TYPE.BOOLEAN, SF_TYPE.TIMESTAMP_LTZ) to null,
            Pair(SF_TYPE.BOOLEAN, SF_TYPE.TIMESTAMP_TZ) to null,
            Pair(SF_TYPE.TIME, SF_TYPE.TIMESTAMP_LTZ) to null,
            // I triple checked this.
            // Time and timestamp NTZ doesn't coerce to a common type,
            // Time and timestamp LTZ don't coerce to a common type,
            // but time and TZ aware does... and only if the coalesce has exactly two arguments?
            // ... To be blunt, this seems like a bug, so I'm going to dissalow the cast.
            Pair(SF_TYPE.TIME, SF_TYPE.TIMESTAMP_TZ) to null,
            Pair(SF_TYPE.FLOAT, SF_TYPE.TIMESTAMP_LTZ) to null,
            Pair(SF_TYPE.FLOAT, SF_TYPE.TIMESTAMP_TZ) to null,
            Pair(SF_TYPE.NUMBER, SF_TYPE.TIMESTAMP_LTZ) to null,
            Pair(SF_TYPE.NUMBER, SF_TYPE.TIMESTAMP_TZ) to null,
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.VARCHAR) to Pair(SF_TYPE.TIMESTAMP_LTZ, CastingOperatorTable.TO_TIMESTAMP_LTZ),
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.BINARY) to null,
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.TIMESTAMP_NTZ) to Pair(SF_TYPE.TIMESTAMP_LTZ, CastingOperatorTable.TO_TIMESTAMP_LTZ),
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.DATE) to Pair(SF_TYPE.TIMESTAMP_LTZ, CastingOperatorTable.TO_TIMESTAMP_LTZ),
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.BOOLEAN) to null,
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.TIME) to null,
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.FLOAT) to null,
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.NUMBER) to null,
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.TIMESTAMP_TZ) to Pair(SF_TYPE.TIMESTAMP_TZ, CastingOperatorTable.TO_TIMESTAMP_TZ),
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.VARCHAR) to Pair(SF_TYPE.TIMESTAMP_TZ, CastingOperatorTable.TO_TIMESTAMP_TZ),
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.BINARY) to null,
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.TIMESTAMP_NTZ) to Pair(SF_TYPE.TIMESTAMP_TZ, CastingOperatorTable.TO_TIMESTAMP_TZ),
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.DATE) to Pair(SF_TYPE.TIMESTAMP_TZ, CastingOperatorTable.TO_TIMESTAMP_TZ),
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.BOOLEAN) to null,
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.TIME) to Pair(SF_TYPE.TIME, CastingOperatorTable.TO_TIME),
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.FLOAT) to null,
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.NUMBER) to null,
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.TIMESTAMP_LTZ) to Pair(SF_TYPE.TIMESTAMP_TZ, CastingOperatorTable.TO_TIMESTAMP_TZ),

            // semi-structured and variant types
            // note that several of the variant types will throw a runtime error
            // if the cast is invalid

            Pair(SF_TYPE.VARCHAR, SF_TYPE.VARIANT) to Pair(SF_TYPE.VARCHAR, CastingOperatorTable.TO_VARCHAR),
            Pair(SF_TYPE.VARCHAR, SF_TYPE.OBJECT) to null,
            Pair(SF_TYPE.VARCHAR, SF_TYPE.ARRAY) to null,
            Pair(SF_TYPE.BINARY, SF_TYPE.VARIANT) to null,
            Pair(SF_TYPE.BINARY, SF_TYPE.OBJECT) to null,
            Pair(SF_TYPE.BINARY, SF_TYPE.ARRAY) to null,
            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.VARIANT) to Pair(SF_TYPE.TIMESTAMP_NTZ, CastingOperatorTable.TO_TIMESTAMP_NTZ),
            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.OBJECT) to null,
            Pair(SF_TYPE.TIMESTAMP_NTZ, SF_TYPE.ARRAY) to null,
            Pair(SF_TYPE.DATE, SF_TYPE.VARIANT) to Pair(SF_TYPE.DATE, CastingOperatorTable.TO_DATE),
            Pair(SF_TYPE.DATE, SF_TYPE.OBJECT) to null,
            Pair(SF_TYPE.DATE, SF_TYPE.ARRAY) to null,
            Pair(SF_TYPE.BOOLEAN, SF_TYPE.VARIANT) to Pair(SF_TYPE.BOOLEAN, CastingOperatorTable.TO_BOOLEAN),
            Pair(SF_TYPE.BOOLEAN, SF_TYPE.OBJECT) to null,
            Pair(SF_TYPE.BOOLEAN, SF_TYPE.ARRAY) to null,
            Pair(SF_TYPE.TIME, SF_TYPE.VARIANT) to Pair(SF_TYPE.TIME, CastingOperatorTable.TO_TIME),
            Pair(SF_TYPE.TIME, SF_TYPE.OBJECT) to null,
            Pair(SF_TYPE.TIME, SF_TYPE.ARRAY) to null,
            Pair(SF_TYPE.FLOAT, SF_TYPE.VARIANT) to Pair(SF_TYPE.VARIANT, CastingOperatorTable.TO_VARIANT),
            Pair(SF_TYPE.FLOAT, SF_TYPE.OBJECT) to null,
            Pair(SF_TYPE.FLOAT, SF_TYPE.ARRAY) to null,
            Pair(SF_TYPE.NUMBER, SF_TYPE.VARIANT) to Pair(SF_TYPE.VARIANT, CastingOperatorTable.TO_VARIANT),
            Pair(SF_TYPE.NUMBER, SF_TYPE.OBJECT) to null,
            Pair(SF_TYPE.NUMBER, SF_TYPE.ARRAY) to null,
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.VARIANT) to Pair(SF_TYPE.TIMESTAMP_LTZ, CastingOperatorTable.TO_TIMESTAMP_LTZ),
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.OBJECT) to null,
            Pair(SF_TYPE.TIMESTAMP_LTZ, SF_TYPE.ARRAY) to null,
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.VARIANT) to Pair(SF_TYPE.TIMESTAMP_TZ, CastingOperatorTable.TO_TIMESTAMP_TZ),
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.OBJECT) to null,
            Pair(SF_TYPE.TIMESTAMP_TZ, SF_TYPE.ARRAY) to null,
            Pair(SF_TYPE.VARIANT, SF_TYPE.VARCHAR) to Pair(SF_TYPE.VARCHAR, CastingOperatorTable.TO_VARCHAR),
            Pair(SF_TYPE.VARIANT, SF_TYPE.BINARY) to null,
            Pair(SF_TYPE.VARIANT, SF_TYPE.TIMESTAMP_NTZ) to Pair(SF_TYPE.TIMESTAMP_NTZ, CastingOperatorTable.TO_TIMESTAMP_NTZ),
            Pair(SF_TYPE.VARIANT, SF_TYPE.DATE) to Pair(SF_TYPE.DATE, CastingOperatorTable.TO_DATE),
            Pair(SF_TYPE.VARIANT, SF_TYPE.BOOLEAN) to Pair(SF_TYPE.VARIANT, CastingOperatorTable.TO_VARIANT),
            Pair(SF_TYPE.VARIANT, SF_TYPE.TIME) to Pair(SF_TYPE.TIME, CastingOperatorTable.TO_TIME),
            Pair(SF_TYPE.VARIANT, SF_TYPE.FLOAT) to Pair(SF_TYPE.VARIANT, CastingOperatorTable.TO_VARIANT),
            Pair(SF_TYPE.VARIANT, SF_TYPE.NUMBER) to Pair(SF_TYPE.VARIANT, CastingOperatorTable.TO_VARIANT),
            Pair(SF_TYPE.VARIANT, SF_TYPE.TIMESTAMP_LTZ) to Pair(SF_TYPE.TIMESTAMP_LTZ, CastingOperatorTable.TO_TIMESTAMP_LTZ),
            Pair(SF_TYPE.VARIANT, SF_TYPE.TIMESTAMP_TZ) to Pair(SF_TYPE.TIMESTAMP_TZ, CastingOperatorTable.TO_TIMESTAMP_TZ),

            // From the explain, this should be "COERCE obj/array as VARIANT"
            // but TO_VARIANT should be equivalent
            Pair(SF_TYPE.VARIANT, SF_TYPE.OBJECT) to Pair(SF_TYPE.VARIANT, CastingOperatorTable.TO_VARIANT),
            Pair(SF_TYPE.VARIANT, SF_TYPE.ARRAY) to Pair(SF_TYPE.VARIANT, CastingOperatorTable.TO_VARIANT),

            Pair(SF_TYPE.OBJECT, SF_TYPE.VARCHAR) to null,
            Pair(SF_TYPE.OBJECT, SF_TYPE.BINARY) to null,
            Pair(SF_TYPE.OBJECT, SF_TYPE.TIMESTAMP_NTZ) to null,
            Pair(SF_TYPE.OBJECT, SF_TYPE.DATE) to null,
            Pair(SF_TYPE.OBJECT, SF_TYPE.BOOLEAN) to null,
            Pair(SF_TYPE.OBJECT, SF_TYPE.TIME) to null,
            Pair(SF_TYPE.OBJECT, SF_TYPE.FLOAT) to null,
            Pair(SF_TYPE.OBJECT, SF_TYPE.NUMBER) to null,
            Pair(SF_TYPE.OBJECT, SF_TYPE.TIMESTAMP_LTZ) to null,
            Pair(SF_TYPE.OBJECT, SF_TYPE.TIMESTAMP_TZ) to null,
            Pair(SF_TYPE.OBJECT, SF_TYPE.VARIANT) to Pair(SF_TYPE.OBJECT, CastingOperatorTable.TO_OBJECT),
            Pair(SF_TYPE.OBJECT, SF_TYPE.ARRAY) to null,
            Pair(SF_TYPE.ARRAY, SF_TYPE.VARCHAR) to null,
            Pair(SF_TYPE.ARRAY, SF_TYPE.BINARY) to null,
            Pair(SF_TYPE.ARRAY, SF_TYPE.TIMESTAMP_NTZ) to null,
            Pair(SF_TYPE.ARRAY, SF_TYPE.DATE) to null,
            Pair(SF_TYPE.ARRAY, SF_TYPE.BOOLEAN) to null,
            Pair(SF_TYPE.ARRAY, SF_TYPE.TIME) to null,
            Pair(SF_TYPE.ARRAY, SF_TYPE.FLOAT) to null,
            Pair(SF_TYPE.ARRAY, SF_TYPE.NUMBER) to null,
            Pair(SF_TYPE.ARRAY, SF_TYPE.TIMESTAMP_LTZ) to null,
            Pair(SF_TYPE.ARRAY, SF_TYPE.TIMESTAMP_TZ) to null,
            Pair(SF_TYPE.ARRAY, SF_TYPE.VARIANT) to Pair(SF_TYPE.ARRAY, ArrayOperatorTable.TO_ARRAY),
            Pair(SF_TYPE.ARRAY, SF_TYPE.OBJECT) to null,
        )

        // Pre-filtered version of the above. This should be used instead of pairTypeMap for any situation
        // EXCEPT for confirming if a type pair's coercion behavior has been checked.
        // NOTE: The explicit cast is necessary, compiler can't infer that the filter makes the output non-null
        val validPairTypeMap: Map<Pair<SF_TYPE, SF_TYPE>, Pair<SF_TYPE, SqlOperator>> =
            pairTypeMap.filterValues { v -> v != null }.toMap() as Map<Pair<SF_TYPE, SF_TYPE>, Pair<SF_TYPE, SqlOperator>>

        // Converts a relDataType to it's corresponding SF type
        @JvmStatic
        fun TO_SF_TYPE(inputVal: RelDataType): SF_TYPE? {
            val tmp: SF_TYPE? = when {
                // == is needed for null safety.
                // IDK if an SF type can not have a family, but better safe than sorry
                SqlTypeFamily.CHARACTER.contains(inputVal) -> SF_TYPE.VARCHAR
                SqlTypeFamily.DATE.contains(inputVal) -> SF_TYPE.DATE
                SqlTypeFamily.TIMESTAMP.contains(inputVal) -> {
                    if (inputVal is TZAwareSqlType) {
                        SF_TYPE.TIMESTAMP_TZ
                    } else {
                        SF_TYPE.TIMESTAMP_NTZ
                    }
                }
                SqlTypeFamily.BINARY.contains(inputVal) -> {
                    SF_TYPE.BINARY
                }
                SqlTypeFamily.TIME.contains(inputVal) -> {
                    SF_TYPE.TIME
                }
                SqlTypeFamily.BOOLEAN.contains(inputVal) -> {
                    SF_TYPE.BOOLEAN
                }

                // I'm not entirely certain how to handle non-exact numerics. Internally in Bodo,
                // we just use float values, but those float values could be fixed point Numbers or
                // Float/Double's in SF. There's no real way to fix this without supporting
                // Fixed point numbers in Bodo/BodoSQL, which is a long ways off,
                // so for right now, I'm just going to map any numerics with somthing to the right
                // of the decimal point to
                // Float, since that's closest to how we treat them internally.
                SqlTypeFamily.APPROXIMATE_NUMERIC.contains(inputVal) -> {
                    SF_TYPE.FLOAT
                }

                SqlTypeFamily.NUMERIC.contains(inputVal) -> {
                    if (inputVal.sqlTypeName.equals(SqlTypeName.DECIMAL) && inputVal.scale > 0) {
                        SF_TYPE.FLOAT
                    } else {
                        SF_TYPE.NUMBER
                    }
                }

                // Semi-structured types
                // Some of these may need to be adjusted.
                SqlTypeFamily.ARRAY.contains(inputVal) -> {
                    SF_TYPE.ARRAY
                }
                SqlTypeFamily.MAP.contains(inputVal) -> {
                    SF_TYPE.OBJECT
                }
                inputVal.isStruct() -> {
                    SF_TYPE.OBJECT
                }
                inputVal is VariantSqlType -> {
                    return SF_TYPE.VARIANT
                }
                // These types don't have a corresponding SF type. In these cases we return null
                SqlTypeFamily.NULL.contains(inputVal) -> {
                    null
                }
                SqlTypeFamily.INTERVAL_YEAR_MONTH.contains(inputVal) -> {
                    null
                }
                SqlTypeFamily.INTERVAL_DAY_TIME.contains(inputVal) -> {
                    null
                }
                else -> throw Exception("Internal error in TO_SF_TYPE: Cannot find corresponding SF type for $inputVal")
            }

            return tmp
        }

        @JvmStatic
        fun sfGetCoalesceTypeAndCastingFn(typ1: SF_TYPE, typ2: SF_TYPE): Pair<SF_TYPE, SqlOperator?> {
            return validPairTypeMap.get(Pair(typ1, typ2))
                ?: throw Exception("Error in Coalesce: non coercible types passed to coalesce")
        }

        /****************************** DEBUGING CODE ******************************/

        /**
         * Everything below this point is debugging code, used to generate testing queries
         * and/or SF queries to check type coercion behavior.
         *
         * This may eventually be useful when we choose to support
         * GEOMETRY or GEOSPATIAL, so I'm opting to keep it in for now.
         */

        /**
         * For each type paring not cataloged in pairTypeMap,
         * generates a query to check the coalescing behavior.
         * (This query should run on SF)
         */
        fun genQueriesForUnconfirmedTypePairs() {
            for (typ1 in SF_TYPE.values()) {
                for (typ2 in SF_TYPE.values()) {
                    if (typ1.equals(typ2)) continue
                    if (!pairTypeMap.containsKey(Pair(typ1, typ2))) {
                        println("EXPLAIN SELECT Coalesce(${SFTypeToColumNameDict.get(typ1)}, ${SFTypeToColumNameDict.get(typ2)}) FROM TEST_DB.PUBLIC.KEATON_LOCAL_TEST;")
                    }
                }
            }
        }

        fun checkTwoHelperFunction(
            key: Pair<SF_TYPE, SF_TYPE>,
            value: Pair<SF_TYPE, SqlOperator>,
        ): Pair<List<String>, SF_TYPE> {
            val lhs: String = SFTypeToColumNameDict.get(key.first)!!
            val rhs: String = SFTypeToColumNameDict.get(key.second)!!
            val outputType: SF_TYPE = value.first
            return Pair(listOf(lhs, rhs), outputType)
        }

        /**
         * Generates a test that check all valid pairs of types for coalesce.
         * see com.bodosql.calcite.application.CoalesceCastTest.testCoalescePairs for an example of where this
         * is used in the bodo testing suite
         */
        public fun genCheckTwo(): List<Pair<List<String>, SF_TYPE>> {
            return validPairTypeMap.map { cast -> checkTwoHelperFunction(cast.key, cast.value) }
        }

        /**
         * Generates a query to sanity check casting behavior. Essentially generates a set of random coalesce
         * statements, that are guaranteed to check every possible pairing of types.
         *
         * @param checkSfBehavior If checkSfBehavior is true, the query generated will be designed to run on SF.
         *                        It will contain a DESC TABLE so the user can manually check output types.
         *                        If checkSfBehavior is false, the query generated will be designed to run in the
         *                        bodo test suite. See
         *                        com.bodosql.calcite.application.CoalesceCastTest.testCoalesceStressTest for an example
         *                        of where this is used in the Bodo testing suite.
         */
        fun genSanityCheck(seed: Int): List<Pair<List<String>, SF_TYPE>> {
            val reamainingKeys: MutableSet<Pair<SF_TYPE, SF_TYPE>> = validPairTypeMap.keys.toMutableSet()
            val outStmts: MutableList<Pair<List<String>, SF_TYPE>> = ArrayList()

            // Idea is fairly simple, pick a random key, chain conversions until there's no more to add, emit it,
            // and then start over. Repeat until there are no more valid conversions.
            var i = 0
            val rngGenerator = Random(seed)
            while (reamainingKeys.isNotEmpty()) {
                val baseKey: Pair<SF_TYPE, SF_TYPE> = reamainingKeys.random(rngGenerator)
                reamainingKeys.remove(baseKey)

                // List is in reverse order. IE:
                // [typ1, typ2, typ3] --> COALESCE(typ3, typ2, typ1)
                val argsList: MutableList<SF_TYPE> = mutableListOf(baseKey.second, baseKey.first)

                var expectedOutType: SF_TYPE = validPairTypeMap.get(baseKey)!!.first
                var nextKeyOptional = findValidCastIfExists(reamainingKeys, expectedOutType)

                while (nextKeyOptional.isPresent) {
                    val nextKeyInChain = nextKeyOptional.get()
                    expectedOutType = validPairTypeMap.get(nextKeyInChain)!!.first
                    argsList.add(nextKeyInChain.first)
                    reamainingKeys.remove(nextKeyInChain)
                    nextKeyOptional = findValidCastIfExists(reamainingKeys, expectedOutType)
                }

                val element: Pair<List<String>, SF_TYPE> = Pair(
                    argsList.reversed().map {
                        SFTypeToColumNameDict[it]!!
                    },
                    expectedOutType,
                )

                outStmts.add(element)
            }

            return outStmts
        }

        val SFTypeToColumNameDict = mapOf(
            SF_TYPE.VARCHAR to "VARCHAR_COLUMN_1",
            SF_TYPE.BINARY to "BINARY_COLUMN_1",
            SF_TYPE.TIMESTAMP_NTZ to "TIMESTAMP_NTZ_COLUMN_1",
            SF_TYPE.DATE to "DATE_COLUMN_1",
            SF_TYPE.BOOLEAN to "BOOL_COLUMN_1",
            SF_TYPE.TIME to "TIME_COLUMN_1",
            SF_TYPE.FLOAT to "FLOAT_COLUMN_1",
            SF_TYPE.NUMBER to "NUMBER_COLUMN_1",
            SF_TYPE.TIMESTAMP_LTZ to "TIMESTAMP_LTZ_COLUMN_1",
            SF_TYPE.TIMESTAMP_TZ to "TIMESTAMP_TZ_COLUMN_1",
            SF_TYPE.OBJECT to "OBJECT_COLUMN_1",
            SF_TYPE.ARRAY to "ARRAY_COLUMN_1",
            SF_TYPE.VARIANT to "VARIANT_COLUMN_1",
        )

        /**
         * Generates a formatted coalesce call.
         *
         * @param argsList Types of arguments to supply to the coalesce call
         * @param expectedOutType The expected SF output type of the coalesce call
         * @param suffix int value to make sure output column names are unique
         */
        fun genFormattedCoalesceStmt(argsList: List<SF_TYPE>, expectedOutType: SF_TYPE, suffix: Int): String {
            // map of type-> expression

            val valuesListAsString: List<String> = argsList.map { t -> SFTypeToColumNameDict.get(t)!! }
            return genFormattedCoalesceStmt2(valuesListAsString, expectedOutType, suffix)
        }

        fun genFormattedCoalesceStmt2(argsListAsString: List<String>, expectedOutType: SF_TYPE, suffix: Int): String {
            // map of type-> expression
            var suffix = suffix

            val valuesListAsString = argsListAsString.reduce { u, v -> "$u, $v" }
            val outColName = "EXPECT_${expectedOutType}_$suffix"
            val coalesce_stmt = "COALESCE($valuesListAsString) as $outColName"
            return coalesce_stmt
        }

        /**
         * Helper function, finds an unused valid cast, from the rhs_type, if such a cast exists.
         */
        fun findValidCastIfExists(unusedCasts: Set<Pair<SF_TYPE, SF_TYPE>>, rhs_type: SF_TYPE): Optional<Pair<SF_TYPE, SF_TYPE>> {
            return unusedCasts.stream().filter { entry -> entry.second == rhs_type }.findAny()
        }
    }
}
