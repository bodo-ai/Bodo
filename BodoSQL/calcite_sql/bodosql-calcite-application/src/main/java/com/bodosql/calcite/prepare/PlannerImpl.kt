/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This file is a derivative work of the PlannerImpl in the core calcite
 * project located here: https://github.com/apache/calcite/blob/main/core/src/main/java/org/apache/calcite/prepare/PlannerImpl.java.
 *
 * It has been modified for Bodo purposes. As this is a derivative work,
 * the license has been retained above.
 */
package com.bodosql.calcite.prepare

import com.bodosql.calcite.plan.CostFactory
import com.bodosql.calcite.sql.parser.SqlBodoParserImpl
import com.google.common.collect.ImmutableList
import org.apache.calcite.avatica.util.Casing
import org.apache.calcite.config.NullCollation
import org.apache.calcite.jdbc.CalciteSchema
import org.apache.calcite.prepare.CalciteCatalogReader
import org.apache.calcite.rel.hint.HintPredicates
import org.apache.calcite.rel.hint.HintStrategyTable
import org.apache.calcite.rel.type.RelDataTypeSystem
import org.apache.calcite.schema.SchemaPlus
import org.apache.calcite.sql.parser.SqlParser
import org.apache.calcite.sql.type.BodoSqlTypeCoercionRule
import org.apache.calcite.sql.validate.SqlConformanceEnum
import org.apache.calcite.sql.validate.SqlValidator
import org.apache.calcite.sql.validate.implicit.BodoTypeCoercionImpl
import org.apache.calcite.sql2rel.SqlToRelConverter
import org.apache.calcite.sql2rel.StandardConvertletTableConfig
import org.apache.calcite.tools.FrameworkConfig
import org.apache.calcite.tools.Frameworks

class PlannerImpl(
    config: Config,
) : AbstractPlannerImpl(frameworkConfig(config)) {
    private val defaultSchemas = config.defaultSchemas

    companion object {
        /**
         * Define the parser configuration to use within BodoSQL. The "target"
         * dialect (Spark or Snowflake) has different case sensitivity rules
         * by default, so we modify the parser based on the "sqlStyle" to
         * allow translating code more easily.
         * @param sqlStyle The base dialect for the majority of the SQL code.
         *     This should be one of "SNOWFLAKE" or "SPARK".
         * @return The parser configuration to use for the given SQL style.
         */
        @JvmStatic
        private fun getParserConfig(sqlStyle: String): SqlParser.Config {
            val baseConfig =
                SqlParser.Config.DEFAULT
                    .withConformance(SqlConformanceEnum.LENIENT)
                    .withParserFactory(SqlBodoParserImpl.FACTORY)
            return when (sqlStyle) {
                "SNOWFLAKE" -> {
                    baseConfig
                        .withCaseSensitive(true)
                        .withQuotedCasing(Casing.UNCHANGED)
                        .withUnquotedCasing(Casing.TO_UPPER)
                }
                "SPARK" -> {
                    baseConfig
                        .withCaseSensitive(false)
                        .withQuotedCasing(Casing.UNCHANGED)
                        .withUnquotedCasing(Casing.UNCHANGED)
                }
                else -> {
                    throw Exception("Unrecognized bodo sql style: $sqlStyle")
                }
            }
        }

        /**
         * Get the Convertlet Table that we use for BodoSQL. A convertlet table is
         * responsible for unifying implementations between similar or identical
         * functions by converting 1 implementation to another. This is useful for
         * reducing code rewrite and increasing the effectiveness of the optimizer.
         * @return The convertlet table to use for BodoSQL.
         */
        @JvmStatic
        private fun getConvertletTable(): BodoConvertletTable =
            BodoConvertletTable(
                StandardConvertletTableConfig(false, false),
            )

        /**
         * Define the validator configuration to use within BodoSQL. The "target"
         * dialect (Spark or Snowflake) has different default null collation rules
         * (e.g. the default nulls first/last) so we modify the validator based on
         * the "sqlStyle" to allow translating code more easily. This is especially
         * import for window functions as it can lead to subtle runtime differences
         * that are hard to debug.
         *
         * @param sqlStyle The base dialect for the majority of the SQL code.
         * @return The validator configuration to use for the given SQL style.
         */
        @JvmStatic
        private fun getValidatorConfig(sqlStyle: String): SqlValidator.Config {
            val baseConfig =
                SqlValidator.Config.DEFAULT
                    .withCallRewrite(false)
                    .withTypeCoercionFactory(BodoTypeCoercionImpl.FACTORY)
                    .withTypeCoercionRules(BodoSqlTypeCoercionRule.instance())
            // Ensure order by defaults match. The only differences
            // are in the default behavior for nulls first/last.
            return when (sqlStyle) {
                "SNOWFLAKE" -> baseConfig.withDefaultNullCollation(NullCollation.HIGH)
                "SPARK" -> baseConfig.withDefaultNullCollation(NullCollation.LOW)
                else ->
                    throw Exception("Unrecognized bodo sql style: $sqlStyle")
            }
        }

        /**
         * Get the SqlToRelConverter configuration to use within BodoSQL. This
         * is used to define characteristics like our hint handling and
         * sub query handling.
         * @return The SqlToRelConverter configuration to use for BodoSQL.
         */
        @JvmStatic
        private fun getSqlToRelConverterConfig(): SqlToRelConverter.Config =
            SqlToRelConverter
                .config()
                .withInSubQueryThreshold(Integer.MAX_VALUE)
                .withHintStrategyTable(getHintStrategyTable())

        /**
         * @return The table with the hints that BodoSQL supports.
         */
        private fun getHintStrategyTable(): HintStrategyTable {
            val hintStrategies = HintStrategyTable.builder()
            hintStrategies.hintStrategy("broadcast", HintPredicates.JOIN)
            hintStrategies.hintStrategy("build", HintPredicates.JOIN)
            return hintStrategies.build()
        }

        private fun frameworkConfig(config: Config): FrameworkConfig {
            val parserConfig = getParserConfig(config.sqlStyle)
            val validatorConfig = getValidatorConfig(config.sqlStyle)
            val convertletTable = getConvertletTable()
            val sqlToRelConverterConfig = getSqlToRelConverterConfig()
            return Frameworks
                .newConfigBuilder()
                .operatorTable(BodoOperatorTable)
                .typeSystem(config.typeSystem)
                .sqlToRelConverterConfig(sqlToRelConverterConfig)
                .parserConfig(parserConfig)
                .convertletTable(convertletTable)
                .sqlValidatorConfig(validatorConfig)
                .costFactory(CostFactory())
                .traitDefs(config.plannerType.traitDefs())
                .programs(config.plannerType.programs().toList())
                .build()
        }

        private fun rootSchema(schema: SchemaPlus): SchemaPlus {
            var currSchema = schema
            while (true) {
                val parentSchema = currSchema.parentSchema ?: return currSchema
                currSchema = parentSchema
            }
        }
    }

    override fun createCatalogReader(): CalciteCatalogReader {
        val rootSchema = rootSchema(defaultSchemas[0])
        val defaultSchemaPaths: ImmutableList.Builder<List<String>> = ImmutableList.builder()
        for (schema in defaultSchemas) {
            defaultSchemaPaths.add(CalciteSchema.from(schema).path(null))
        }
        defaultSchemaPaths.add(listOf())
        return BodoCatalogReader(
            CalciteSchema.from(rootSchema),
            defaultSchemaPaths.build(),
            typeFactory,
            connectionConfig,
        )
    }

    /**
     * Variant of createCatalogReader that allows specifying a path for the defaults. This supersedes
     * the defaultSchemas information that already exists.
     * @param defaultPath A list of elements in the default path. Element i should be the
     * parent of element i + 1.
     * @return A new CalciteCatalogReader.
     */
    override fun createCatalogReaderWithDefaultPath(defaultPath: List<String>): CalciteCatalogReader {
        // Load the root Schema. Note: This traverses the schemas and doesn't reuse
        // the results.
        val rootSchema = rootSchema(defaultSchemas[0])
        var currentSchema = rootSchema
        val defaultSchemaPaths: ImmutableList.Builder<List<String>> = ImmutableList.builder()
        // must go form MOST SPECIFIC to LEAST SPECIFIC,
        // this is enforced by UDF resolution during validation,
        // see getFunctionsFrom in CalciteCatalogReader
        // Last element must be an empty list.
        defaultSchemaPaths.add(listOf())
        for (element in defaultPath) {
            // Load each schema. It must already exist because we were able to resolve the view.
            val newSchema = currentSchema.getSubSchema(element) ?: throw RuntimeException("Internal Error: Unable to locate schema")
            defaultSchemaPaths.add(CalciteSchema.from(newSchema).path(null))
            // Update the parent.
            currentSchema = newSchema
        }

        return BodoCatalogReader(
            CalciteSchema.from(rootSchema),
            defaultSchemaPaths.build().reversed(),
            typeFactory,
            connectionConfig,
        )
    }

    class Config(
        val defaultSchemas: List<SchemaPlus>,
        val typeSystem: RelDataTypeSystem,
        val plannerType: PlannerType,
        val sqlStyle: String,
    )
}
