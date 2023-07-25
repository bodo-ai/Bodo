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
import com.bodosql.calcite.rel.core.RelFactories
import com.bodosql.calcite.sql.parser.SqlBodoParserImpl
import com.bodosql.calcite.sql.validate.implicit.BodoTypeCoercionImpl
import com.google.common.collect.ImmutableList
import org.apache.calcite.avatica.util.Casing
import org.apache.calcite.config.NullCollation
import org.apache.calcite.jdbc.CalciteSchema
import org.apache.calcite.prepare.CalciteCatalogReader
import org.apache.calcite.rel.type.RelDataTypeSystem
import org.apache.calcite.schema.SchemaPlus
import org.apache.calcite.sql.parser.SqlParser
import org.apache.calcite.sql.validate.SqlConformanceEnum
import org.apache.calcite.sql.validate.SqlValidator
import org.apache.calcite.sql2rel.SqlToRelConverter
import org.apache.calcite.sql2rel.StandardConvertletTableConfig
import org.apache.calcite.tools.FrameworkConfig
import org.apache.calcite.tools.Frameworks

class PlannerImpl(config: Config) : AbstractPlannerImpl(frameworkConfig(config)) {
    private val defaultSchemas = config.defaultSchemas

    companion object {
        private fun frameworkConfig(config: Config): FrameworkConfig {
            return Frameworks.newConfigBuilder()
                .operatorTable(BodoOperatorTable)
                .typeSystem(config.typeSystem)
                .sqlToRelConverterConfig(
                    SqlToRelConverter.config()
                        .withExpand(false)
                        .withInSubQueryThreshold(Integer.MAX_VALUE)
                )
                .parserConfig(
                    SqlParser.Config.DEFAULT
                        .withCaseSensitive(false)
                        .withQuotedCasing(Casing.UNCHANGED)
                        .withUnquotedCasing(Casing.UNCHANGED)
                        .withConformance(SqlConformanceEnum.LENIENT)
                        .withParserFactory(SqlBodoParserImpl.FACTORY)
                )
                .convertletTable(
                    BodoConvertletTable(
                        StandardConvertletTableConfig(false, false)
                    )
                )
                .sqlValidatorConfig(
                    SqlValidator.Config.DEFAULT
                        .withNamedParamTableName(config.namedParamTableName)
                        .withDefaultNullCollation(NullCollation.LOW)
                        .withCallRewrite(false)
                        .withTypeCoercionFactory(BodoTypeCoercionImpl.FACTORY)
                )
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
            typeFactory, connectionConfig,
        )
    }

    class Config(
        val defaultSchemas: List<SchemaPlus>,
        val typeSystem: RelDataTypeSystem,
        val namedParamTableName: String,
        val plannerType: PlannerType,
    )
}
