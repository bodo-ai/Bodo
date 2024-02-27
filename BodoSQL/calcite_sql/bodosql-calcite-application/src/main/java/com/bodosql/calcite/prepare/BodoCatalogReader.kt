package com.bodosql.calcite.prepare

import org.apache.calcite.config.CalciteConnectionConfig
import org.apache.calcite.jdbc.CalciteSchema
import org.apache.calcite.prepare.CalciteCatalogReader
import org.apache.calcite.rel.type.RelDataTypeFactory
import org.apache.calcite.sql.validate.SqlNameMatchers

class BodoCatalogReader(
    rootSchema: CalciteSchema,
    defaultSchemaPaths: List<List<String>>,
    typeFactory: RelDataTypeFactory,
    config: CalciteConnectionConfig?,
) :
    CalciteCatalogReader(
            rootSchema,
            SqlNameMatchers.withCaseSensitive(config != null && config.caseSensitive()),
            defaultSchemaPaths,
            typeFactory,
            config,
        )
