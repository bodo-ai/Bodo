package org.apache.calcite.sql.type;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableSet;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Bodo's extension of SqlTypeCoercionRule with
 * our additional types.
 */
public class BodoSqlTypeCoercionRule {

    private static final SqlTypeCoercionRule INSTANCE;
    /**
     * Define the SqlTypeCoercionRule as an extension of the existing
     * SqlTypeCoercionRule but allowing every type be cast to and from
     * Variant.
     */
    static {
        SqlTypeMappingRules.Builder builder = SqlTypeMappingRules.builder();
        final SqlTypeCoercionRule defaultRules = SqlTypeCoercionRule.instance();
        builder.addAll(defaultRules.getTypeMapping());
        // Allow casting from boolean to integer types.
        SqlTypeName[] numericTypes = {SqlTypeName.TINYINT, SqlTypeName.SMALLINT, SqlTypeName.INTEGER, SqlTypeName.BIGINT};
        ImmutableCollection.Builder<SqlTypeName> booleanRule = builder.copyValues(SqlTypeName.BOOLEAN);
        for (SqlTypeName type: numericTypes) {
            builder.add(type, builder.copyValues(type).add(SqlTypeName.BOOLEAN).build());
            // Add this type to the boolean set
            booleanRule.add(type);
        }
        // Add the boolean rule
        builder.add(SqlTypeName.BOOLEAN, booleanRule.build().stream().collect(Collectors.toSet()));


        // NOTE: SqlTypeName.OTHER means variant
        // Allow casting variant to everything
        Set<SqlTypeName> variantRule = new HashSet();
        variantRule.add(SqlTypeName.OTHER);
        // Allow casting everything to variant.
        for (SqlTypeName key: builder.map.keySet()) {
            builder.add(key, builder.copyValues(key).add(SqlTypeName.OTHER).build());
            // Add this type to the variant set
            variantRule.add(key);
        }
        // Add the variant rule
        builder.add(SqlTypeName.OTHER, variantRule);
        INSTANCE = SqlTypeCoercionRule.instance(builder.map);
    }
    public static SqlTypeCoercionRule instance() {
        return INSTANCE;
    }
}

