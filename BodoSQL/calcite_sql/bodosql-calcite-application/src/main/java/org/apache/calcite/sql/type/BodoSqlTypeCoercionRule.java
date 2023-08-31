package org.apache.calcite.sql.type;

import java.util.HashSet;
import java.util.Set;

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

