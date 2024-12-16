package org.apache.calcite.util;

import org.apache.calcite.runtime.BodoSQLResource;
import org.apache.calcite.runtime.Resources;

/**
 * Bodo equivalent to Static.java inside Calcite for Bodo
 * specific objects.
 *
 * Definitions of objects to be statically imported.
 *
 * <h2>Note to developers</h2>
 *
 * <p>Please give careful consideration before including an object in this
 * class. Pros:
 * <ul>
 * <li>Code that uses these objects will be terser.
 * </ul>
 *
 * <p>Cons:</p>
 * <ul>
 * <li>Namespace pollution,
 * <li>code that is difficult to understand (a general problem with static
 * imports),
 * <li>potential cyclic initialization.
 * </ul>
 */
public abstract class BodoStatic {
    private BodoStatic() {}

    /** Resources. */
    public static final BodoSQLResource BODO_SQL_RESOURCE =
            Resources.create(BodoSQLResource.class);
}
