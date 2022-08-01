/*
 * Copyright 2018 Bodo, Inc.
 */

package com.bodosql.calcite.sql.validate;

import org.apache.calcite.sql.validate.SqlAbstractConformance;

public class BodoSqlSqlConformance extends SqlAbstractConformance {
	@Override
	public boolean
	isPercentRemainderAllowed() {
		return true;
	}
}
