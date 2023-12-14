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
 */
package org.apache.calcite.plan;

import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;

import java.util.HashMap;
import java.util.Map;

/**
 * Bodo extension to Strong.java with support for handling OTHER_FUNCTION
 * SqlKind in policy. Calcite has added a `operator.getStrongPolicyInference()`
 * method, which is a lambda for determining the policy at a non-kind per function
 * level instead. As a result, any operators that we define should leverage this
 * interface instead.
 */
public class BodoStrong {
    // Mapping of function names for functions that are defined in Calcite and need to be
    // added to the Strong map.
    private static final Map<String, Strong.Policy> OTHER_FUNCTION_MAP = createOtherFunctionMap();

    /**
     * Returns how to deduce whether a particular {@link RexNode} expression is null,
     * given whether its arguments are null.
     */
    public static Strong.Policy policy(RexNode rexNode) {
        if (rexNode instanceof RexCall) {
            return policy(((RexCall) rexNode).getOperator());
        }
        return Strong.policy(rexNode);
    }

    /**
     * Returns how to deduce whether a particular {@link SqlOperator} expression is null,
     * given whether its arguments are null.
     */
    public static Strong.Policy policy(SqlOperator operator) {
        if (operator.getStrongPolicyInference() != null) {
            return operator.getStrongPolicyInference().get();
        } else if (operator.kind == SqlKind.OTHER_FUNCTION && OTHER_FUNCTION_MAP.containsKey(operator.getName())) {
            return OTHER_FUNCTION_MAP.get(operator.getName());
        } else {
            return Strong.policy(operator);
        }
    }


    private static Map<String, Strong.Policy> createOtherFunctionMap() {
        Map<String, Strong.Policy> map = new HashMap<>();
        map.put(SqlStdOperatorTable.LOWER.getName(), Strong.Policy.ANY);
        map.put(SqlStdOperatorTable.UPPER.getName(), Strong.Policy.ANY);
        return map;
    }
}
