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
package com.bodosql.calcite.application.bodo_sql_rules;

import java.lang.annotation.ElementType;
import java.lang.annotation.Target;
import org.immutables.value.Value;

/**
 * Annotation to be used to convert interfaces/abstract classes into Immutable POJO using Immutables
 * package. This is copied from the calcite equivalent file
 * (core/src/main/java/org/apache/calcite/CalciteImmutable.java) so that we have the same method
 * names when we are working with our custom rules.
 */
@Target({ElementType.PACKAGE, ElementType.TYPE})
@Value.Style(
    visibility = Value.Style.ImplementationVisibility.PACKAGE,
    defaults = @Value.Immutable(builder = true, singleton = true),
    get = {"is*", "get*"},
    init = "with*")
public @interface BodoSQLStyleImmutable {}
