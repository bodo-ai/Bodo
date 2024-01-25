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
package com.bodosql.calcite.application.logicalRules;

import com.bodosql.calcite.application.utils.BodoSQLStyleImmutable;
import com.bodosql.calcite.rel.core.Flatten;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Correlate;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.rules.FilterCorrelateRule;
import org.immutables.value.Value;

/**
 * Modified version of Calcite's FilterCorrelateRule that will avoid firing in cases like this:
 *
 * <blockquote>
 *
 * <pre>
 *        Correlate(...)
 *          RelNode(...)
 *          Flatten(...)
 *            RelNode(...)
 *     </pre>
 *
 * </blockquote>
 */
@BodoSQLStyleImmutable
@Value.Enclosing
public class BodoFilterCorrelateRule extends FilterCorrelateRule {

  protected BodoFilterCorrelateRule(BodoFilterCorrelateRule.BodoConfig config) {
    super(config);
  }

  // BODO CHANGE: using this function as a predicate to ensure that the right spine of the
  // correlate is not a flatten node or a chain of filters leading to a flatten node.
  public static Boolean doesNotLeadToFlatten(Correlate correlate) {
    // Loop down the right spine (skipping over filters) until a flatten
    // node is found, a non-filter node is found, or at most 100 iterations.
    RelNode rel = correlate.getInput(1);
    for (int iter = 0; iter < 100; iter++) {
      if (rel instanceof HepRelVertex) {
        rel = ((HepRelVertex) rel).getCurrentRel();
        continue;
      }
      if (rel instanceof Flatten) return false;
      if (!(rel instanceof Filter)) return true;
      rel = rel.getInput(0);
    }
    return true;
  }

  /** Rule configuration. */
  @Value.Immutable
  public interface BodoConfig extends FilterCorrelateRule.Config {
    Config DEFAULT =
        ImmutableBodoFilterCorrelateRule.BodoConfig.of()
            .withOperandFor(Filter.class, Correlate.class);

    @Override
    default BodoFilterCorrelateRule toRule() {
      return new BodoFilterCorrelateRule(this);
    }

    /** Defines an operand tree for the given classes. */
    default Config withOperandFor(
        Class<? extends Filter> filterClass, Class<? extends Correlate> correlateClass) {
      return withOperandSupplier(
              b0 ->
                  b0.operand(filterClass)
                      .predicate(x -> !x.containsOver())
                      .oneInput(
                          b1 ->
                              b1.operand(correlateClass)
                                  .predicate(BodoFilterCorrelateRule::doesNotLeadToFlatten)
                                  .anyInputs()))
          .as(Config.class);
    }
  }
}
