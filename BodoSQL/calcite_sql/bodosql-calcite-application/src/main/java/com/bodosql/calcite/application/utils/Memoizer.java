package com.bodosql.calcite.application.utils;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * Implementation of a basic memoization function used for caching functions that always output the
 * same result. This is adopted from this blog post:
 * https://dzone.com/articles/java-8-automatic-memoization.
 */
public class Memoizer<T, U> {
  private final Map<T, U> cache = new ConcurrentHashMap<>();

  private Memoizer() {}

  private Function<T, U> doMemoize(final Function<T, U> function) {
    return input -> cache.computeIfAbsent(input, function::apply);
  }

  public static <T, U> Function<T, U> memoize(final Function<T, U> function) {
    return new Memoizer<T, U>().doMemoize(function);
  }
}
