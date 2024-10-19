package com.bodo.iceberg;

/** Template class for storing a triple of objects. */
public class Triple<X, Y, Z> {
  private final X first;
  private final Y second;
  private final Z third;

  public Triple(X first, Y second, Z third) {
    this.first = first;
    this.second = second;
    this.third = third;
  }

  public X getFirst() {
    return first;
  }

  public Y getSecond() {
    return second;
  }

  public Z getThird() {
    return third;
  }
}
