package com.bodosql.calcite.application;

import java.nio.ByteBuffer;

public interface IService {
  ByteBuffer process(ByteBuffer buffer);
}
