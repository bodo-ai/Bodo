package com.bodo.iceberg.gson;

import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Base64;

public class ByteBufferAdapter extends TypeAdapter<ByteBuffer> {

  @Override
  public void write(JsonWriter jsonWriter, ByteBuffer byteBuffer) throws IOException {
    throw new RuntimeException("Writing ByteBuffer not implemented yet.");
  }

  public ByteBuffer read(JsonReader reader) throws IOException {
    if (reader.peek() == JsonToken.NULL) {
      reader.nextNull();
      return null;
    }
    String stringValue = reader.nextString();
    return ByteBuffer.wrap(Base64.getDecoder().decode(stringValue));
  }
}
