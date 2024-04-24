package com.bodo.iceberg.gson;

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import java.lang.reflect.Type;

/**
 * Provide an implementation for interface T, which cannot be removed from the class, and enables
 * serialization and deserialization of the interface by mapping it to an implementation class.
 */
public class InterfaceAdapter<T> implements JsonDeserializer {
  private final Class implementationClass;

  public InterfaceAdapter(Class implementationClass) {
    this.implementationClass = implementationClass;
  }

  public T deserialize(
      JsonElement jsonElement, Type type, JsonDeserializationContext jsonDeserializationContext)
      throws JsonParseException {
    JsonObject jsonObject = jsonElement.getAsJsonObject();
    return jsonDeserializationContext.deserialize(jsonObject, implementationClass);
  }
}
