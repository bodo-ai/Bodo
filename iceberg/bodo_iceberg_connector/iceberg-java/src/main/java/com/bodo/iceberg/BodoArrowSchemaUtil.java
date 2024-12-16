/**
 * Shamelessly copied from org.apache.iceberg.arrow.ArrowSchemaUtil. The difference is that we
 * include the Iceberg Field ID in the field metadata of the Arrow schema. The original
 * implementation also had a weird conversion for Maps which we've simplified.
 */
package com.bodo.iceberg;

import java.util.List;
import java.util.Map;
import org.apache.arrow.vector.types.DateUnit;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.TimeUnit;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.FieldType;
import org.apache.arrow.vector.types.pojo.Schema;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableMap;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.types.Types;
import org.apache.iceberg.types.Types.ListType;
import org.apache.iceberg.types.Types.MapType;
import org.apache.iceberg.types.Types.NestedField;
import org.apache.iceberg.types.Types.StructType;

public class BodoArrowSchemaUtil {
  // Bodo change: Remove these:
  // private static final String ORIGINAL_TYPE = "originalType";
  // private static final String MAP_TYPE = "mapType";

  private BodoArrowSchemaUtil() {}

  // Must match 'ICEBERG_FIELD_ID_MD_KEY' in schema_helper.py.
  private static String ICEBERG_FIELD_ID_MD_KEY = "PARQUET:field_id";

  /**
   * Convert Iceberg schema to Arrow Schema.
   *
   * @param schema iceberg schema
   * @return arrow schema
   */
  @SuppressWarnings("null")
  public static Schema convert(final org.apache.iceberg.Schema schema) {
    ImmutableList.Builder<Field> fields = ImmutableList.builder();

    for (NestedField f : schema.columns()) {
      fields.add(convert(f));
    }

    return new Schema(fields.build());
  }

  @SuppressWarnings({"null"})
  public static Field convert(final NestedField field) {
    final ArrowType arrowType;

    final List<Field> children = Lists.newArrayList();
    // Bodo change: Include the Iceberg Field ID in the field metadata.
    Map<String, String> metadata =
        ImmutableMap.of(ICEBERG_FIELD_ID_MD_KEY, Integer.toString(field.fieldId()));

    switch (field.type().typeId()) {
      case BINARY:
        arrowType = ArrowType.Binary.INSTANCE;
        break;
      case FIXED:
        final Types.FixedType fixedType = (Types.FixedType) field.type();
        arrowType = new ArrowType.FixedSizeBinary(fixedType.length());
        break;
      case BOOLEAN:
        arrowType = ArrowType.Bool.INSTANCE;
        break;
      case INTEGER:
        arrowType = new ArrowType.Int(Integer.SIZE, true /* signed */);
        break;
      case LONG:
        arrowType = new ArrowType.Int(Long.SIZE, true /* signed */);
        break;
      case FLOAT:
        arrowType = new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE);
        break;
      case DOUBLE:
        arrowType = new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
        break;
      case DECIMAL:
        final Types.DecimalType decimalType = (Types.DecimalType) field.type();
        // Bodo change: Use newer API which requires setting the bitWidth.
        arrowType = new ArrowType.Decimal(decimalType.precision(), decimalType.scale(), 128);
        break;
      case STRING:
        arrowType = ArrowType.Utf8.INSTANCE;
        break;
      case TIME:
        arrowType = new ArrowType.Time(TimeUnit.MICROSECOND, Long.SIZE);
        break;
      case UUID:
        arrowType = new ArrowType.FixedSizeBinary(16);
        break;
      case TIMESTAMP:
        arrowType =
            new ArrowType.Timestamp(
                TimeUnit.MICROSECOND,
                ((Types.TimestampType) field.type()).shouldAdjustToUTC() ? "UTC" : null);
        break;
      case DATE:
        arrowType = new ArrowType.Date(DateUnit.DAY);
        break;
      case STRUCT:
        final StructType struct = field.type().asStructType();
        arrowType = ArrowType.Struct.INSTANCE;

        for (NestedField nested : struct.fields()) {
          children.add(convert(nested));
        }
        break;
      case LIST:
        final ListType listType = field.type().asListType();
        arrowType = ArrowType.List.INSTANCE;

        for (NestedField nested : listType.fields()) {
          children.add(convert(nested));
        }
        break;
      case MAP:
        // Bodo change: Removed the complex logic that created a map within a map
        // for no apparent reason.
        final MapType mapType = field.type().asMapType();
        arrowType = new ArrowType.Map(false);
        for (NestedField nested : mapType.fields()) {
          children.add(convert(nested));
        }
        break;
      default:
        throw new UnsupportedOperationException("Unsupported field type: " + field);
    }

    return new Field(
        field.name(), new FieldType(field.isOptional(), arrowType, null, metadata), children);
  }
}
