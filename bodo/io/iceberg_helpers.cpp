#include "./iceberg_helpers.h"

#include <arrow/util/key_value_metadata.h>
#include <fmt/format.h>

int get_iceberg_field_id(const std::shared_ptr<arrow::Field>& field) {
    std::shared_ptr<const arrow::KeyValueMetadata> field_md = field->metadata();
    if (field_md == nullptr || !field_md->Contains(ICEBERG_FIELD_ID_MD_KEY)) {
        throw std::runtime_error(
            fmt::format("Iceberg Field ID not found in the field! Field:\n{}",
                        field->ToString(/*show_metadata*/ true)));
    }
    int iceberg_field_id = static_cast<int>(
        std::stoi(field_md->Get(ICEBERG_FIELD_ID_MD_KEY).ValueOrDie()));
    return iceberg_field_id;
}
