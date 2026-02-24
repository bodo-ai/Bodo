#include "_bodo_write_function.h"
#include "physical/write_iceberg.h"
#include "physical/write_parquet.h"
#include "physical/write_s3_vectors.h"
#if USE_CUDF
#include "physical/gpu_write_parquet.h"
#endif

std::variant<std::shared_ptr<PhysicalSink>, std::shared_ptr<PhysicalGPUSink>>
ParquetWriteFunctionData::CreatePhysicalOperator(
    std::shared_ptr<bodo::Schema> in_table_schema, bool run_on_gpu) {
#ifdef USE_CUDF
    if (run_on_gpu) {
        return std::make_shared<PhysicalGPUWriteParquet>(in_table_schema,
                                                         *this);
    }
#endif
    return std::make_shared<PhysicalWriteParquet>(in_table_schema, *this);
}

std::variant<std::shared_ptr<PhysicalSink>, std::shared_ptr<PhysicalGPUSink>>
IcebergWriteFunctionData::CreatePhysicalOperator(
    std::shared_ptr<bodo::Schema> in_table_schema, bool run_on_gpu) {
    return std::make_shared<PhysicalWriteIceberg>(in_table_schema, *this);
}

std::variant<std::shared_ptr<PhysicalSink>, std::shared_ptr<PhysicalGPUSink>>
S3VectorsWriteFunctionData::CreatePhysicalOperator(
    std::shared_ptr<bodo::Schema> in_table_schema, bool run_on_gpu) {
    return std::make_shared<PhysicalWriteS3Vectors>(in_table_schema, *this);
}
