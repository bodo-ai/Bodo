#include "../io/_s3_reader.h"
#include "./test.hpp"

bodo::tests::suite iceberg_rest_aws_credentials_provider_tests([] {
    bodo::tests::test("test_credentials_refresh", [] {
        return;
        // std::string bearer_token = getenv("TABULAR_BEARER_TOKEN");
        // bodo::tests::check(bearer_token != "",
        //                    "TABULAR_BEARER_TOKEN environment variable not
        //                    set");

        // IcebergRestAwsCredentialsProvider provider(
        //     "https://api.tabular.io/ws", bearer_token,
        //     "Bodo-Test-Iceberg-Warehouse", "examples", "nyc_taxi_locations",
        //     0);
        // auto creds1 = provider.GetAWSCredentials();
        // auto creds2 = provider.GetAWSCredentials();
        // bodo::tests::check(creds1 != creds2);
    });
});
