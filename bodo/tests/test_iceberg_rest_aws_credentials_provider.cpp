#include "../io/_s3_reader.h"
#include "./test.hpp"

bodo::tests::suite iceberg_rest_aws_credentials_provider_tests([] {
    bodo::tests::test(
        "test_credentials_refresh",
        [] {
            char *credential = getenv("TABULAR_CREDENTIAL");
            bodo::tests::check(
                credential != nullptr,
                "TABULAR_CREDENTIAL environment variable not set");

            std::string bearer_token =
                IcebergRestAwsCredentialsProvider::getToken(
                    "https://api.tabular.io/ws", credential);

            IcebergRestAwsCredentialsProvider provider(
                "https://api.tabular.io/ws", bearer_token,
                "Bodo-Test-Iceberg-Warehouse", "examples", "nyc_taxi_locations",
                0);
            auto creds1 = provider.GetAWSCredentials();
            auto creds2 = provider.GetAWSCredentials();
            bodo::tests::check(creds1 != creds2);
        },
        {"tabular", "iceberg"});
});
