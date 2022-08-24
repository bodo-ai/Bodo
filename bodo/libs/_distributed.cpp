// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_distributed.h"
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <ctime>
#include <fstream>
#include <vector>

#if defined(CHECK_LICENSE_EXPIRED) || defined(CHECK_LICENSE_CORE_COUNT) || \
    defined(CHECK_LICENSE_PLATFORM)

#define FREE_MAX_CORES 8
#define EXPIRATION_COUNTDOWN_DAYS 6

/*
 * A license consists of a header, content and signature.
 * The header is one 32-bit integer. The first 16 bits currently specify
 * the license type (note that we don't need 16 bits for the type and can use
 * some of these in the future for other purposes). The second 16 bits specify
 * the total length (in bytes) of the license.
 * The license content varies by license type.
 * - Type 0 (regular):
 *   The content is four 32-bit ints: max core count, expiration year,
 *   expiration month, expiration day
 * - Type 1 (Bodo Platform on AWS):
 *   The content is the EC2 Instance ID (19 ASCII characters) licensed to run
 * Bodo
 * - Type 2 (Bodo Platform on Azure):
 *   The content is the Azure VM Unique ID (36 ASCII characters) licensed to run
 * Bodo The signature is obtained by getting a SHA-256 digest of the license
 * content and signing with a private key using RSA. The license that is
 * provided to customers is encoded in Base64
 */

#define REGULAR_LIC_TYPE 0
#define PLATFORM_LIC_TYPE_AWS 1
#define PLATFORM_LIC_TYPE_AZURE 2

#define PLATFORM_LIC_CHECK_MAX_RETRIES 10
#define PLATFORM_LIC_CHECK_RETRY_MAX_DELAY 5.0

#define HEADER_LEN_MASK \
    0xffff  // mask to get the total license length from license header

// public key to verify signature of license file
const char *pubkey =
    "-----BEGIN PUBLIC KEY-----\n"
    "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAom4enwn8WEk3saSpAhOd\n"
    "n3JCkEf00p+EJuVV9WYK1DbAAog0Szm8TSAtNLNp2cQavctkEmq2qDhPHZL8MhFm\n"
    "x83FVDo/4Jv4U99Jk1T+Xuy25REsbzQ4dPm0tl0yFRcTOOkMrTH2lZ/NOZ6EXIUQ\n"
    "btvjAtgyq3mzFuKZognaterprzlEvNWsMh0ZaO6fF4bT65L9Zv6BCSuwJC0jgssJ\n"
    "wnKmdK2q/zCBnBTlsU3VMYp0WcyWY4hQKzGfKi1zW04kGLKBLIBF/xAR4U/mIj23\n"
    "zk2rTVKWc5IJaYCNG9XiFt71CLosb+sSA6PVVPLz4AKjoUkslhYSzjSkn6uP9ebj\n"
    "lQIDAQAB\n"
    "-----END PUBLIC KEY-----";

/**
 * Read license from environment variable BODO_LICENSE or from license file.
 * @param[out] data: all the license data including signature
 * @param[in] b64_encoded: true if license is Base64 encoded
 */
static int read_license_common(std::vector<char> &data, bool b64_encoded) {
    // first check if license is in environment variable
    char *license_text = std::getenv("BODO_LICENSE");
    if (license_text != NULL && strlen(license_text) > 0) {
        data.assign(license_text, license_text + strlen(license_text));
    } else {
        // read from "bodo.lic" file in: 1) cwd or 2) /etc
        // TODO: check HOME directory or some other directory?
        std::ifstream f("bodo.lic", std::ios::binary);
        if (f.fail()) {
            f = std::ifstream("/etc/bodo.lic", std::ios::binary);
            if (f.fail()) {
                fprintf(stderr, "Bodo license not found\n");
                return 0;
            }
        }
        data.assign(std::istreambuf_iterator<char>{f}, {});
    }

    if (b64_encoded) {
        // decode from Base64 to binary
        // decoded data is smaller than encoded
        std::vector<char> decoded_data(data.size());
        int dlen = EVP_DecodeBlock((unsigned char *)decoded_data.data(),
                                   (unsigned char *)data.data(), data.size());
        if (dlen == -1) {
            // ERR_print_errors_fp(stderr);  // print ssl errors
            fprintf(stderr, "Invalid license\n");
            return 0;
        }
        int header = ((int *)decoded_data.data())[0];
        int actual_size = header & HEADER_LEN_MASK;
        decoded_data.resize(actual_size);
        data = std::move(decoded_data);
    }
    return 1;
}

/**
 * Return public key in `const char *pubkey` global as openssl data structure.
 */
static EVP_PKEY *get_public_key() {
    BIO *mem = BIO_new_mem_buf((void *)pubkey, strlen(pubkey));
    EVP_PKEY *key = PEM_read_bio_PUBKEY(mem, NULL, NULL, NULL);
    BIO_free(mem);
    return key;
}

/**
 * Verifies that the license file is valid (originates from Bodo).
 * @param msg: byte array with license content
 * @param mlen: length of msg in bytes
 * @param sig: signature (obtained when signing license content with Bodo
 * private key)
 * @param slen: length of signature in bytes
 * @return : 1 if verified correctly, 0 otherwise.
 */
static int verify_license(EVP_PKEY *key, const void *msg, const size_t mlen,
                          const unsigned char *sig, size_t slen) {
    EVP_MD_CTX *mdctx = NULL;

    // create the Message Digest Context
    if (!(mdctx = EVP_MD_CTX_create())) return 0;

    // initialize the DigestSign operation - use SHA-256 as the message
    // digest function
    if (1 != EVP_DigestVerifyInit(mdctx, NULL, EVP_sha256(), NULL, key))
        return 0;

    // call update with the message
    if (1 != EVP_DigestVerifyUpdate(mdctx, msg, mlen)) return 0;

    // verify the signature
    if (1 != EVP_DigestVerifyFinal(mdctx, sig, slen)) return 0;

    // clean up
    if (mdctx) EVP_MD_CTX_destroy(mdctx);

    return 1;  // verified correctly
}

#endif  // CHECK_LICENSE_EXPIRED || CHECK_LICENSE_CORE_COUNT ||
        // CHECK_LICENSE_PLATFORM

#if defined(CHECK_LICENSE_PLATFORM)

#include <curl/curl.h>
#include "gason.h"  // lightweight JSON parser: https://github.com/vivkin/gason

#define EC2_INSTANCE_ID_LEN 19  // number of characters in a EC2 instance ID
#define AZURE_INSTANCE_ID_LEN \
    36  // number of characters in an Azure instance ID

// license checking error codes
#define LICENSE_ERR_DOWNLOAD \
    100  // could not download Instance Identity Document
#define LICENSE_ERR_JSON_PARSE 101  // Error parsing Identity Document
#define LICENSE_ERR_INSTANCE_ID_NOT_FOUND \
    102  // Instance ID not found in Identity Document
#define LICENSE_ERR_INVALID_INSTANCE 103  // License is not for this instance
#define LICENSE_ERR_INVALID_LICENSE \
    104  // Invalid license (for example tampered license, invalid/no signature,
         // etc.)
#define LICENSE_ERR_INVALID_LICENSE_TYPE 105  // Invalid license type

/**
 * Reads the license file, verifies the signature and returns the instance ID
 * contained in the license.
 * @param[out] license_type: the license type (see PLATFORM_LIC_TYPE_X above)
 * @param[out] lic_instance_id: the Instance ID contained in the license
 */
static int get_license_platform(int &license_type,
                                std::string &lic_instance_id) {
    std::vector<char> data;   // store license
    bool b64_encoded = true;  // license encoded in Base64
    int read_license = read_license_common(data, b64_encoded);
    if (!read_license) return 0;

    license_type = ((int *)data.data())[0] >> 16;
    if (license_type != PLATFORM_LIC_TYPE_AWS &&
        license_type != PLATFORM_LIC_TYPE_AZURE) {
        fprintf(stderr, "Could not verify license. Code %d\n",
                LICENSE_ERR_INVALID_LICENSE_TYPE);
        return 0;
    }

    int instance_id_len;
    if (license_type == PLATFORM_LIC_TYPE_AWS)
        instance_id_len = EC2_INSTANCE_ID_LEN;
    else  // AZURE (guaranteed due to the earlier check)
        instance_id_len = AZURE_INSTANCE_ID_LEN;

    lic_instance_id.assign(data.begin() + sizeof(int),
                           data.begin() + sizeof(int) + instance_id_len);
    std::vector<char> signature;  // to store signature contained in license
    signature.assign(data.begin() + sizeof(int) + instance_id_len, data.end());

    EVP_PKEY *key = get_public_key();
    if (!key) {
        // ERR_print_errors_fp(stderr);  // print ssl errors
        fprintf(stderr, "Error obtaining public key\n");
        return 0;
    }

    if (!verify_license(key, lic_instance_id.c_str(), instance_id_len,
                        (unsigned char *)signature.data(), signature.size())) {
        // ERR_print_errors_fp(stderr);  // print ssl errors
        fprintf(stderr, "Could not verify license. Code %d\n",
                LICENSE_ERR_INVALID_LICENSE);
        return 0;
    }
    EVP_PKEY_free(key);

    return 1;
}

namespace {
/**
 * Callback used by CURL to store the result of a HTTP GET operation.
 * @param[in] in: data obtained by CURL
 * @param[in] size: size of elements
 * @param[in] num: number of elements
 * @param[in,out]: container where we are adding the data read by CURL.
 */
std::size_t callback(const char *in, std::size_t size, std::size_t num,
                     std::string *out) {
    const std::size_t totalBytes(size * num);
    out->append(in, totalBytes);
    return totalBytes;
}
}  // namespace

/**
 * Verify license for this EC2 instance. Obtains the instance ID of the
 * instance that this is running on and compares it with the instance ID from
 * the license.
 */
static int verify_license_aws(std::string &lic_instance_id) {
    const std::string url(
        "http://169.254.169.254/latest/dynamic/instance-identity/document");

    // Even though the check only takes place on one rank,
    // we do the check PLATFORM_LIC_CHECK_MAX_RETRIES+1 times for
    // better resilience.
    for (int i = 0; i <= PLATFORM_LIC_CHECK_MAX_RETRIES; i++) {
        CURL *curl = curl_easy_init();

        // Set remote URL
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        // Don't use IPv6, which would increase DNS resolution time
        curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);

        // Time out after 10 seconds
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10);

        // Response information
        int http_code = 0;
        std::string http_data;

        // Set data handling function
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, callback);

        // Set data container (will be passed as the last parameter to the
        // callback handling function)
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &http_data);

        // Run HTTP GET command, capture HTTP response code, and clean up
        curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_easy_cleanup(curl);

        if (http_code == 200) {
            char *endptr;
            JsonValue value;
            JsonAllocator allocator;
            int status = jsonParse((char *)http_data.c_str(), &endptr, &value,
                                   allocator);
            if (status != JSON_OK) {
                fprintf(stderr, "Could not verify license. Code %d\n",
                        LICENSE_ERR_JSON_PARSE);
                return 0;
            }
            for (auto i : value) {
                if (strcmp(i->key, "instanceId") == 0) {
                    std::string cur_instance_id = i->value.toString();
                    if (cur_instance_id != lic_instance_id) {
                        fprintf(stderr, "Could not verify license. Code %d\n",
                                LICENSE_ERR_INVALID_INSTANCE);
                        return 0;
                    }
                    return 1;
                }
            }
            fprintf(stderr, "Could not verify license. Code %d\n",
                    LICENSE_ERR_INSTANCE_ID_NOT_FOUND);
            return 0;
        } else {
            // Sleep and retry
            // Simple exponential backoff.
            const double sleep_secs =
                std::min((std::exp2(i) * 0.25) + ((std::rand() % 10) / 100.0),
                         PLATFORM_LIC_CHECK_RETRY_MAX_DELAY);
            sleep(sleep_secs);
        }
    }
    fprintf(stderr, "Could not verify license. Code %d\n",
            LICENSE_ERR_DOWNLOAD);
    return 0;
}

/**
 * Verify license for this Azure instance. Obtains the instance ID of the
 * instance that this is running on and compares it with the instance ID from
 * the license.
 */
static int verify_license_azure(std::string &lic_instance_id) {
    // Can get the Azure VM Unique ID with curl:
    // https://docs.microsoft.com/en-us/azure/virtual-machines/windows/instance-metadata-service?tabs=linux#sample-1-tracking-vm-running-on-azure
    // curl -H Metadata:true --noproxy "*"
    // "http://169.254.169.254/metadata/instance/compute/vmId?api-version=2017-08-01&format=text"
    const std::string url(
        "http://169.254.169.254/metadata/instance/compute/"
        "vmId?api-version=2017-08-01&format=text");

    // Even though the check only takes place on one rank,
    // we do the check PLATFORM_LIC_CHECK_MAX_RETRIES+1 times for
    // better resilience.
    for (int i = 0; i <= PLATFORM_LIC_CHECK_MAX_RETRIES; i++) {
        CURL *curl = curl_easy_init();

        // Set remote URL
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        // Don't use IPv6, which would increase DNS resolution time
        curl_easy_setopt(curl, CURLOPT_IPRESOLVE, CURL_IPRESOLVE_V4);

        // Time out after 10 seconds
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10);

        // set custom header "Metadata:true"
        struct curl_slist *list = NULL;
        list = curl_slist_append(list, "Metadata:true");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);

        // --noproxy "*"
        curl_easy_setopt(curl, CURLOPT_NOPROXY, "*");

        // Response information
        int http_code = 0;
        std::string http_data;

        // Set data handling function
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, callback);

        // Set data container (will be passed as the last parameter to the
        // callback handling function)
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &http_data);

        // Run HTTP GET command, capture HTTP response code, and clean up
        curl_easy_perform(curl);
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_easy_cleanup(curl);
        curl_slist_free_all(list);

        if (http_code == 200) {
            if (http_data != lic_instance_id) {
                fprintf(stderr, "Could not verify license. Code %d\n",
                        LICENSE_ERR_INVALID_INSTANCE);
                return 0;
            }
            return 1;
        } else {
            // Sleep and retry
            // Simple exponential backoff.
            const double sleep_secs =
                std::min((std::exp2(i) * 0.25) + ((std::rand() % 10) / 100.0),
                         PLATFORM_LIC_CHECK_RETRY_MAX_DELAY);
            sleep(sleep_secs);
        }
    }
    fprintf(stderr, "Could not verify license. Code %d\n",
            LICENSE_ERR_DOWNLOAD);
    return 0;
}

static int verify_license_platform() {
    /**
     * We only perform the license check on rank 0.
     * Instance Metadata Services have API limits per
     * node per second. In case of Azure this is very low (5)
     * and hence it's essential to only do the check as few times as possible.
     * Technically, we could be doing the check on one rank
     * for each node, but that would only be useful in very
     * malicious use cases (e.g. someone uses one node
     * that was provisioned by the platform and has the license,
     * but the rest don't.)
     */
    int success = 0;
    if (dist_get_rank() == 0) {
        int license_type;
        std::string instance_id;
        int rc = get_license_platform(license_type, instance_id);
        if (rc == 0) {
            success = 0;
        } else {
            if (license_type == PLATFORM_LIC_TYPE_AWS)
                success = verify_license_aws(instance_id);
            if (license_type == PLATFORM_LIC_TYPE_AZURE)
                success = verify_license_azure(instance_id);
        }
    }
    MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return success;
}

#endif  // CHECK_LICENSE_PLATFORM

MPI_Datatype decimal_mpi_type = MPI_DATATYPE_NULL;

#if defined(CHECK_LICENSE_EXPIRED) || defined(CHECK_LICENSE_CORE_COUNT)

/**
 * Determine if current date is greater than expiration date.
 * @param exp_year: expiration year
 * @param exp_month: expiration month
 * @param exp_day: expiration day
 * @return : true if expired, false otherwise
 */
static bool is_expired(int exp_year, int exp_month, int exp_day) {
    std::time_t t = std::time(0);  // get time now
    std::tm *now = std::localtime(&t);
    int year_now = now->tm_year + 1900;
    int month_now = now->tm_mon + 1;
    int day_now = now->tm_mday;

    if (year_now > exp_year) return true;
    if (year_now == exp_year) {
        if (month_now > exp_month) return true;
        if (month_now == exp_month) return day_now > exp_day;
    }
    return false;
}

static int num_days_till_license_expiration(int exp_year, int exp_month,
                                            int exp_day) {
    /**
     * Return the number of days between expiry and current day. Return -1 in
     * case of error.
     * @param exp_year: expiration year
     * @param exp_month: expiration month
     * @param exp_day: expiration day
     * @return : number of days till expiration
     */

    // Copied from
    // https://stackoverflow.com/questions/14218894/number-of-days-between-two-dates-c

    struct std::tm exp = {0, 0, 0, exp_day, exp_month - 1, exp_year - 1900};
    std::time_t expiration_ts = std::mktime(&exp);
    std::time_t now_ts = std::time(nullptr);  // get time now

    int num_days_left = 0;

    if (expiration_ts != (std::time_t)(-1) && now_ts != (std::time_t)(-1)) {
        num_days_left =
            (int)(std::difftime(expiration_ts, now_ts) / (60 * 60 * 24));
    } else {
        num_days_left = -1;
    }
    return num_days_left;
}

#endif  // CHECK_LICENSE_EXPIRED) || CHECK_LICENSE_CORE_COUNT

PyMODINIT_FUNC PyInit_hdist(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "hdist", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

#if defined(CHECK_LICENSE_PLATFORM)
    int num_pes_plat;
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes_plat);

    if ((num_pes_plat > FREE_MAX_CORES) && !verify_license_platform()) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid license\n");
        return NULL;
    }
#endif

#if defined(CHECK_LICENSE_EXPIRED) || defined(CHECK_LICENSE_CORE_COUNT)
    int num_pes;
    int max_cores = FREE_MAX_CORES;
    int year;
    int month;
    int day;
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    if (num_pes > FREE_MAX_CORES) {
        // get max cores and expiration date from license, and verify license
        // using digital signature with asymmetric cryptography

        bool b64_encoded = true;  // license encoded in Base64
        std::vector<char> data;   // store license
        int read_license = read_license_common(data, b64_encoded);
        if (!read_license) {
            PyErr_SetString(PyExc_RuntimeError, "Error reading license");
            return NULL;
        }
        const int *_data = (int *)data.data();
        int license_type = _data[0] >> 16;
        if (license_type != 0) {
            fprintf(stderr, "Invalid license type\n");
            return NULL;
        }
        max_cores = _data[1];
        year = _data[2];
        month = _data[3];
        day = _data[4];
        std::vector<char> signature;  // to store signature contained in license
        signature.assign(data.begin() + sizeof(int) * 5, data.end());
        std::vector<int> msg = {max_cores, year, month, day};

        EVP_PKEY *key = get_public_key();
        if (!key) {
            // ERR_print_errors_fp(stderr);  // print ssl errors
            PyErr_SetString(PyExc_RuntimeError, "Error obtaining public key");
            return NULL;
        }

        if (!verify_license(key, msg.data(), msg.size() * sizeof(int),
                            (unsigned char *)signature.data(),
                            signature.size())) {
            // ERR_print_errors_fp(stderr);  // print ssl errors
            PyErr_SetString(PyExc_RuntimeError, "Invalid license\n");
            return NULL;
        }
        EVP_PKEY_free(key);
    }
#endif

#ifdef CHECK_LICENSE_EXPIRED
    // check expiration date
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    if ((num_pes > FREE_MAX_CORES) && is_expired(year, month, day)) {
        PyErr_SetString(PyExc_RuntimeError, "Bodo license has expired");
        return NULL;
    }

    // license countdown message if within EXPIRATION_COUNTDOWN_DAYS
    // days of license expiry and using more than FREE_MAX_CORES cores
    if ((num_pes > FREE_MAX_CORES)) {
        int countdown_days = num_days_till_license_expiration(year, month, day);
        if (countdown_days < 0) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Error in license countdown check.");
            return NULL;
        }
        // only print reminder on rank 0
        if (countdown_days < EXPIRATION_COUNTDOWN_DAYS && rank == 0) {
            fprintf(stdout, "Reminder: Bodo License will expire in %d days.\n",
                    countdown_days);
        }
    }
#endif

    // make sure MPI is initialized, assuming this will be called
    // on all processes
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) MPI_Init(NULL, NULL);

#ifdef CHECK_LICENSE_CORE_COUNT
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    if (num_pes > max_cores) {
        char error_msg[100];
        sprintf(error_msg, "License is for %d cores. Max core count exceeded.",
                max_cores);
        PyErr_SetString(PyExc_RuntimeError, error_msg);
        return NULL;
    }
#endif

    // initialize decimal_mpi_type
    // TODO: free when program exits
    if (decimal_mpi_type == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(2, MPI_LONG_LONG_INT, &decimal_mpi_type);
        MPI_Type_commit(&decimal_mpi_type);
    }

    int decimal_bytes;
    MPI_Type_size(decimal_mpi_type, &decimal_bytes);
    // decimal_value should be exactly 128 bits to match Python
    if (decimal_bytes != 16)
        std::cerr << "invalid decimal mpi type size" << std::endl;

    PyObject_SetAttrString(m, "dist_get_rank",
                           PyLong_FromVoidPtr((void *)(&dist_get_rank)));
    PyObject_SetAttrString(m, "dist_get_size",
                           PyLong_FromVoidPtr((void *)(&dist_get_size)));
    PyObject_SetAttrString(m, "dist_get_start",
                           PyLong_FromVoidPtr((void *)(&dist_get_start)));
    PyObject_SetAttrString(m, "dist_get_end",
                           PyLong_FromVoidPtr((void *)(&dist_get_end)));
    PyObject_SetAttrString(
        m, "dist_get_node_portion",
        PyLong_FromVoidPtr((void *)(&dist_get_node_portion)));
    PyObject_SetAttrString(m, "dist_get_time",
                           PyLong_FromVoidPtr((void *)(&dist_get_time)));
    PyObject_SetAttrString(m, "get_time",
                           PyLong_FromVoidPtr((void *)(&get_time)));
    PyObject_SetAttrString(m, "barrier",
                           PyLong_FromVoidPtr((void *)(&barrier)));

    PyObject_SetAttrString(m, "dist_reduce",
                           PyLong_FromVoidPtr((void *)(&dist_reduce)));
    PyObject_SetAttrString(m, "dist_exscan",
                           PyLong_FromVoidPtr((void *)(&dist_exscan)));
    PyObject_SetAttrString(m, "dist_arr_reduce",
                           PyLong_FromVoidPtr((void *)(&dist_arr_reduce)));
    PyObject_SetAttrString(m, "dist_irecv",
                           PyLong_FromVoidPtr((void *)(&dist_irecv)));
    PyObject_SetAttrString(m, "dist_isend",
                           PyLong_FromVoidPtr((void *)(&dist_isend)));
    PyObject_SetAttrString(m, "dist_recv",
                           PyLong_FromVoidPtr((void *)(&dist_recv)));
    PyObject_SetAttrString(m, "dist_send",
                           PyLong_FromVoidPtr((void *)(&dist_send)));
    PyObject_SetAttrString(m, "dist_wait",
                           PyLong_FromVoidPtr((void *)(&dist_wait)));
    PyObject_SetAttrString(
        m, "dist_get_item_pointer",
        PyLong_FromVoidPtr((void *)(&dist_get_item_pointer)));
    PyObject_SetAttrString(m, "get_dummy_ptr",
                           PyLong_FromVoidPtr((void *)(&get_dummy_ptr)));
    PyObject_SetAttrString(m, "c_gather_scalar",
                           PyLong_FromVoidPtr((void *)(&c_gather_scalar)));
    PyObject_SetAttrString(m, "c_gatherv",
                           PyLong_FromVoidPtr((void *)(&c_gatherv)));
    PyObject_SetAttrString(m, "c_allgatherv",
                           PyLong_FromVoidPtr((void *)(&c_allgatherv)));
    PyObject_SetAttrString(m, "c_scatterv",
                           PyLong_FromVoidPtr((void *)(&c_scatterv)));
    PyObject_SetAttrString(m, "c_bcast",
                           PyLong_FromVoidPtr((void *)(&c_bcast)));
    PyObject_SetAttrString(m, "c_alltoallv",
                           PyLong_FromVoidPtr((void *)(&c_alltoallv)));
    PyObject_SetAttrString(m, "c_alltoall",
                           PyLong_FromVoidPtr((void *)(&c_alltoall)));
    PyObject_SetAttrString(m, "allgather",
                           PyLong_FromVoidPtr((void *)(&allgather)));
    PyObject_SetAttrString(m, "finalize",
                           PyLong_FromVoidPtr((void *)(&finalize)));
    PyObject_SetAttrString(m, "oneD_reshape_shuffle",
                           PyLong_FromVoidPtr((void *)(&oneD_reshape_shuffle)));
    PyObject_SetAttrString(m, "permutation_int",
                           PyLong_FromVoidPtr((void *)(&permutation_int)));
    PyObject_SetAttrString(
        m, "permutation_array_index",
        PyLong_FromVoidPtr((void *)(&permutation_array_index)));

    // add actual int value to module
    PyObject_SetAttrString(m, "mpi_req_num_bytes",
                           PyLong_FromSize_t(get_mpi_req_num_bytes()));
    PyObject_SetAttrString(m, "ANY_SOURCE",
                           PyLong_FromLong((long)MPI_ANY_SOURCE));
    return m;
}
