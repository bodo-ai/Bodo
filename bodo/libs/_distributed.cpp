// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_distributed.h"
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <ctime>
#include <fstream>
#include <vector>

MPI_Datatype decimal_mpi_type = MPI_DATATYPE_NULL;

#if defined(CHECK_LICENSE_EXPIRED) || defined(CHECK_LICENSE_CORE_COUNT)

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
 * @param sig: signature (obtained when signing license content with Bodo private key)
 * @param slen: length of signature in bytes
 * @return : 1 if verified correctly, 0 otherwise.
 */
static int verify_license(EVP_PKEY *key, void *msg, const size_t mlen,
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

/**
 * Read license from environment variable BODO_LICENSE or from license file.
 * The license contains max cores and expiration date followed by digital
 * signature (of max cores and expiration date).
 * Note that the first int of the decoded license is the total number of bytes
 * of everything in the decoded license.
 * @param[out] num_cores: max cores in the license
 * @param[out] year: expiration year
 * @param[out] month: expiration month
 * @param[out] day: expiration day
 * @param[out] signature: binary digital signature
 * @param[in] b64_encoded: true if license is Base64 encoded
 */
static void read_license(int &num_cores, int &year, int &month, int &day,
                         std::vector<char> &signature, bool b64_encoded) {
    std::vector<char> data;  // store license

    // first check if license is in environment variable
    char *license_text = std::getenv("BODO_LICENSE");
    if (license_text != NULL && strlen(license_text) > 0) {
        data.assign(license_text, license_text + strlen(license_text));
    } else {
        // read from "bodo.lic" file in cwd
        // TODO: check HOME directory or some other directory?
        std::ifstream f("bodo.lic", std::ios::binary);
        if (f.fail()) {
            Bodo_PyErr_SetString(PyExc_RuntimeError, "Bodo license not found");
            return;
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
            Bodo_PyErr_SetString(PyExc_RuntimeError, "Invalid license");
            return;
        }
        int actual_size = ((int *)decoded_data.data())[0];
        decoded_data.resize(actual_size);
        data = std::move(decoded_data);
    }

    const int *msg = (int *)data.data();
    num_cores = msg[1];
    year = msg[2];
    month = msg[3];
    day = msg[4];
    signature.assign(data.begin() + sizeof(int) * 5, data.end());
}

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

#endif

PyMODINIT_FUNC PyInit_hdist(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "hdist", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

#if defined(CHECK_LICENSE_EXPIRED) || defined(CHECK_LICENSE_CORE_COUNT)
    // get max cores and expiration date from license, and verify license
    // using digital signature with asymmetric cryptography
    int max_cores = -1;
    int year, month, day;
    std::vector<char> signature;  // to store signature contained in license

    bool b64_encoded = true;  // license encoded in Base64
    read_license(max_cores, year, month, day, signature, b64_encoded);
    if (max_cores == -1)
        return NULL;  // error has been set in read_license function
    std::vector<int> msg = {max_cores, year, month, day};

    EVP_PKEY *key = get_public_key();
    if (!key) {
        // ERR_print_errors_fp(stderr);  // print ssl errors
        Bodo_PyErr_SetString(PyExc_RuntimeError, "Error obtaining public key");
        return NULL;
    }

    if (!verify_license(key, msg.data(), msg.size() * sizeof(int),
                        (unsigned char *)signature.data(), signature.size())) {
        // ERR_print_errors_fp(stderr);  // print ssl errors
        Bodo_PyErr_SetString(PyExc_RuntimeError, "Invalid license\n");
        return NULL;
    }
    EVP_PKEY_free(key);
#endif

#ifdef CHECK_LICENSE_EXPIRED
    // check expiration date
    if (is_expired(year, month, day)) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Bodo trial period has expired!");
        return NULL;
    }
#endif

    // make sure MPI is initialized, assuming this will be called
    // on all processes
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) MPI_Init(NULL, NULL);

#ifdef CHECK_LICENSE_CORE_COUNT
    int num_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    if (num_pes > max_cores) {
        char error_msg[100];
        sprintf(error_msg, "License is for %d cores. Max core count exceeded.",
                max_cores);
        Bodo_PyErr_SetString(PyExc_RuntimeError, error_msg);
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
    PyObject_SetAttrString(m, "comm_req_alloc",
                           PyLong_FromVoidPtr((void *)(&comm_req_alloc)));
    PyObject_SetAttrString(m, "req_array_setitem",
                           PyLong_FromVoidPtr((void *)(&req_array_setitem)));
    PyObject_SetAttrString(m, "dist_waitall",
                           PyLong_FromVoidPtr((void *)(&dist_waitall)));
    PyObject_SetAttrString(m, "comm_req_dealloc",
                           PyLong_FromVoidPtr((void *)(&comm_req_dealloc)));

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
