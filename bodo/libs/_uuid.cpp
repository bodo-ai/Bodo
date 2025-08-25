#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "_bodo_common.h"

void outputUUID(char* output, const boost::uuids::uuid& uuid) {
    std::string uuid_str = boost::uuids::to_string(uuid);
    strncpy(output, uuid_str.c_str(), 36);
}

/**
 * Create a randomly generated UUID and store the string representation in
 * output. This requires that output must be at least 37 bytes (36 + null byte).
 */
void uuidV4(char* output) {
    auto uuid = boost::uuids::random_generator()();
    outputUUID(output, uuid);
}

/**
 * Create a name based UUID and store the string representation in
 * output. This requires that output must be at least 37 bytes. Note that
 * namespace_ must be a valid string representation of a UUID. If not, this
 * method will set output to be the empty string.
 */
void uuidV5(char* output, char* namespace_, int64_t ns_length, char* name,
            int64_t name_length) {
    std::string namespace_str(namespace_, ns_length);
    boost::uuids::string_generator gen;
    boost::uuids::uuid ns_uuid;
    try {
        ns_uuid = gen(namespace_str);
    } catch (...) {
        output[0] = 0;
        return;
    }

    if (ns_uuid.version() == boost::uuids::uuid::version_unknown) {
        output[0] = 0;
        return;
    }

    std::string namestr(name, name_length);
    boost::uuids::name_generator named_gen(ns_uuid);
    auto uuid = named_gen(namestr);

    outputUUID(output, uuid);
}

PyMODINIT_FUNC PyInit_uuid_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "uuid_cpp", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, uuidV4);
    SetAttrStringFromVoidPtr(m, uuidV5);
    return m;
}
