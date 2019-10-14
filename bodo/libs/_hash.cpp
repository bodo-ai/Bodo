#include "_murmurhash3.h"
//#include <numpy/arrayobject.h>
#include <iostream> //TODO: take this out 
#include <bitset> //TODO: take this out 

#define HASHSEED 0xb0d01289

//hash string
//out hash: uint32_t*, 32 bits
void hash_string_32(const char* str, const int len, uint32_t* out_hash){
	MurmurHash3_x64_32 (str, len, HASHSEED, (void*)out_hash);
}

template <class T>
void hash_inner_32(T* data, uint32_t* out_hash){
    MurmurHash3_x64_32 ((const void *)data, sizeof(T), HASHSEED, (void*)out_hash);
}


//hash string(for testing purpose)
//out hash: uint64_t*, 128 bits
void hash_string_128(const char* str, const int len, uint64_t* out_hash){
	MurmurHash3_x64_128 ((const void*)str, len, HASHSEED, (void*)out_hash);
}

//hash string array
//out_hashes: uint32_t*, each hash has 32 bits
void hash_array_string(uint32_t* out_hashes, char* data, uint32_t* offsets, size_t n_rows)
{
    uint32_t start_offset = 0;
    for (size_t i=0; i<n_rows; i++) {
        uint32_t end_offset = offsets[i+1];
        uint32_t len = end_offset - start_offset;
        std::string val(&data[start_offset], len);
        const char* val_chars = val.c_str();
        hash_string_32(val_chars, (const int)len, &out_hashes[i]);
        start_offset = end_offset;
    }
}

void test_hash_string(){
	const char* str_list[] = {"what", "837r89efjahjkhdakjhfak", 
                    "dsahk shdfjakh akhdja khjdskfakjsakkb, dhafjkfhakk"};
    size_t list_len = 3;

    for(size_t i = 0; i<list_len; i++){
        const char* str = str_list[i];
        const int len = strlen(str);


        uint32_t out_hash_32 = 0;
        hash_string_32(str, len, &out_hash_32);

        uint64_t out_hash_128[] = { 0, 0 };
        hash_string_128(str, len, out_hash_128);
        uint32_t out_hash_128_32 = out_hash_128[0] & 0xFFFFFFFF;

        if((out_hash_32) == out_hash_128_32){
            std::cout << str << ": they are equal " << '\n';
            std::bitset<32> y(out_hash_32);
            std::cout << "32 hash is " << y << '\n';
        }
        else{
            std::cout << str << ": they are not equal " << '\n';
            std::bitset<32> y(out_hash_32);
            std::cout << "32 hash is " << y << '\n';

            std::bitset<32> x(out_hash_128_32);
            std::cout << "128 hash is " << x << '\n';
        }
    }
}

void test_hash_string_array(){
    uint32_t out_hashes[] = {0, 0, 0, 0, 0};
    std::string str = "ab123412345678900987654321asdqwezxcrtyfghvbnyuihjknm";
    char* data = (char*)str.c_str();
    uint32_t offsets[] = {0, 0, 2, 6, 26, 53};
    size_t n_rows = 5;
    hash_array_string(out_hashes, data, offsets, n_rows);
    std::cout << "HASHING STRING ARRAY \n";
    for (size_t i = 0; i < n_rows; i++){
        std::cout << std::bitset<32>(out_hashes[i]) << '\n';
    }
}

int main()
{
    test_hash_string();
    test_hash_string_array();
}

// //hash string(for testing purpose)
// //out hash: uint64_t*, 128 bits
// void hash_string_128(const char* str, const int len, uint64_t* out_hash){
//     MurmurHash3_x64_128 (str, len, HASHSEED, (void*)out_hash);
// }

// //hash string array
// //out_hashes: uint32_t*, each hash has 32 bits
// void hash_array_string(uint32_t* out_hashes, char* data, uint32_t* offsets, size_t n_rows)
// {
//     uint32_t start_offset = 0;
//     for (size_t i=0; i<n_rows; i++) {
//         uint32_t end_offset = offsets[i+1];
//         uint32_t len = end_offset - start_offset;
//         std::string val(&data[start_offset], len);
//         const char* val_chars = val.c_str();
//         hash_string_32(val_chars, (const int)len, &out_hashes[i]);
//         start_offset = end_offset;
//     }
// }

// void test_hash_string(){
// 	const char* str_list[] = {"what", "837r89efjahjkhdakjhfak", 
//                     "dsahk shdfjakh akhdja khjdskfakjsakkb, dhafjkfhakk"};
//     size_t list_len = 3;

//     for(size_t i = 0; i<list_len; i++){
//         const char* str = str_list[i];
//         const int len = strlen(str);


//         uint32_t out_hash_32 = 0;
//         hash_string_32(str, len, &out_hash_32);

//         uint64_t out_hash_128[] = { 0, 0 };
//         hash_string_128(str, len, out_hash_128);
//         uint32_t out_hash_128_32 = out_hash_128[0] & 0xFFFFFFFF;

//         if((out_hash_32) == out_hash_128_32){
//             std::cout << str << ": they are equal " << '\n';
//             std::bitset<32> y(out_hash_32);
//             std::cout << "32 hash is " << y << '\n';
//         }
//         else{
//             std::cout << str << ": they are not equal " << '\n';
//             std::bitset<32> y(out_hash_32);
//             std::cout << "32 hash is " << y << '\n';

//             std::bitset<32> x(out_hash_128_32);
//             std::cout << "128 hash is " << x << '\n';
//         }
//     }
// }

// void test_hash_inner_32(){
//     std::cout << "TESTING HASH INNER \n";
//     int8_t v_1 = -120;
//     uint32_t out_1 = 0;
//     hash_inner_32<int8_t>(&v_1, &out_1);
//     std::cout << "int8_hash" << static_cast<int16_t>(v_1) << ": "<< std::bitset<32>(out_1) << '\n';

//     uint8_t v_2 = 250;
//     uint32_t out_2 = 0;
//     hash_inner_32<uint8_t>(&v_2, &out_2);
//     std::cout << "uint8_hash" << static_cast<uint16_t>(v_2) << ": "<< std::bitset<32>(out_2) << '\n';

//     int16_t v_3 = -30000;
//     uint32_t out_3 = 0;
//     hash_inner_32<int16_t>(&v_3, &out_3);
//     std::cout << "int16_hash" << v_3 << ": "<< std::bitset<32>(out_3) << '\n';

//     uint16_t v_4 = 60000;
//     uint32_t out_4 = 0;
//     hash_inner_32<uint16_t>(&v_4, &out_4);
//     std::cout << "uint16_hash" << v_4 << ": "<< std::bitset<32>(out_4) << '\n';

//     int32_t v_5 = -2147000000;
//     uint32_t out_5 = 0;
//     hash_inner_32<int32_t>(&v_5, &out_5);
//     std::cout << "int32_hash" << v_5 << ": "<< std::bitset<32>(out_5) << '\n';

//     uint32_t v_6 = 4294000000;
//     uint32_t out_6 = 0;
//     hash_inner_32<uint32_t>(&v_6, &out_6);
//     std::cout << "uint32_hash" << v_6 << ": "<< std::bitset<32>(out_6) << '\n';

//     int64_t v_7 = -9000000000000000;
//     uint32_t out_7 = 0;
//     hash_inner_32<int64_t>(&v_7, &out_7);
//     std::cout << "int64_hash" << v_7 << ": "<< std::bitset<32>(out_7) << '\n';

//     uint64_t v_8 = 18000000000000000;
//     uint32_t out_8 = 0;
//     hash_inner_32<uint64_t>(&v_8, &out_8);
//     std::cout << "uint64_hash" << v_8 << ": "<< std::bitset<32>(out_8) << '\n';

//     float v_9 = 18000000000000000;
//     uint32_t out_9 = 0;
//     hash_inner_32<float>(&v_9, &out_9);
//     std::cout << "float_hash" << v_9 << ": "<< std::bitset<32>(out_9) << '\n';

//     double v_10 = 18000000000000000;
//     uint32_t out_10 = 0;
//     hash_inner_32<double >(&v_10, &out_10);
//     std::cout << "double _hash" << v_10 << ": "<< std::bitset<32>(out_10) << '\n';
// }

// void test_hash_string_array(){
//     uint32_t out_hashes[] = {0, 0, 0, 0, 0};
//     std::string str = "ab123412345678900987654321asdqwezxcrtyfghvbnyuihjknm";
//     char* data = (char*)str.c_str();
//     uint32_t offsets[] = {0, 0, 2, 6, 26, 53};
//     size_t n_rows = 5;
//     hash_array_string(out_hashes, data, offsets, n_rows);
//     std::cout << "HASHING STRING ARRAY \n";
//     for (size_t i = 0; i < n_rows; i++){
//         std::cout << std::bitset<32>(out_hashes[i]) << '\n';
//     }
// }

// int main()
// {
//     test_hash_string();
//     test_hash_inner_32();
//     // test_hash_string_array();
// }
