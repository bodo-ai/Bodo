#include "_murmurhash3.h"
//#include <numpy/arrayobject.h>
#include <iostream> //TODO: take this out 
#include <bitset> //TODO: take this out 

#define HASHSEED 0xb0d01289

//hash string
//out hash: uint32_t*, 32 bits
void hash_string_32(const char* str, const int len, uint32_t* out_hash){
	MurmurHash3_x64_32 ((const void*)str, len, HASHSEED, (void*)out_hash);
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
