// Extracted from duckdb fmt code to use with vendorded files to avoid building the entire customized fmt which conflicts with regular fmt
// https://github.com/duckdb/duckdb/blob/9a396a9bd7ef3cad31c5d4f3432a280edece89ab/third_party/fmt/include/fmt/format-inl.h#L185

#pragma once


namespace duckdb_fmt {
namespace internal {

    template <typename T = void> struct basic_data {
        static const char digits[];
      };

    template <typename T>
    const char basic_data<T>::digits[] =
        "0001020304050607080910111213141516171819"
        "2021222324252627282930313233343536373839"
        "4041424344454647484950515253545556575859"
        "6061626364656667686970717273747576777879"
        "8081828384858687888990919293949596979899";

    struct data : basic_data<> {};
}
}
