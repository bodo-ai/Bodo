#pragma once

// Generate a Parquet file name prefix for each iteration in such a way that
// iteration file names are lexicographically sorted. This allows the data
// to be read in the written order later. Same as the Bodo JIT
// implementation:
// https://github.com/bodo-ai/Bodo/blob/ebf3022eb443d7562dbc0282c346b4d8cf65f209/bodo/io/stream_parquet_write.py#L337
inline std::string get_fname_prefix(int64_t iter) {
    std::string base_prefix = "part-";

    int MAX_ITER = 1000;
    int n_max_digits = static_cast<int>(std::ceil(std::log10(MAX_ITER)));

    // Number of prefix characters to add ("batch" number)
    int n_prefix =
        (iter == 0)
            ? 0
            : static_cast<int>(std::floor(std::log(iter) / std::log(MAX_ITER)));

    std::string iter_str = std::to_string(iter);
    int n_zeros =
        ((n_prefix + 1) * n_max_digits) - static_cast<int>(iter_str.length());
    iter_str = std::string(n_zeros, '0') + iter_str;

    return base_prefix + std::string(n_prefix, 'b') + iter_str + "-";
}
