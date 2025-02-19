#include <Python.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

#include "_array_utils.h"
#include "_bodo_common.h"
#include "_bodo_tdigest.h"
#include "_distributed.h"
#include "_mpi.h"

constexpr int root = 0;

template <typename T, typename Alloc>
std::pair<T, T> get_lower_upper_kth_parallel(std::vector<T, Alloc> my_array,
                                             int64_t total_size, int myrank,
                                             int n_pes, int64_t k,
                                             int type_enum) {
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    int64_t local_size = my_array.size();
    // This random number generation is deterministic
    std::default_random_engine r_engine(myrank);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    int64_t sample_size = (int64_t)(pow(10.0, 5.0) / n_pes);  // 100000 total
    int64_t my_sample_size = std::min(sample_size, local_size);

    std::vector<T> my_sample;
    for (int64_t i = 0; i < my_sample_size; i++) {
        int64_t index = (int64_t)(local_size * uniform_dist(r_engine));
        my_sample.push_back(my_array[index]);
    }
    /* select sample */
    // get total sample size;
    bodo::vector<T> all_sample_vec;
    int *rcounts = new int[n_pes];
    int *displs = new int[n_pes];
    int total_sample_size = 0;
    // gather the sample sizes
    CHECK_MPI(MPI_Gather(&my_sample_size, 1, MPI_INT, rcounts, 1, MPI_INT, root,
                         MPI_COMM_WORLD),
              "get_lower_upper_kth_parallel: MPI error on MPI_Gather:");
    // calculate size and displacements on root
    if (myrank == root) {
        for (int i = 0; i < n_pes; i++) {
            // printf("rc %d\n", rcounts[i]);
            displs[i] = total_sample_size;
            total_sample_size += rcounts[i];
        }
        // printf("total sample size: %d\n", total_sample_size);
        all_sample_vec.resize(total_sample_size);
    }
    // gather sample data
    CHECK_MPI(MPI_Gatherv(my_sample.data(), my_sample_size, mpi_typ,
                          all_sample_vec.data(), rcounts, displs, mpi_typ, root,
                          MPI_COMM_WORLD),
              "get_lower_upper_kth_parallel: MPI error on MPI_Gatherv:");
    T k1_val;
    T k2_val;
    if (myrank == root) {
        int local_k = (int)(k * (total_sample_size / (double)total_size));
        // printf("k:%ld local_k:%d\n", k, local_k);
        int k1 = (int)(local_k - sqrt(total_sample_size * log(total_size)));
        int k2 = (int)(local_k + sqrt(total_sample_size * log(total_size)));
        k1 = std::max(k1, 0);
        k2 = std::min(k2, total_sample_size - 1);
        // printf("k1: %d k2: %d\n", k1, k2);
        std::nth_element(all_sample_vec.begin(), all_sample_vec.begin() + k1,
                         all_sample_vec.end());
        k1_val = all_sample_vec[k1];
        std::nth_element(all_sample_vec.begin(), all_sample_vec.begin() + k2,
                         all_sample_vec.end());
        k2_val = all_sample_vec[k2];
        // printf("k1: %d k2: %d k1_val: %lf k2_val:%lf\n", k1, k2, k1_val,
        // k2_val);
    }
    CHECK_MPI(MPI_Bcast(&k1_val, 1, mpi_typ, root, MPI_COMM_WORLD),
              "get_lower_upper_kth_parallel: MPI error on MPI_Bcast[k1]:");
    CHECK_MPI(MPI_Bcast(&k2_val, 1, mpi_typ, root, MPI_COMM_WORLD),
              "get_lower_upper_kth_parallel: MPI error on MPI_Bcast[k2]:");
    // cleanup
    delete[] rcounts;
    delete[] displs;
    return std::make_pair(k1_val, k2_val);
}

template <typename T, typename Alloc>
T small_get_nth_parallel(std::vector<T, Alloc> my_array, int64_t total_size,
                         int myrank, int n_pes, int64_t k, int type_enum);

template <typename T, typename Alloc>
T get_nth_parallel(std::vector<T, Alloc> my_array, int64_t k, int myrank,
                   int n_pes, int type_enum);

double quantile_sequential(void *data, int64_t local_size, double quantile,
                           int type_enum);
double quantile_parallel(void *data, int64_t local_size, int64_t total_size,
                         double quantile, int type_enum);
template <typename T>
double quantile_int(T *data, int64_t local_size, double at, int type_enum,
                    bool parallel);
template <typename T>
double quantile_float(T *data, int64_t local_size, double quantile,
                      int type_enum, bool parallel);

double quantile_dispatch(void *data, int64_t local_size, double quantile,
                         double at, int type_enum, bool parallel);

double approx_percentile_py_entrypt(array_info *arr_raw, double percentile,
                                    bool parallel);

double percentile_py_entrypt(array_info *arr_raw, double percentile,
                             bool interpolate, bool parallel);

/** Compute the median of the series.
    @param arr: The array_info used for the computation
    @param parallel: whether to use the parallel algorithm
    @param skipna: whether to skip the nan entries or not.
    @output res: the median as output
 */
void median_series_computation_py_entry(double *res, array_info *arr,
                                        bool parallel, bool skip_na_data);

/* Compute the autocorrelation of the series.
   @output res: the autocorrelation as output
   @param arr: The array_info used for the computation
   @param lag: The lag for the autocorrelation
   @param parallel: whether it is parallel or not
 */
void autocorr_series_computation_py_entry(double *res, array_info *arr,
                                          int64_t lag, bool is_parallel);

/* Compute whether the array is monotonic or not
   @output res : the return value 1 for monotonic and 0 otherwise
   @param arr : the input array
   @param inc_dec : 1 for testing is_monotonic_increasing / 2 for
   is_monotonic_decreasing
   @param parallel: whether it is parallel or not
 */
void compute_series_monotonicity_py_entry(double *res, array_info *arr,
                                          int64_t inc_dec, bool is_parallel);

PyMODINIT_FUNC PyInit_quantile_alg(void) {
    PyObject *m;
    MOD_DEF(m, "quantile_alg", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, quantile_sequential);
    SetAttrStringFromVoidPtr(m, quantile_parallel);
    SetAttrStringFromVoidPtr(m, median_series_computation_py_entry);
    SetAttrStringFromVoidPtr(m, autocorr_series_computation_py_entry);
    SetAttrStringFromVoidPtr(m, compute_series_monotonicity_py_entry);
    SetAttrStringFromVoidPtr(m, approx_percentile_py_entrypt);
    SetAttrStringFromVoidPtr(m, percentile_py_entrypt);

    return m;
}

double quantile_sequential(void *data, int64_t local_size, double quantile,
                           int type_enum) {
    try {
        // return NA if no elements
        if (local_size == 0) {
            return std::nan("");
        }

        double at = quantile * (local_size - 1);
        return quantile_dispatch(data, local_size, quantile, at, type_enum,
                                 false);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
}

double quantile_parallel(void *data, int64_t local_size, int64_t total_size,
                         double quantile, int type_enum) {
    try {
        if (total_size == 0)
            CHECK_MPI(MPI_Allreduce(&local_size, &total_size, 1,
                                    MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD),
                      "quantile_parallel: MPI error on MPI_Allreduce:");

        // return NA if no elements
        if (total_size == 0) {
            return std::nan("");
        }

        double at = quantile * (total_size - 1);
        return quantile_dispatch(data, local_size, quantile, at, type_enum,
                                 true);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
}

double quantile_dispatch(void *data, int64_t local_size, double quantile,
                         double at, int type_enum, bool parallel) {
    switch (type_enum) {
        case Bodo_CTypes::INT8:
            return quantile_int((char *)data, local_size, at, type_enum,
                                parallel);
        case Bodo_CTypes::_BOOL:
            // if we use bool type here, compiler gives a strange error in
            // MPI_Gatherv call, possibly because the MPI type we are using
            // for bool is MPI_UNSIGNED_CHAR (see _distributed.h)
        case Bodo_CTypes::UINT8:
            return quantile_int((unsigned char *)data, local_size, at,
                                type_enum, parallel);
        case Bodo_CTypes::INT32:
            return quantile_int((int *)data, local_size, at, type_enum,
                                parallel);
        case Bodo_CTypes::UINT32:
            return quantile_int((uint32_t *)data, local_size, at, type_enum,
                                parallel);
        case Bodo_CTypes::INT64:
            return quantile_int((int64_t *)data, local_size, at, type_enum,
                                parallel);
        case Bodo_CTypes::UINT64:
            return quantile_int((uint64_t *)data, local_size, at, type_enum,
                                parallel);
        case Bodo_CTypes::FLOAT32:
            return quantile_float((float *)data, local_size, quantile,
                                  type_enum, parallel);
        case Bodo_CTypes::FLOAT64:
            return quantile_float((double *)data, local_size, quantile,
                                  type_enum, parallel);
        default:
            throw std::runtime_error(
                "_quantile_alg.cpp::quantile_dispatch: unknown quantile data "
                "type");
    }

    return -1.0;
}

template <typename T, typename Alloc>
double get_nth_q(std::vector<T, Alloc> my_array, int64_t local_size, int64_t k,
                 int type_enum, int myrank, int n_pes, bool parallel) {
    // get nth element and store in res pointer
    // assuming NA values of floats are already removed
    T val;

    if (parallel) {
        val = get_nth_parallel(my_array, k, myrank, n_pes, type_enum);
    } else {
        // If q is 1.0 we may request a value longer than the array,
        // so return the last element.
        if (k >= local_size) {
            k = local_size - 1;
        }
        // Modifies array in place
        std::nth_element(my_array.begin(), my_array.begin() + k,
                         my_array.end());
        val = my_array[k];
    }
    return (double)val;
}

template <typename T>
double quantile_int(T *data, int64_t local_size, double at, int type_enum,
                    bool parallel) {
    int64_t k1 = (int64_t)at;
    int64_t k2 = k1 + 1;
    double fraction = at - (double)k1;
    bodo::vector<T> my_array(data, data + local_size);

    int myrank, n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    double res1 =
        get_nth_q(my_array, local_size, k1, type_enum, myrank, n_pes, parallel);
    double res2 =
        get_nth_q(my_array, local_size, k2, type_enum, myrank, n_pes, parallel);

    // linear method, TODO: support other methods
    return res1 + (res2 - res1) * fraction;
}

template <typename T>
double quantile_float(T *data, int64_t local_size, double quantile,
                      int type_enum, bool parallel) {
    int myrank, n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    bodo::vector<T> my_array(data, data + local_size);
    // delete NaNs
    my_array.erase(std::remove_if(std::begin(my_array), std::end(my_array),
                                  [](T d) { return std::isnan(d); }),
                   my_array.end());
    local_size = my_array.size();
    // recalculate total size since there could be NaNs
    int64_t total_size = local_size;
    if (parallel) {
        CHECK_MPI(MPI_Allreduce(&local_size, &total_size, 1, MPI_LONG_LONG_INT,
                                MPI_SUM, MPI_COMM_WORLD),
                  "quantile_float: MPI error on MPI_Allreduce:");
    }
    double at = quantile * (total_size - 1);
    int64_t k1 = (int64_t)at;
    int64_t k2 = k1 + 1;
    double fraction = at - (double)k1;

    double res1 =
        get_nth_q(my_array, local_size, k1, type_enum, myrank, n_pes, parallel);
    double res2 =
        get_nth_q(my_array, local_size, k2, type_enum, myrank, n_pes, parallel);

    // linear method, TODO: support other methods
    return res1 + (res2 - res1) * fraction;
}

template <typename T, typename Alloc>
T get_nth_parallel(std::vector<T, Alloc> my_array, int64_t k, int myrank,
                   int n_pes, int type_enum) {
    int64_t local_size = my_array.size();
    int64_t total_size;
    CHECK_MPI(MPI_Allreduce(&local_size, &total_size, 1, MPI_LONG_LONG_INT,
                            MPI_SUM, MPI_COMM_WORLD),
              "get_nth_parallel: MPI error on MPI_Allreduce:");
    // printf("total size: %ld k: %ld\n", total_size, k);
    int64_t threshold = (int64_t)pow(10.0, 7.0);  // 100 million
    // If q is 1.0 we may request a value longer than the array,
    // so return the last element.
    if (k >= total_size) {
        k = total_size - 1;
    }
    // int64_t threshold = 20;
    if (total_size < threshold || n_pes == 1) {
        return small_get_nth_parallel(my_array, total_size, myrank, n_pes, k,
                                      type_enum);
    } else {
        std::pair<T, T> kths = get_lower_upper_kth_parallel(
            my_array, total_size, myrank, n_pes, k, type_enum);
        T k1_val = kths.first;
        T k2_val = kths.second;
        // printf("k1_val: %lf  k2_val: %lf\n", k1_val, k2_val);
        int64_t local_l0_num = 0, local_l1_num = 0, local_l2_num = 0;
        int64_t l0_num = 0, l1_num = 0, l2_num = 0;
        for (auto val : my_array) {
            if (val < k1_val) {
                local_l0_num++;
            }
            if (val >= k1_val && val < k2_val) {
                local_l1_num++;
            }
            if (val >= k2_val) {
                local_l2_num++;
            }
        }
        CHECK_MPI(MPI_Allreduce(&local_l0_num, &l0_num, 1, MPI_LONG_LONG_INT,
                                MPI_SUM, MPI_COMM_WORLD),
                  "get_nth_parallel: MPI error on MPI_Allreduce[l0_num]:");
        CHECK_MPI(MPI_Allreduce(&local_l1_num, &l1_num, 1, MPI_LONG_LONG_INT,
                                MPI_SUM, MPI_COMM_WORLD),
                  "get_nth_parallel: MPI error on MPI_Allreduce[l1_num]:");
        CHECK_MPI(MPI_Allreduce(&local_l2_num, &l2_num, 1, MPI_LONG_LONG_INT,
                                MPI_SUM, MPI_COMM_WORLD),
                  "get_nth_parallel: MPI error on MPI_Allreduce[l2_num]:");
        // printf("set sizes: %ld %ld %ld\n", l0_num, l1_num, l2_num);
        assert(l0_num + l1_num + l2_num == total_size);
        // []----*---o----*-----]
        assert(l0_num < k);

        bodo::vector<T> new_my_array;
        int64_t new_k = k;

        int64_t new_ind = 0;
        if (k < l0_num) {
            // first set
            // printf("first set\n");
            new_my_array.resize(local_l0_num);
            // throw away
            for (auto val : my_array) {
                if (val < k1_val) {
                    new_my_array[new_ind] = val;
                    new_ind++;
                }
            }
            // new_k doesn't change
        } else if (k < l0_num + l1_num) {
            // middle set
            // printf("second set\n");
            new_my_array.resize(local_l1_num);
            for (auto val : my_array) {
                if (val >= k1_val && val < k2_val) {
                    new_my_array[new_ind] = val;
                    new_ind++;
                }
            }
            new_k -= l0_num;
        } else {
            // last set
            // printf("last set\n");
            new_my_array.resize(local_l2_num);
            for (auto val : my_array) {
                if (val >= k2_val) {
                    new_my_array[new_ind] = val;
                    new_ind++;
                }
            }
            new_k -= (l0_num + l1_num);
        }
        return get_nth_parallel(new_my_array, new_k, myrank, n_pes, type_enum);
    }
    return (T)-1.0;
}

template <typename T, typename Alloc>
T small_get_nth_parallel(std::vector<T, Alloc> my_array, int64_t total_size,
                         int myrank, int n_pes, int64_t k, int type_enum) {
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    T res;
    int my_data_size = my_array.size();
    int total_data_size = 0;
    bodo::vector<T> all_data_vec;

    // no need to gather data if only 1 processor
    if (n_pes == 1) {
        // In place modifies the array
        std::nth_element(my_array.begin(), my_array.begin() + k,
                         my_array.end());
        res = my_array[k];
        return res;
    }

    // gather the data sizes
    int *rcounts = new int[n_pes];
    int *displs = new int[n_pes];
    CHECK_MPI(MPI_Gather(&my_data_size, 1, MPI_INT, rcounts, 1, MPI_INT, root,
                         MPI_COMM_WORLD),
              "small_get_nth_parallel: MPI error on MPI_Gather:");
    // calculate size and displacements on root
    if (myrank == root) {
        for (int i = 0; i < n_pes; i++) {
            // printf("rc %d\n", rcounts[i]);
            displs[i] = total_data_size;
            total_data_size += rcounts[i];
        }
        // printf("total small data size: %d\n", total_data_size);
        all_data_vec.resize(total_data_size);
    }
    // gather data
    CHECK_MPI(
        MPI_Gatherv(my_array.data(), my_data_size, mpi_typ, all_data_vec.data(),
                    rcounts, displs, mpi_typ, root, MPI_COMM_WORLD),
        "small_get_nth_parallel: MPI error on MPI_Gatherv:");
    // get nth element on root
    if (myrank == root) {
        std::nth_element(all_data_vec.begin(), all_data_vec.begin() + k,
                         all_data_vec.end());
        res = all_data_vec[k];
    }
    CHECK_MPI(MPI_Bcast(&res, 1, mpi_typ, root, MPI_COMM_WORLD),
              "small_get_nth_parallel: MPI error on MPI_Bcast:");
    delete[] rcounts;
    delete[] displs;
    return res;
}

struct local_global_stat_nan {
    int64_t glob_nb_ok;
    int64_t glob_nb_miss;
    int64_t loc_nb_ok;
    int64_t loc_nb_miss;
};

/** Computation of the n-th entry
    @param my_array: the local list of non-nan entries (in output)
    @param arr: the array containing the data
    @param e_stat: the local/global statistical information
    ---
    If there is no nan entries, then the std::vector can be built directly.
    Note however that a copy happens anyway. This copy needs to happen, that is
    a pointer movement would not be adequate. The reason is that the
   std::nth_element changes the array put in argument.
    ---
    Depending on numpy/nullable-int-bool case a different also is used.
    This is only for non-decimal arrays.
 */
template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires(!decimal<DType>)
inline void collecting_non_nan_entries(bodo::vector<T> &my_array,
                                       std::shared_ptr<array_info> arr,
                                       local_global_stat_nan const &e_stat) {
    if (e_stat.loc_nb_miss == 0) {
        // Remark: The data is indeed copied in that case as well.
        T *data = (T *)arr->data1();
        my_array = bodo::vector<T>(data, data + e_stat.loc_nb_ok);
    } else {
        if (arr->arr_type == bodo_array_type::NUMPY) {
            for (size_t i_row = 0; i_row < arr->length; i_row++) {
                T eVal = arr->at<T, bodo_array_type::NUMPY>(i_row);
                bool isna = isnan_alltype<T, DType>(eVal);
                if (!isna) {
                    my_array.emplace_back(eVal);
                }
            }
        }
        if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            const uint8_t *null_bitmask =
                (uint8_t *)
                    arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>();
            for (size_t i_row = 0; i_row < arr->length; i_row++) {
                if (GetBit(null_bitmask, i_row)) {
                    T eVal =
                        arr->at<T, bodo_array_type::NULLABLE_INT_BOOL>(i_row);
                    my_array.emplace_back(eVal);
                }
            }
        }
    }
}

/** Computation of the n-th entry
    @param my_array: the local list of non-nan entries (in output)
    @param arr: the array containing the data
    @param e_stat: the local/global statistical information
    ---
    This code is for the case of decimal arrays and it returns an array of
   doubles. We could formally return a median that is a decimal from a decimal
   input. But: 1) This would be more complex code 2) Pandas uses double just as
   well.
 */
template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires decimal<DType>
inline void collecting_non_nan_entries(bodo::vector<T> &my_array,
                                       std::shared_ptr<array_info> arr,
                                       local_global_stat_nan const &e_stat) {
    const uint8_t *null_bitmask = (uint8_t *)arr->null_bitmask();
    for (size_t i_row = 0; i_row < arr->length; i_row++) {
        if (GetBit(null_bitmask, i_row)) {
            __int128_t eVal = arr->at<__int128_t>(i_row);
            double eVal_d = decimal_to_double(eVal);
            my_array.emplace_back(eVal_d);
        }
    }
}

/** Computation of the n-th entry
    @param my_array: the local list of non-nan entries
    @param k: the index that we are looking for.
    @param parallel: whether to use parallel or not
    @output: the k-th entry in the list
    ---
    Two algorithms depending on serial or not.
 */
template <typename T, int dtype, typename Alloc>
T get_nth(std::vector<T, Alloc> my_array, int64_t k, bool parallel) {
    if (parallel) {
        int myrank, n_pes;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        return get_nth_parallel(my_array, k, myrank, n_pes, dtype);
    } else {
        std::nth_element(my_array.begin(), my_array.begin() + k,
                         my_array.end());
        return my_array[k];
    }
}

/** Computation of local number of non-nan/nan entries
    @param array: the local list of non-nan entries
    @output pair: the number of non-nan/nan entries.
    ---
    This is templatized code. Two cases to consider:
    1) The nullable_int_bool case (which covers decimal)
    2) the case of integer/float is covered by the isnan_alltype function
 */
template <typename T, Bodo_CTypes::CTypeEnum DType>
std::pair<int64_t, int64_t> nb_entries_local(std::shared_ptr<array_info> arr) {
    int64_t nb_ok = 0, nb_miss = 0;
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        const uint8_t *null_bitmask =
            (uint8_t *)arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>();
        for (size_t i_row = 0; i_row < arr->length; i_row++) {
            if (GetBit(null_bitmask, i_row)) {
                nb_ok++;
            } else {
                nb_miss++;
            }
        }
    }
    if (arr->arr_type == bodo_array_type::NUMPY) {
        for (size_t i_row = 0; i_row < arr->length; i_row++) {
            T eVal = arr->at<T, bodo_array_type::NUMPY>(i_row);
            bool isna = isnan_alltype<T, DType>(eVal);
            if (isna) {
                nb_miss++;
            } else {
                nb_ok++;
            }
        }
    }
    return {nb_ok, nb_miss};
}

/** Computation of global number of non-nan/nan entries
    @param array: the local list of non-nan entries
    @param parallel: whether to use the parallel algorithm
    @output local_global_stat_nan: the statistical information coming from the
   computation
    ---
    First compute the local statistical information.
    Then use MPI_Allreduce to agglomerate them
 */
template <typename T, Bodo_CTypes::CTypeEnum DType>
local_global_stat_nan nb_entries_global(std::shared_ptr<array_info> arr,
                                        bool parallel) {
    std::pair<int64_t, int64_t> pair = nb_entries_local<T, DType>(arr);
    if (!parallel) {
        return {.glob_nb_ok = pair.first,
                .glob_nb_miss = pair.second,
                .loc_nb_ok = pair.first,
                .loc_nb_miss = pair.second};
    }
    int64_t loc_nb_ok = pair.first, loc_nb_miss = pair.second, glob_nb_ok = 0,
            glob_nb_miss = 0;
    CHECK_MPI(MPI_Allreduce(&loc_nb_ok, &glob_nb_ok, 1, MPI_LONG_LONG_INT,
                            MPI_SUM, MPI_COMM_WORLD),
              "nb_entries_global: MPI error on MPI_Allreduce:");
    CHECK_MPI(MPI_Allreduce(&loc_nb_miss, &glob_nb_miss, 1, MPI_LONG_LONG_INT,
                            MPI_SUM, MPI_COMM_WORLD),
              "nb_entries_global: MPI error on MPI_Allreduce:");
    return {.glob_nb_ok = glob_nb_ok,
            .glob_nb_miss = glob_nb_miss,
            .loc_nb_ok = loc_nb_ok,
            .loc_nb_miss = loc_nb_miss};
}

/** Effective computation of the median
    @param my_array: the local list of non-nan entries
    @param parallel: whether to use the parallel algorithm
    @param glob_nb_ok: the total number of non-nan entries
    @output res: the median as output
    ---
    If the total number of entries is odd then return the middle.
    If the total number of entries is even then return the average of two middle
   entries
 */
template <typename T, int dtype, typename Alloc>
void median_series_computation_eff(double *res, std::vector<T, Alloc> my_array,
                                   bool parallel, int64_t glob_nb_ok) {
    if (glob_nb_ok % 2 == 1) {
        int kMid = glob_nb_ok / 2;
        T eVal = get_nth<T, dtype, Alloc>(my_array, kMid, parallel);
        *res = double(eVal);
    } else {
        int kMid1 = glob_nb_ok / 2;
        int kMid2 = kMid1 - 1;
        T eVal1 = get_nth<T, dtype, Alloc>(my_array, kMid1, parallel);
        T eVal2 = get_nth<T, dtype, Alloc>(my_array, kMid2, parallel);
        *res = (double(eVal1) + double(eVal2)) / 2;
    }
}

/** Compute the median of the series.
    @param arr: The array_info used for the computation
    @param parallel: whether to use the parallel algorithm
    @param skipna: whether to skip the nan entries or not.
    @output res: the median as output
    ---
    The template argument 1 is the datatype and dtype is the corresponding
   bodo_common type.
    ---
    First step is determination of total number of non-nan and nan entries. In
   two cases this suffices to conclude. Second step is to determine the non-nan
   entries (templatized code for decimal/non-decimal) Third step is to compute
   the median from the list of non-nan entries with two cases to consider.
 */
template <typename T, Bodo_CTypes::CTypeEnum DType>
void median_series_computation_T(double *res, std::shared_ptr<array_info> arr,
                                 bool parallel, bool skip_na_data) {
    local_global_stat_nan e_stat = nb_entries_global<T, DType>(arr, parallel);
    if ((e_stat.glob_nb_miss > 0 && !skip_na_data) || e_stat.glob_nb_ok == 0) {
        *res = std::nan("");
        return;
    }
    bodo::vector<T> my_array;
    collecting_non_nan_entries<T, DType>(my_array, arr, e_stat);
    int64_t glob_nb_ok = e_stat.glob_nb_ok;
    if (DType == Bodo_CTypes::DECIMAL) {
        median_series_computation_eff<T, Bodo_CTypes::FLOAT64,
                                      bodo::DefaultSTLBufferPoolAllocator<T>>(
            res, my_array, parallel, glob_nb_ok);
    } else {
        median_series_computation_eff<T, DType,
                                      bodo::DefaultSTLBufferPoolAllocator<T>>(
            res, my_array, parallel, glob_nb_ok);
    }
}

/** Compute the median of the series.
    @param arr: The array_info used for the computation
    @param parallel: whether to use the parallel algorithm
    @param skipna: whether to skip the nan entries or not.
    @output res: the median as output
    ---
    According to the data type we trigger a different templatized function.
 */
void median_series_computation_py_entry(double *res, array_info *p_arr,
                                        bool parallel, bool skip_na_data) {
    try {
        std::shared_ptr<array_info> arr(p_arr);
        Bodo_CTypes::CTypeEnum dtype = arr->dtype;
        switch (dtype) {
            case Bodo_CTypes::INT8:
                return median_series_computation_T<int8_t, Bodo_CTypes::INT8>(
                    res, arr, parallel, skip_na_data);
            case Bodo_CTypes::UINT8:
                return median_series_computation_T<uint8_t, Bodo_CTypes::UINT8>(
                    res, arr, parallel, skip_na_data);
            case Bodo_CTypes::INT16:
                return median_series_computation_T<int16_t, Bodo_CTypes::INT16>(
                    res, arr, parallel, skip_na_data);
            case Bodo_CTypes::UINT16:
                return median_series_computation_T<uint16_t,
                                                   Bodo_CTypes::UINT16>(
                    res, arr, parallel, skip_na_data);
            case Bodo_CTypes::INT32:
                return median_series_computation_T<int32_t, Bodo_CTypes::INT32>(
                    res, arr, parallel, skip_na_data);
            case Bodo_CTypes::UINT32:
                return median_series_computation_T<uint32_t,
                                                   Bodo_CTypes::UINT32>(
                    res, arr, parallel, skip_na_data);
            case Bodo_CTypes::INT64:
                return median_series_computation_T<int64_t, Bodo_CTypes::INT64>(
                    res, arr, parallel, skip_na_data);
            case Bodo_CTypes::UINT64:
                return median_series_computation_T<uint64_t,
                                                   Bodo_CTypes::UINT64>(
                    res, arr, parallel, skip_na_data);
            case Bodo_CTypes::FLOAT32:
                return median_series_computation_T<float, Bodo_CTypes::FLOAT32>(
                    res, arr, parallel, skip_na_data);
            case Bodo_CTypes::FLOAT64:
                return median_series_computation_T<double,
                                                   Bodo_CTypes::FLOAT64>(
                    res, arr, parallel, skip_na_data);
            case Bodo_CTypes::DECIMAL:
                // We choose to return double in case of decimal (while a
                // decimal return is conceptually feasible) because it would
                // require more work and pandas uses double in that case as
                // well.
                return median_series_computation_T<double,
                                                   Bodo_CTypes::DECIMAL>(
                    res, arr, parallel, skip_na_data);
            default:
                throw std::runtime_error(
                    "_quantile_alg.cpp::median_series_computation: type not "
                    "supported by median_series_computation");
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

/* The computation of several operations requires one to know the rows
   which are just next to the rows currently on the nodes. In numerical
   analysis those are called "ghost nodes" and so we call them "ghost rows".
   Where we depart from numerical analysis is that in this concept all the
   nodes and ghost nodes are stored in one single array with exchanges occurring
   only for the ghost nodes. Here we cannot do that unless changing the
   structure of bodo.
   So, what we return is an array that contains only the ghost rows following
   the point. The algorithm is waterproof in the sense that any depth is
   possible. We may need to use one level of next entries or more but the
   algorithm will handle that.
   ----
   Limitations:
   ---We do the operation only for numerical NUMPY values.
   ---We only return the next rows. We could also return the previous ones.
*/
std::shared_ptr<array_info> compute_ghost_rows(std::shared_ptr<array_info> arr,
                                               uint64_t const &level_next) {
    int myrank, n_pes;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    // For a nodes of rank r the ListPrevSizes is the list of number of rows
    // before the rank r, say ranks r-1, r-2, ...., r-k
    // The ListNextSizes, the ranks are r+1, r+2, ...., r+k
    // We do not share all the number of rows. Only the ones that are just
    // previous and next. We stop when ListPrevSizes[r-1] + ... +
    // ListPrevSizes[r-k] <= lag on all nodes. For the nodes of rank 0 the
    // vectors are smaller of course.
    bodo::vector<size_t> ListPrevSizes, ListNextSizes;
    size_t loc_nrows = arr->length;
    MPI_Datatype mpi_row_typ = MPI_LONG_LONG_INT;
    using T_row_typ = int64_t;
    int k = 0;
    while (true) {
        k++;
        std::vector<MPI_Request> ListReq;
        //
        std::vector<T_row_typ> V(4);
        int idx = 0;
        int idx_recv_next = -1, idx_recv_prev = -1;
        // Sending number of rows to myrank - k
        if (myrank >= k) {
            MPI_Request mpi_send_next;
            int pe = myrank - k;
            V[idx] = loc_nrows;
            T_row_typ *ptr_T = V.data() + idx;
            int tag = 10000 + pe;
            CHECK_MPI(MPI_Isend((void *)ptr_T, 1, mpi_row_typ, pe, tag,
                                MPI_COMM_WORLD, &mpi_send_next),
                      "compute_ghost_rows: MPI error on MPI_Isend:");
            ListReq.push_back(mpi_send_next);
            idx++;
        }
        // Sending number of rows to myrank + k
        if (myrank < n_pes - k) {
            MPI_Request mpi_send_prev;
            int pe = myrank + k;
            int tag = 20000 + pe;
            V[idx] = loc_nrows;
            T_row_typ *ptr_T = V.data() + idx;
            CHECK_MPI(MPI_Isend((void *)ptr_T, 1, mpi_row_typ, pe, tag,
                                MPI_COMM_WORLD, &mpi_send_prev),
                      "compute_ghost_rows: MPI error on MPI_Isend:");
            ListReq.push_back(mpi_send_prev);
            idx++;
        }
        // Receiving number of rows from myrank + k
        if (myrank < n_pes - k) {
            MPI_Request mpi_recv_next;
            int pe = myrank + k;
            int tag = 10000 + myrank;
            T_row_typ *ptr_T = V.data() + idx;
            CHECK_MPI(MPI_Irecv((void *)ptr_T, 1, mpi_row_typ, pe, tag,
                                MPI_COMM_WORLD, &mpi_recv_next),
                      "compute_ghost_rows: MPI error on MPI_Irecv:");
            idx_recv_next = idx;
            ListReq.push_back(mpi_recv_next);
            idx++;
        }
        // Receiving number of rows from myrank - k
        if (myrank >= k) {
            MPI_Request mpi_recv_prev;
            int pe = myrank - k;
            int tag = 20000 + myrank;
            T_row_typ *ptr_T = V.data() + idx;
            CHECK_MPI(MPI_Irecv((void *)ptr_T, 1, mpi_row_typ, pe, tag,
                                MPI_COMM_WORLD, &mpi_recv_prev),
                      "compute_ghost_rows: MPI error on MPI_Irecv:");
            idx_recv_prev = idx;
            ListReq.push_back(mpi_recv_prev);
            idx++;
        }
        // Now doing the exchanges.
        if (ListReq.size() > 0) {
            CHECK_MPI(MPI_Waitall(ListReq.size(), ListReq.data(),
                                  MPI_STATUSES_IGNORE),
                      "compute_ghost_rows: MPI error on MPI_Waitall:");
            // Putting the values where they should be.
            if (idx_recv_prev != -1) {
                ListPrevSizes.push_back(V[idx_recv_prev]);
            }
            if (idx_recv_next != -1) {
                ListNextSizes.push_back(V[idx_recv_next]);
            }
        }
        size_t sumprev = std::accumulate(ListPrevSizes.begin(),
                                         ListPrevSizes.end(), size_t(0));
        size_t sumnext = std::accumulate(ListNextSizes.begin(),
                                         ListNextSizes.end(), size_t(0));
        bool test_final;
        // If myrank - k > 0 we can continue.
        // If myrank + k < n_pes - 1 we can continue
        // If myrank - k <= 0 no way forward
        // If myrank + k >= n_pes - 1 no way forward
        if (myrank <= k && myrank + k >= n_pes - 1) {
            test_final = true;
        } else {
            test_final = (sumprev >= level_next) && (sumnext >= level_next);
        }
        int value_tot, value = !test_final;
        // value=0 means all is ok. All nodes should be ok for the loop to
        // terminate
        CHECK_MPI(MPI_Allreduce(&value, &value_tot, 1, MPI_INT, MPI_SUM,
                                MPI_COMM_WORLD),
                  "compute_ghost_rows: MPI error on MPI_Allreduce:");
        if (value_tot == 0) {
            break;
        }
    }
    // Now agglomerating the data
    int64_t n_next = ListNextSizes.size();
    int64_t n_prev = ListPrevSizes.size();
    size_t sumnext =
        std::accumulate(ListNextSizes.begin(), ListNextSizes.end(), size_t(0));
    uint64_t ghost_length = std::min(sumnext, size_t(level_next));
    std::shared_ptr<array_info> ghost_arr =
        alloc_numpy(ghost_length, arr->dtype);
    uint64_t siztype = numpy_item_size[arr->dtype];
    MPI_Datatype mpi_typ = get_MPI_typ(arr->dtype);
    uint64_t pos_index = 0;
    std::vector<MPI_Request> ListReq;
    for (int64_t i_next = 0; i_next < n_next; i_next++) {
        size_t siz = ListNextSizes[i_next];
        size_t siz_recv;
        if (siz + pos_index <= ghost_length) {
            siz_recv = siz;
        } else {
            siz_recv = ghost_length - pos_index;
        }
        if (siz_recv > 0) {
            MPI_Request mpi_recv;
            char *ptr_recv = ghost_arr->data1<bodo_array_type::NUMPY>() +
                             siztype * pos_index;
            int tag = 2046 + n_pes * i_next + myrank;
            int pe = myrank + 1 + i_next;
            CHECK_MPI(MPI_Irecv((void *)ptr_recv, siz_recv, mpi_typ, pe, tag,
                                MPI_COMM_WORLD, &mpi_recv),
                      "compute_ghost_rows: MPI error on MPI_Irecv:");
            ListReq.push_back(mpi_recv);
            pos_index += siz_recv;
        }
    }
    uint64_t pos_already = 0;
    char *arr_data1 = arr->data1();
    for (int64_t i_prev = 0; i_prev < n_prev; i_prev++) {
        size_t siz_prev = ListPrevSizes[i_prev];
        size_t siz_send;
        if (loc_nrows + pos_already <= level_next) {
            siz_send = loc_nrows;
        } else {
            if (level_next >= pos_already) {
                siz_send = level_next - pos_already;
            } else {
                siz_send = 0;
            }
        }
        if (siz_send > 0) {
            MPI_Request mpi_send;
            char *ptr_send = arr_data1;
            int pe = myrank - 1 - i_prev;
            int tag = 2046 + n_pes * i_prev + pe;
            CHECK_MPI(MPI_Isend((void *)ptr_send, siz_send, mpi_typ, pe, tag,
                                MPI_COMM_WORLD, &mpi_send),
                      "compute_ghost_rows: MPI error on MPI_Isend:");
            ListReq.push_back(mpi_send);
            pos_already += siz_prev;
        }
    }
    if (ListReq.size() > 0) {
        CHECK_MPI(
            MPI_Waitall(ListReq.size(), ListReq.data(), MPI_STATUSES_IGNORE),
            "compute_ghost_rows: MPI error on MPI_Waitall:");
    }
    return ghost_arr;
}

/* Compute whether the array is monotonic or not
   @param res : the return value 1 for monotonic and 0 otherwise
   @param arr : the input array
   @param inc_dec : 1 for testing is_monotonic_increasing / 2 for
   is_monotonic_decreasing
 */
void compute_series_monotonicity_py_entry(double *res, array_info *p_arr,
                                          int64_t inc_dec, bool is_parallel) {
    try {
        std::shared_ptr<array_info> arr(p_arr);
        int64_t n_rows = arr->length;
        uint64_t siztype = numpy_item_size[arr->dtype];
        // First checking monotonicity locally
        auto do_local_computation = [&]() -> int {
            char *arr_data1 = arr->data1();
            for (int64_t i_row = 0; i_row < n_rows - 1; i_row++) {
                char *ptr1 = arr_data1 + siztype * i_row;
                char *ptr2 = arr_data1 + siztype * (i_row + 1);
                bool na_position = false;
                int test =
                    NumericComparison(arr->dtype, ptr1, ptr2, na_position);
                if (test == -1) {  // this corresponds to *ptr1 > *ptr2
                    if (inc_dec == 1) {
                        return 1;  // We reach a contradiction
                    }
                }
                if (test == 1) {  // this corresponds to *ptr1 < *ptr2
                    if (inc_dec == 2) {
                        return 1;  // We reach a contradiction
                    }
                }
            }
            return 0;
        };
        int value = do_local_computation();
        if (!is_parallel) {
            if (value > 0) {
                *res = 0.0;
            } else {
                *res = 1.0;
            }
            return;
        }
        int value_tot;
        CHECK_MPI(MPI_Allreduce(&value, &value_tot, 1, MPI_INT, MPI_SUM,
                                MPI_COMM_WORLD),
                  "compute_series_monotonicity_py_entry: MPI error on "
                  "MPI_Allreduce:");
        if (value_tot > 0) {  // At least one node find a contradiction locally.
                              // Enough to conclude
            *res = 0.0;
            return;
        }
        // We need to compute the ghost rows to conclude
        std::shared_ptr<array_info> ghost_arr = compute_ghost_rows(arr, 1);
        int value_glob_tot, value_glob = 0;
        if (ghost_arr->length > 0 &&
            n_rows >
                0) {  // It will be empty on the last node and maybe others.
            char *ptr1 = arr->data1() + siztype * (n_rows - 1);
            char *ptr2 = ghost_arr->data1();
            bool na_position = false;
            int test = NumericComparison(arr->dtype, ptr1, ptr2, na_position);
            if (test == -1) {  // this corresponds to *ptr1 > *ptr2
                if (inc_dec == 1) {
                    value_glob = 1;  // We reach a contradiction
                }
            }
            if (test == 1) {  // this corresponds to *ptr1 < *ptr2
                if (inc_dec == 2) {
                    value_glob = 1;  // We reach a contradiction
                }
            }
        }
        CHECK_MPI(MPI_Allreduce(&value_glob, &value_glob_tot, 1, MPI_INT,
                                MPI_SUM, MPI_COMM_WORLD),
                  "compute_series_monotonicity_py_entry: MPI error on "
                  "MPI_Allreduce:");
        if (value_glob_tot > 0) {
            *res = 0.0;
        } else {
            *res = 1.0;
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

void autocorr_series_computation_py_entry(double *res, array_info *p_arr,
                                          int64_t lag, bool is_parallel) {
    try {
        std::shared_ptr<array_info> arr(p_arr);
        uint64_t n_rows = arr->length;
        uint64_t siztype = numpy_item_size[arr->dtype];
        if (!is_parallel) {
            if (uint64_t(lag) >= n_rows - 1) {
                *res = std::nan("1.0");
                return;
            }
            double sum1 = 0, sum2 = 0, sum12 = 0, sum11 = 0, sum22 = 0;
            const char *arr_data1 = arr->data1();
            for (uint64_t i_row = 0; i_row < n_rows - lag; i_row++) {
                const char *ptr1 = arr_data1 + siztype * i_row;
                const char *ptr2 = arr_data1 + siztype * (i_row + lag);
                double val1 = GetDoubleEntry(arr->dtype, ptr1);
                double val2 = GetDoubleEntry(arr->dtype, ptr2);
                sum1 += val1;
                sum2 += val2;
                sum12 += val1 * val2;
                sum11 += val1 * val1;
                sum22 += val2 * val2;
            }
            double fac = double(1) / double(n_rows - lag);
            double avg1 = sum1 * fac;
            double avg2 = sum2 * fac;
            double avg11 = sum11 * fac;
            double avg12 = sum12 * fac;
            double avg22 = sum22 * fac;
            double scal = avg12 - avg1 * avg2;
            double norm1 = sqrt(avg11 - avg1 * avg1);
            double norm2 = sqrt(avg22 - avg2 * avg2);
            double autocorr = scal / (norm1 * norm2);
            *res = autocorr;
            return;
        }
        uint64_t n_rows_tot;
        CHECK_MPI(MPI_Allreduce(&n_rows, &n_rows_tot, 1, MPI_UNSIGNED_LONG_LONG,
                                MPI_SUM, MPI_COMM_WORLD),
                  "autocorr_series_computation_py_entry: MPI error on "
                  "MPI_Allreduce:");
        if (uint64_t(lag) >= n_rows_tot - 1) {
            *res = std::nan("1.0");
            return;
        }
        std::shared_ptr<array_info> ghost_arr = compute_ghost_rows(arr, lag);
        uint64_t ghost_siz = ghost_arr->length;
        std::vector<double> V(5, 0);
        double &sum1 = V[0];
        double &sum2 = V[1];
        double &sum11 = V[2];
        double &sum12 = V[3];
        double &sum22 = V[4];
        if (n_rows + ghost_siz >= uint64_t(lag)) {
            uint64_t n_rows_cons =
                n_rows + ghost_siz -
                lag;  // the ghost may provide the additional rows or may not
            const char *arr_data1 = arr->data1();
            const char *ghost_arr_data1 = ghost_arr->data1();
            for (uint64_t i_row = 0; i_row < n_rows_cons; i_row++) {
                const char *ptr1 = arr_data1 + siztype * i_row;
                double val1 = GetDoubleEntry(arr->dtype, ptr1);
                const char *ptr2 =
                    (i_row < n_rows - lag)
                        ? (arr_data1 + siztype * (i_row + lag))
                        : (ghost_arr_data1 + siztype * (i_row - n_rows + lag));
                double val2 = GetDoubleEntry(arr->dtype, ptr2);
                sum1 += val1;
                sum2 += val2;
                sum11 += val1 * val1;
                sum12 += val1 * val2;
                sum22 += val2 * val2;
            }
        }
        std::vector<double> Vtot(5);
        CHECK_MPI(MPI_Allreduce(V.data(), Vtot.data(), 5, MPI_DOUBLE, MPI_SUM,
                                MPI_COMM_WORLD),
                  "autocorr_series_computation_py_entry: MPI error on "
                  "MPI_Allreduce:");
        double fac = double(1) / double(n_rows_tot - lag);
        double avg1 = Vtot[0] * fac;
        double avg2 = Vtot[1] * fac;
        double avg11 = Vtot[2] * fac;
        double avg12 = Vtot[3] * fac;
        double avg22 = Vtot[4] * fac;
        double scal = avg12 - avg1 * avg2;
        double norm1 = sqrt(avg11 - avg1 * avg1);
        double norm2 = sqrt(avg22 - avg2 * avg2);
        double autocorr = scal / (norm1 * norm2);
        *res = autocorr;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

/** Helper function for approx_percentile that does the main calculation.
 *
 * TODO: make this function able to take in multiple percentile values and
 * calculate all of them at once by re-using the same t-digest intermediary
 * calculations.
 *
 * @param data the bodo array whose percentile is being approximated
 * @param size the size of the array
 * @param percentile the percentile value being sought (e.g. 0.5=median)
 * @param parallel whether this is occurring across multiple ranks
 * @return the approximate percentile value as a double
 */
template <typename T, Bodo_CTypes::CTypeEnum DType>
double approx_percentile_T(std::shared_ptr<array_info> arr, double percentile,
                           bool parallel) {
    /* TODO: investigate adjusting the hyperparameters for speed, memory and
     * accuracy. Each rank likely needs to have the same values.
     */
    TDigest td(/*delta*/ 100, /*buffer_size*/ 500);
    tracing::Event ev("approx_percentile_T", parallel);
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        const uint8_t *null_bitmask =
            (uint8_t *)arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>();
        for (uint64_t i = 0; i < arr->length; i++) {
            if (GetBit(null_bitmask, i)) {
                double val_double = to_double<T, DType>(
                    getv<T, bodo_array_type::NULLABLE_INT_BOOL>(arr, i));
                td.NanAdd(val_double);
            }
        }
    } else {
        for (uint64_t i = 0; i < arr->length; i++) {
            double val_double = to_double<T, DType>(getv<T>(arr, i));
            td.NanAdd(val_double);
        }
    }
    if (parallel) {
        td.MPI_Merge();
    }
    return td.Quantile(percentile, parallel);
}

/** Calculates the approximate percentile of an array using PyArrow's t-digest
 * implementation.
 *
 * @param arr_raw the bodo array whose percentile is being approximated
 * @param percentile the percentile value being sought (e.g. 0.5=median)
 * @param parallel whether the operation is being done in parallel or not
 * @return the approximate percentile value as a double
 */
double approx_percentile_py_entrypt(array_info *arr_raw, double percentile,
                                    bool parallel) {
    try {
        std::shared_ptr<array_info> arr(arr_raw);
        switch (arr->dtype) {
            case Bodo_CTypes::INT8:
                return approx_percentile_T<int8_t, Bodo_CTypes::INT8>(
                    arr, percentile, parallel);
            case Bodo_CTypes::_BOOL:
            case Bodo_CTypes::UINT8:
                return approx_percentile_T<uint8_t, Bodo_CTypes::UINT8>(
                    arr, percentile, parallel);
            case Bodo_CTypes::INT16:
                return approx_percentile_T<int16_t, Bodo_CTypes::INT16>(
                    arr, percentile, parallel);
            case Bodo_CTypes::UINT16:
                return approx_percentile_T<uint16_t, Bodo_CTypes::UINT16>(
                    arr, percentile, parallel);
            case Bodo_CTypes::INT32:
                return approx_percentile_T<int32_t, Bodo_CTypes::INT32>(
                    arr, percentile, parallel);
            case Bodo_CTypes::UINT32:
                return approx_percentile_T<uint32_t, Bodo_CTypes::UINT32>(
                    arr, percentile, parallel);
            case Bodo_CTypes::INT64:
                return approx_percentile_T<int64_t, Bodo_CTypes::INT64>(
                    arr, percentile, parallel);
            case Bodo_CTypes::UINT64:
                return approx_percentile_T<uint64_t, Bodo_CTypes::UINT64>(
                    arr, percentile, parallel);
            case Bodo_CTypes::FLOAT32:
                return approx_percentile_T<float, Bodo_CTypes::FLOAT32>(
                    arr, percentile, parallel);
            case Bodo_CTypes::FLOAT64:
                return approx_percentile_T<double, Bodo_CTypes::FLOAT64>(
                    arr, percentile, parallel);
            default:
                throw std::runtime_error(
                    "_quantile_alg.cpp::approx_percentile_py_entrypt:"
                    " unknown "
                    "approx_percentile data type");
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return 0.0;
    }
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
double percentile_util(const std::unique_ptr<array_info> &arr,
                       double percentile, bool interpolate, bool parallel) {
    using T = typename dtype_to_type<DType>::type;
    // Place all of the non-null entries in a vector as doubles
    std::vector<double> vect;
    for (uint64_t i = 0; i < arr->length; i++) {
        if (non_null_at<ArrType, T, DType>(*arr, i)) {
            T val = get_arr_item<ArrType, T, DType>(*arr, i);
            vect.emplace_back(to_double<T, DType>(val));
        }
    }

    int64_t len = vect.size();

    int myrank, n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // Calculate the total number of elements across all ranks.
    int64_t total_size = len;
    if (parallel) {
        CHECK_MPI(MPI_Allreduce(&len, &total_size, 1, MPI_LONG_LONG_INT,
                                MPI_SUM, MPI_COMM_WORLD),
                  "percentile_util: MPI error on MPI_Allreduce:");
    }

    // If there were no non-null entries, output null
    // (using NaN as a sentinel).
    if (total_size == 0) {
        return nan_val<double, Bodo_CTypes::FLOAT64>();
    }

    if (interpolate) {
        // Algorithm for PERCENTILE_CONT
        double k_approx = (total_size - 1) * percentile;
        int64_t k_exact = (int64_t)k_approx;
        if (k_approx == k_exact) {
            // If a value lies exactly at the percentile value being sought,
            // return that value.
            return get_nth_q(vect, len, k_exact, Bodo_CTypes::FLOAT64, myrank,
                             n_pes, parallel);
        } else {
            // Otherwise, find the nearest value above
            // and below, then linearly interpolate between them.
            double v1 = get_nth_q(vect, len, k_exact, Bodo_CTypes::FLOAT64,
                                  myrank, n_pes, parallel);
            double v2 = get_nth_q(vect, len, k_exact + 1, Bodo_CTypes::FLOAT64,
                                  myrank, n_pes, parallel);
            return v1 + (k_approx - k_exact) * (v2 - v1);
        }
    } else {
        // Algorithm for PERCENTILE_DISC
        double k_approx = total_size * percentile;
        int64_t k_exact = (int64_t)k_approx;
        // The following formula will choose the ordinal formula
        // corresponding to the first location whose cumulative distribution
        // is >= percentile.
        int64_t k_select = (k_approx == k_exact) ? (k_exact - 1) : k_exact;
        if (k_select < 0) {
            k_select = 0;
        }
        return get_nth_q(vect, len, k_select, Bodo_CTypes::FLOAT64, myrank,
                         n_pes, parallel);
    }
}

template <bodo_array_type::arr_type_enum ArrType>
double percentile_dtype_helper(const std::unique_ptr<array_info> &arr,
                               double percentile, bool interpolate,
                               bool parallel) {
    switch (arr->dtype) {
        case Bodo_CTypes::INT8: {
            return percentile_util<ArrType, Bodo_CTypes::INT8>(
                arr, percentile, interpolate, parallel);
        }
        case Bodo_CTypes::UINT8: {
            return percentile_util<ArrType, Bodo_CTypes::UINT8>(
                arr, percentile, interpolate, parallel);
        }
        case Bodo_CTypes::INT16: {
            return percentile_util<ArrType, Bodo_CTypes::INT16>(
                arr, percentile, interpolate, parallel);
        }
        case Bodo_CTypes::UINT16: {
            return percentile_util<ArrType, Bodo_CTypes::UINT16>(
                arr, percentile, interpolate, parallel);
        }
        case Bodo_CTypes::INT32: {
            return percentile_util<ArrType, Bodo_CTypes::INT32>(
                arr, percentile, interpolate, parallel);
        }
        case Bodo_CTypes::UINT32: {
            return percentile_util<ArrType, Bodo_CTypes::UINT32>(
                arr, percentile, interpolate, parallel);
        }
        case Bodo_CTypes::INT64: {
            return percentile_util<ArrType, Bodo_CTypes::INT64>(
                arr, percentile, interpolate, parallel);
        }
        case Bodo_CTypes::UINT64: {
            return percentile_util<ArrType, Bodo_CTypes::UINT64>(
                arr, percentile, interpolate, parallel);
        }
        case Bodo_CTypes::FLOAT32: {
            return percentile_util<ArrType, Bodo_CTypes::FLOAT32>(
                arr, percentile, interpolate, parallel);
        }
        case Bodo_CTypes::FLOAT64: {
            return percentile_util<ArrType, Bodo_CTypes::FLOAT64>(
                arr, percentile, interpolate, parallel);
        }
        default: {
            throw std::runtime_error("Unsupported DType for percentile: " +
                                     GetDtype_as_string(arr->dtype));
        }
    }
}

double percentile_py_entrypt(array_info *arr_raw, double percentile,
                             bool interpolate, bool parallel) {
    std::unique_ptr<array_info> arr(arr_raw);
    double res;
    switch (arr->arr_type) {
        case bodo_array_type::NULLABLE_INT_BOOL: {
            res = percentile_dtype_helper<bodo_array_type::NULLABLE_INT_BOOL>(
                arr, percentile, interpolate, parallel);
            break;
        }
        case bodo_array_type::NUMPY: {
            res = percentile_dtype_helper<bodo_array_type::NUMPY>(
                arr, percentile, interpolate, parallel);
            break;
        }
        default: {
            throw std::runtime_error("Unsupported ArrayType for percentile: " +
                                     GetArrType_as_string(arr->arr_type));
        }
    }
    if (parallel) {
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::FLOAT64);
        CHECK_MPI(MPI_Bcast(&res, 1, mpi_typ, 0, MPI_COMM_WORLD),
                  "percentile_py_entrypt: MPI error on MPI_Bcast:");
    }
    return res;
}
