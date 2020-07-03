// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>
#include "mpi.h"

#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"

#define root 0

MPI_Datatype decimal_mpi_type = MPI_DATATYPE_NULL;

template <class T>
std::pair<T, T> get_lower_upper_kth_parallel(std::vector<T> &my_array,
                                             int64_t total_size, int myrank,
                                             int n_pes, int64_t k,
                                             int type_enum);

template <class T>
T small_get_nth_parallel(std::vector<T> &my_array, int64_t total_size,
                         int myrank, int n_pes, int64_t k, int type_enum);

template <class T>
T get_nth_parallel(std::vector<T> &my_array, int64_t k, int myrank, int n_pes,
                   int type_enum);

double quantile_sequential(void *data, int64_t local_size, double quantile,
                           int type_enum);
double quantile_parallel(void *data, int64_t local_size, int64_t total_size,
                         double quantile, int type_enum);
template <class T>
double quantile_int(T *data, int64_t local_size, double at, int type_enum,
                    bool parallel);
template <class T>
double quantile_float(T *data, int64_t local_size, double quantile,
                      int type_enum, bool parallel);

double quantile_dispatch(void *data, int64_t local_size, double quantile,
                         double at, int type_enum, bool parallel);

/** Compute the median of the series.
    @param arr: The array_info used for the computation
    @param parallel: whether to use the parallel algorithm
    @param skipna: whether to skip the nan entries or not.
    @output res: the median as output
 */
void median_series_computation(double *res, array_info *arr, bool parallel,
                               bool skipna);

PyMODINIT_FUNC PyInit_quantile_alg(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "quantile_alg", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    PyObject_SetAttrString(m, "quantile_sequential",
                           PyLong_FromVoidPtr((void *)(&quantile_sequential)));
    PyObject_SetAttrString(m, "quantile_parallel",
                           PyLong_FromVoidPtr((void *)(&quantile_parallel)));
    PyObject_SetAttrString(
        m, "median_series_computation",
        PyLong_FromVoidPtr((void *)(&median_series_computation)));
    return m;
}

double quantile_sequential(void *data, int64_t local_size, double quantile,
                           int type_enum) {
    // return NA if no elements
    if (local_size == 0) {
        return std::nan("");
    }

    double at = quantile * (local_size - 1);
    return quantile_dispatch(data, local_size, quantile, at, type_enum, false);
}

double quantile_parallel(void *data, int64_t local_size, int64_t total_size,
                         double quantile, int type_enum) {
    int myrank, n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (total_size == 0)
        MPI_Allreduce(&local_size, &total_size, 1, MPI_LONG_LONG_INT, MPI_SUM,
                      MPI_COMM_WORLD);

    // return NA if no elements
    if (total_size == 0) {
        return std::nan("");
    }

    double at = quantile * (total_size - 1);
    return quantile_dispatch(data, local_size, quantile, at, type_enum, true);
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
            std::cerr << "unknown quantile data type"
                      << "\n";
    }

    return -1.0;
}

template <class T>
double get_nth_q(std::vector<T> &my_array, int64_t local_size, int64_t k,
                 int type_enum, int myrank, int n_pes, bool parallel) {
    // get nth element and store in res pointer
    // assuming NA values of floats are already removed
    T val;

    if (parallel) {
        val = get_nth_parallel(my_array, k, myrank, n_pes, type_enum);
    } else {
        std::nth_element(my_array.begin(), my_array.begin() + k,
                         my_array.end());
        val = my_array[k];
    }
    return (double)val;
}

template <class T>
double quantile_int(T *data, int64_t local_size, double at, int type_enum,
                    bool parallel) {
    int64_t k1 = (int64_t)at;
    int64_t k2 = k1 + 1;
    double fraction = at - (double)k1;
    std::vector<T> my_array(data, data + local_size);

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

template <class T>
double quantile_float(T *data, int64_t local_size, double quantile,
                      int type_enum, bool parallel) {
    int myrank, n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::vector<T> my_array(data, data + local_size);
    // delete NaNs
    my_array.erase(std::remove_if(std::begin(my_array), std::end(my_array),
                                  [](T d) { return std::isnan(d); }),
                   my_array.end());
    local_size = my_array.size();
    // recalculate total size since there could be NaNs
    int64_t total_size = local_size;
    if (parallel) {
        MPI_Allreduce(&local_size, &total_size, 1, MPI_LONG_LONG_INT, MPI_SUM,
                      MPI_COMM_WORLD);
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

template <class T>
T get_nth_parallel(std::vector<T> &my_array, int64_t k, int myrank, int n_pes,
                   int type_enum) {
    int64_t local_size = my_array.size();
    int64_t total_size;
    MPI_Allreduce(&local_size, &total_size, 1, MPI_LONG_LONG_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    // printf("total size: %ld k: %ld\n", total_size, k);
    int64_t threshold = (int64_t)pow(10.0, 7.0);  // 100 million
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
            if (val < k1_val) local_l0_num++;
            if (val >= k1_val && val < k2_val) local_l1_num++;
            if (val >= k2_val) local_l2_num++;
        }
        MPI_Allreduce(&local_l0_num, &l0_num, 1, MPI_LONG_LONG_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&local_l1_num, &l1_num, 1, MPI_LONG_LONG_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(&local_l2_num, &l2_num, 1, MPI_LONG_LONG_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        // printf("set sizes: %ld %ld %ld\n", l0_num, l1_num, l2_num);
        assert(l0_num + l1_num + l2_num == total_size);
        // []----*---o----*-----]
        // if there are more elements in the last set than elemenet k to end,
        // this means k2 is equal to k
        if (l2_num > total_size - k) return k2_val;
        assert(l0_num < k);

        std::vector<T> new_my_array;
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

template <class T>
std::pair<T, T> get_lower_upper_kth_parallel(std::vector<T> &my_array,
                                             int64_t total_size, int myrank,
                                             int n_pes, int64_t k,
                                             int type_enum) {
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    int64_t local_size = my_array.size();
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
    std::vector<T> all_sample_vec;
    int *rcounts = new int[n_pes];
    int *displs = new int[n_pes];
    int total_sample_size = 0;
    // gather the sample sizes
    MPI_Gather(&my_sample_size, 1, MPI_INT, rcounts, 1, MPI_INT, root,
               MPI_COMM_WORLD);
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
    MPI_Gatherv(my_sample.data(), my_sample_size, mpi_typ,
                all_sample_vec.data(), rcounts, displs, mpi_typ, root,
                MPI_COMM_WORLD);
    T k1_val;
    T k2_val;
    if (myrank == root) {
        int local_k = (int)(k * (total_sample_size / (T)total_size));
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
    MPI_Bcast(&k1_val, 1, mpi_typ, root, MPI_COMM_WORLD);
    MPI_Bcast(&k2_val, 1, mpi_typ, root, MPI_COMM_WORLD);
    // cleanup
    delete[] rcounts;
    delete[] displs;
    return std::make_pair(k1_val, k2_val);
}

template <class T>
T small_get_nth_parallel(std::vector<T> &my_array, int64_t total_size,
                         int myrank, int n_pes, int64_t k, int type_enum) {
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    T res;
    int my_data_size = my_array.size();
    int total_data_size = 0;
    std::vector<T> all_data_vec;

    // no need to gather data if only 1 processor
    if (n_pes == 1) {
        std::nth_element(my_array.begin(), my_array.begin() + k,
                         my_array.end());
        res = my_array[k];
        return res;
    }

    // gather the data sizes
    int *rcounts = new int[n_pes];
    int *displs = new int[n_pes];
    MPI_Gather(&my_data_size, 1, MPI_INT, rcounts, 1, MPI_INT, root,
               MPI_COMM_WORLD);
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
    MPI_Gatherv(my_array.data(), my_data_size, mpi_typ, all_data_vec.data(),
                rcounts, displs, mpi_typ, root, MPI_COMM_WORLD);
    // get nth element on root
    if (myrank == root) {
        std::nth_element(all_data_vec.begin(), all_data_vec.begin() + k,
                         all_data_vec.end());
        res = all_data_vec[k];
    }
    MPI_Bcast(&res, 1, mpi_typ, root, MPI_COMM_WORLD);
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
template <class T, int dtype>
inline typename std::enable_if<!is_decimal<dtype>::value, void>::type
collecting_non_nan_entries(std::vector<T> &my_array, array_info *arr,
                           local_global_stat_nan const &e_stat) {
    if (e_stat.loc_nb_miss == 0) {
        // Remark: The data is indeed copied in that case as well.
        T *data = (T *)arr->data1;
        my_array = std::vector<T>(data, data + e_stat.loc_nb_ok);
    } else {
        if (arr->arr_type == bodo_array_type::NUMPY) {
            for (int64_t i_row = 0; i_row < arr->length; i_row++) {
                T eVal = arr->at<T>(i_row);
                bool isna = isnan_alltype<T, dtype>(eVal);
                if (!isna) my_array.emplace_back(eVal);
            }
        }
        if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            for (int64_t i_row = 0; i_row < arr->length; i_row++) {
                if (GetBit((uint8_t *)arr->null_bitmask, i_row)) {
                    T eVal = arr->at<T>(i_row);
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
template <class T, int dtype>
inline typename std::enable_if<is_decimal<dtype>::value, void>::type
collecting_non_nan_entries(std::vector<T> &my_array, array_info *arr,
                           local_global_stat_nan const &e_stat) {
    for (int64_t i_row = 0; i_row < arr->length; i_row++) {
        if (GetBit((uint8_t *)arr->null_bitmask, i_row)) {
            decimal_value_cpp eVal = arr->at<decimal_value_cpp>(i_row);
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
template <class T, int dtype>
T get_nth(std::vector<T> &my_array, int64_t k, bool parallel) {
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
template <typename T, int dtype>
std::pair<int64_t, int64_t> nb_entries_local(array_info *arr) {
    int64_t nb_ok = 0, nb_miss = 0;
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        for (int i_row = 0; i_row < arr->length; i_row++) {
            if (GetBit((uint8_t *)arr->null_bitmask, i_row))
                nb_ok++;
            else
                nb_miss++;
        }
    }
    if (arr->arr_type == bodo_array_type::NUMPY) {
        for (int i_row = 0; i_row < arr->length; i_row++) {
            T eVal = arr->at<T>(i_row);
            bool isna = isnan_alltype<T, dtype>(eVal);
            if (isna)
                nb_miss++;
            else
                nb_ok++;
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
template <typename T, int dtype>
local_global_stat_nan nb_entries_global(array_info *arr, bool parallel) {
    std::pair<int64_t, int64_t> pair = nb_entries_local<T, dtype>(arr);
    if (!parallel) return {pair.first, pair.second, pair.first, pair.second};
    int64_t loc_nb_ok = pair.first, loc_nb_miss = pair.second, glob_nb_ok = 0,
            glob_nb_miss = 0;
    MPI_Allreduce(&loc_nb_ok, &glob_nb_ok, 1, MPI_LONG_LONG_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&loc_nb_miss, &glob_nb_miss, 1, MPI_LONG_LONG_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    return {glob_nb_ok, glob_nb_miss, loc_nb_ok, loc_nb_miss};
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
template <typename T, int dtype>
void median_series_computation_eff(double *res, std::vector<T> &my_array,
                                   bool parallel, int64_t glob_nb_ok) {
    if (glob_nb_ok % 2 == 1) {
        int kMid = glob_nb_ok / 2;
        T eVal = get_nth<T, dtype>(my_array, kMid, parallel);
        *res = double(eVal);
    } else {
        int kMid1 = glob_nb_ok / 2;
        int kMid2 = kMid1 - 1;
        T eVal1 = get_nth<T, dtype>(my_array, kMid1, parallel);
        T eVal2 = get_nth<T, dtype>(my_array, kMid2, parallel);
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
template <typename T, int dtype>
void median_series_computation_T(double *res, array_info *arr, bool parallel,
                                 bool skipna) {
    local_global_stat_nan e_stat = nb_entries_global<T, dtype>(arr, parallel);
    if ((e_stat.glob_nb_miss > 0 && !skipna) || e_stat.glob_nb_ok == 0) {
        *res = std::nan("");
        return;
    }
    std::vector<T> my_array;
    collecting_non_nan_entries<T, dtype>(my_array, arr, e_stat);
    int64_t glob_nb_ok = e_stat.glob_nb_ok;
    if (dtype == Bodo_CTypes::DECIMAL)
        median_series_computation_eff<T, Bodo_CTypes::FLOAT64>(
            res, my_array, parallel, glob_nb_ok);
    else
        median_series_computation_eff<T, dtype>(res, my_array, parallel,
                                                glob_nb_ok);
}

/** Compute the median of the series.
    @param arr: The array_info used for the computation
    @param parallel: whether to use the parallel algorithm
    @param skipna: whether to skip the nan entries or not.
    @output res: the median as output
    ---
    According to the data type we trigger a different templatized function.
 */
void median_series_computation(double *res, array_info *arr, bool parallel,
                               bool skipna) {
    Bodo_CTypes::CTypeEnum dtype = arr->dtype;
    switch (dtype) {
        case Bodo_CTypes::INT8:
            return median_series_computation_T<int8_t, Bodo_CTypes::INT8>(
                res, arr, parallel, skipna);
        case Bodo_CTypes::UINT8:
            return median_series_computation_T<uint8_t, Bodo_CTypes::UINT8>(
                res, arr, parallel, skipna);
        case Bodo_CTypes::INT16:
            return median_series_computation_T<int16_t, Bodo_CTypes::INT16>(
                res, arr, parallel, skipna);
        case Bodo_CTypes::UINT16:
            return median_series_computation_T<uint16_t, Bodo_CTypes::UINT16>(
                res, arr, parallel, skipna);
        case Bodo_CTypes::INT32:
            return median_series_computation_T<int32_t, Bodo_CTypes::INT32>(
                res, arr, parallel, skipna);
        case Bodo_CTypes::UINT32:
            return median_series_computation_T<uint32_t, Bodo_CTypes::UINT32>(
                res, arr, parallel, skipna);
        case Bodo_CTypes::INT64:
            return median_series_computation_T<int64_t, Bodo_CTypes::INT64>(
                res, arr, parallel, skipna);
        case Bodo_CTypes::UINT64:
            return median_series_computation_T<uint64_t, Bodo_CTypes::UINT64>(
                res, arr, parallel, skipna);
        case Bodo_CTypes::FLOAT32:
            return median_series_computation_T<float, Bodo_CTypes::FLOAT32>(
                res, arr, parallel, skipna);
        case Bodo_CTypes::FLOAT64:
            return median_series_computation_T<double, Bodo_CTypes::FLOAT64>(
                res, arr, parallel, skipna);
        case Bodo_CTypes::DECIMAL:
            // We choose to return double in case of decimal (while a decimal
            // return is conceptually feasible) because it would require more
            // work and pandas uses double in that case as well.
            return median_series_computation_T<double, Bodo_CTypes::DECIMAL>(
                res, arr, parallel, skipna);
        default:
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                "type not supported by median_series_computation");
    }
}
