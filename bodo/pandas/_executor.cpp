#include "_executor.h"
#include <limits.h>
#include <fstream>
#include "_bodo_scan_function.h"

// enable and build to print debug info on the pipeline
#define DEBUG_GPU_SELECTOR
#ifdef DEBUG_GPU_SELECTOR
#include <iostream>
#endif

enum DEVICE { CPU = 0, GPU = 1 };

/* -----------------------------
 * Cost model parameters
 * -----------------------------
 */

// Transfer model (PCIe-like)
double PCIe_BW = 12e9;             // bytes/s
double TRANSFER_OVERHEAD = 10e-6;  // 10 microseconds
double KERNEL_LAUNCH = 10e-6;      // 10 microseconds

double transfer_time(uint64_t bytes_size) {
    return TRANSFER_OVERHEAD + (bytes_size / PCIe_BW);
}

std::map<duckdb::LogicalOperatorType, std::vector<double>> ALPHA{
    {duckdb::LogicalOperatorType::LOGICAL_GET, {1e-9, 2e-10}},
    {duckdb::LogicalOperatorType::LOGICAL_COPY_TO_FILE, {1e-9, 2e-10}},
    {duckdb::LogicalOperatorType::LOGICAL_PROJECTION, {1e-9, 2e-10}},
    {duckdb::LogicalOperatorType::LOGICAL_LIMIT, {1e-9, 5e-10}},
    {duckdb::LogicalOperatorType::LOGICAL_FILTER, {2e-9, 3e-10}},
    {duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR, {0, 0}},
    {duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN, {8e-9, 1.5e-9}},
    {duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT, {1e-8, 2.5e-9}},
    {duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY,
     {5e-9, 8e-10}},
    {duckdb::LogicalOperatorType::LOGICAL_ORDER_BY, {6e-9, 1e-9}},
    {duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE, {1e-10, 1e-10}},
    {duckdb::LogicalOperatorType::LOGICAL_CTE_REF, {1e-10, 1e-10}},
    {duckdb::LogicalOperatorType::LOGICAL_TOP_N, {8e-8, 1.5e-8}},
    {duckdb::LogicalOperatorType::LOGICAL_SAMPLE, {3e-9, 1e-9}},
    {duckdb::LogicalOperatorType::LOGICAL_UNION, {1.5e-9, 7e-10}},
    {duckdb::LogicalOperatorType::LOGICAL_DISTINCT, {1e-7, 2e-8}}};

std::map<duckdb::LogicalOperatorType, uint64_t> GPU_MIN_SIZE{
    {duckdb::LogicalOperatorType::LOGICAL_GET, 10 * 1024 * 1024},
    {duckdb::LogicalOperatorType::LOGICAL_COPY_TO_FILE, 10 * 1024 * 1024},
    {duckdb::LogicalOperatorType::LOGICAL_PROJECTION, 10 * 1024 * 1024},
    {duckdb::LogicalOperatorType::LOGICAL_LIMIT, 1 * 1024 * 1024},
    {duckdb::LogicalOperatorType::LOGICAL_FILTER, 10 * 1024 * 1024},
    {duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR, 0},
    {duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN, 5 * 1024 * 1024},
    {duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT, 5 * 1024 * 1024},
    {duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY,
     5 * 1024 * 1024},
    {duckdb::LogicalOperatorType::LOGICAL_ORDER_BY, 5 * 1024 * 1024},
    {duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE, 0},
    {duckdb::LogicalOperatorType::LOGICAL_CTE_REF, 0},
    {duckdb::LogicalOperatorType::LOGICAL_TOP_N, 5 * 1024 * 1024},
    {duckdb::LogicalOperatorType::LOGICAL_SAMPLE, 3 * 1024 * 1024},
    {duckdb::LogicalOperatorType::LOGICAL_UNION, 4 * 1024 * 1024},
    {duckdb::LogicalOperatorType::LOGICAL_DISTINCT, 2 * 1024 * 1024}};

class DevicePlanNode {
   private:
    duckdb::LogicalOperator &op;
    std::vector<std::shared_ptr<DevicePlanNode>> children;
    uint64_t rows_in, rows_in_width;
    uint64_t rows_out, rows_out_width;
    bool gpu_capable;
    uint64_t id;
    DevicePlanNode *cte_root = nullptr;

    static uint64_t next_id;
    static std::map<duckdb::idx_t, DevicePlanNode *> cte_map;

    bool determineGPUCapable(duckdb::LogicalOperator &op) {
        switch (op.type) {
            case duckdb::LogicalOperatorType::LOGICAL_GET: {
                duckdb::LogicalGet &lget = op.Cast<duckdb::LogicalGet>();
                BodoScanFunctionData &scan_data =
                    lget.bind_data->Cast<BodoScanFunctionData>();
                return scan_data.canRunOnGPU(
                    lget.table_filters.filters.size() != 0,
                    (bool)lget.extra_info.limit_val);
            }

            case duckdb::LogicalOperatorType::LOGICAL_COPY_TO_FILE:
                return true;

            case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
                return true;

            case duckdb::LogicalOperatorType::LOGICAL_FILTER:
                return false;

            case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
                return false;

            case duckdb::LogicalOperatorType::LOGICAL_CTE_REF:
                return false;

            case duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
                return false;

            case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
                return op.Cast<duckdb::LogicalComparisonJoin>().join_type ==
                       duckdb::JoinType::INNER;
            }

            case duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
                return false;

            case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY:
                return false;

            case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
                return false;

            case duckdb::LogicalOperatorType::LOGICAL_TOP_N:
                return false;

            case duckdb::LogicalOperatorType::LOGICAL_SAMPLE:
                return false;

            case duckdb::LogicalOperatorType::LOGICAL_UNION:
                return false;

            case duckdb::LogicalOperatorType::LOGICAL_DISTINCT:
                return false;

            case duckdb::LogicalOperatorType::LOGICAL_EMPTY_RESULT:
                return false;

            case duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
                return true;

            default:
                throw std::runtime_error(
                    "determineGPUCapable doesn't support op." + op.ToString());
        }
    }

   public:
    DevicePlanNode(duckdb::LogicalOperator &_op) : op(_op) {
        id = DevicePlanNode::next_id;
        DevicePlanNode::next_id++;

        if (op.type == duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
            duckdb::LogicalMaterializedCTE &cte =
                op.Cast<duckdb::LogicalMaterializedCTE>();
            cte_map.insert({cte.table_index, this});
        } else if (op.type == duckdb::LogicalOperatorType::LOGICAL_CTE_REF) {
            duckdb::LogicalCTERef &cteref = op.Cast<duckdb::LogicalCTERef>();
            auto it = cte_map.find(cteref.cte_index);
            if (it != cte_map.end()) {
                cte_root = it->second;
            } else {
                throw std::runtime_error(
                    "LogicalCTERef couldn't find matching table_index in "
                    "DevicePlanNode.");
            }
        }

        for (auto &child : op.children) {
            children.push_back(std::make_shared<DevicePlanNode>(*child));
        }

        gpu_capable = determineGPUCapable(op);

        if (!op.has_estimated_cardinality) {
            gpu_capable = false;
#ifdef DEBUG_GPU_SELECTOR
            std::cout << "DevicePlanNode operator didn't have cardinality.\n"
                      << op.ToString() << std::endl;
#endif
            //            throw std::runtime_error(
            //                "DevicePlanNode operator didn't have
            //                cardinality.");
        }
        rows_out = op.estimated_cardinality;
        rows_out_width = 0;
        for (auto &type : op.types) {
            rows_out_width += GetTypeIdSize(type.InternalType());
        }
        rows_in = 0;
        rows_in_width = 0;
        for (auto &dpchild : children) {
            rows_in += dpchild->getRowsOut();
            rows_in_width += dpchild->getRowsOutWidth();
        }
#ifdef DEBUG_GPU_SELECTOR
        std::cout << "DevicePlanNode " << id << " IN(" << rows_in << " "
                  << rows_in_width << ") OUT(" << rows_out << " "
                  << rows_out_width << ") GPU? " << gpu_capable << " "
                  << cte_root << "\n"
                  << op.ToString() << std::endl;
#endif
    }

    duckdb::LogicalOperator &getOp() const { return op; }

    bool isGPUCapable() const { return gpu_capable; }

    uint64_t getId() const { return id; }

    uint64_t getRowsIn() const { return rows_in; }

    uint64_t getRowsInWidth() const { return rows_in_width; }

    uint64_t getInputSize() const { return rows_in * rows_in_width; }

    uint64_t getRowsOut() const { return rows_out; }

    uint64_t getRowsOutWidth() const { return rows_out_width; }

    uint64_t getOutputSize() const { return rows_out * rows_out_width; }

    const std::vector<std::shared_ptr<DevicePlanNode>> getChildren() const {
        return children;
    }

    static void startDeviceAssignment();
    static void endDeviceAssignment();
};

uint64_t DevicePlanNode::next_id = 0;
std::map<duckdb::idx_t, DevicePlanNode *> DevicePlanNode::cte_map;

void DevicePlanNode::startDeviceAssignment() {}
void DevicePlanNode::endDeviceAssignment() { cte_map.clear(); }

using NodeDeviceMap = std::map<uint64_t, DEVICE>;

class DPCost {
   public:
    double cpu_cost;
    double gpu_cost;
    NodeDeviceMap cpu_children_choice;
    NodeDeviceMap gpu_children_choice;
};

using NodeCostMap = std::map<uint64_t, DPCost>;

struct Calibration {
    double pcie_bw = 12e9;  // bytes/s
    double transfer_overhead = 10e-6;
    double kernel_launch = 10e-6;
    double cpu_alpha_filter = 2e-9;
    double gpu_alpha_filter = 3e-10;
    double cpu_alpha_sort = 6e-8;
    double gpu_alpha_sort = 1e-8;
    bool valid = false;
};

static Calibration g_calib;
static std::string get_calib_path() {
    const char *home = std::getenv("BOD_GPU_CALIBRATION");
    if (!home) {
        home = ".";
    }
    std::string ret = std::string(home) + "/.bodo_gpu_calibration.txt";
#ifdef DEBUG_GPU_SELECTOR
    std::cout << "GPU calibration file " << ret << std::endl;
#endif
    return ret;
}

static void save_calibration(const Calibration &c) {
    std::ofstream out(get_calib_path());
    if (!out)
        return;
#ifdef DEBUG_GPU_SELECTOR
    std::cout << "saving GPU calibration" << std::endl;
#endif
    out << "pcie_bw=" << c.pcie_bw << "\n";
    out << "transfer_overhead=" << c.transfer_overhead << "\n";
    out << "kernel_launch=" << c.kernel_launch << "\n";
    out << "cpu_alpha_filter=" << c.cpu_alpha_filter << "\n";
    out << "gpu_alpha_filter=" << c.gpu_alpha_filter << "\n";
    out << "cpu_alpha_sort=" << c.cpu_alpha_sort << "\n";
    out << "gpu_alpha_sort=" << c.gpu_alpha_sort << "\n";
}

static bool load_calibration(Calibration &c) {
    std::ifstream in(get_calib_path());
#ifdef DEBUG_GPU_SELECTOR
    std::cout << "loading GPU calibration" << std::endl;
#endif
    if (!in)
        return false;
#ifdef DEBUG_GPU_SELECTOR
    std::cout << "found GPU calibration" << std::endl;
#endif
    std::string line;
    while (std::getline(in, line)) {
        auto pos = line.find('=');
        if (pos == std::string::npos)
            continue;
        std::string key = line.substr(0, pos);
        double val = std::stod(line.substr(pos + 1));
        if (key == "pcie_bw")
            c.pcie_bw = val;
        else if (key == "transfer_overhead")
            c.transfer_overhead = val;
        else if (key == "kernel_launch")
            c.kernel_launch = val;
        else if (key == "cpu_alpha_filter")
            c.cpu_alpha_filter = val;
        else if (key == "gpu_alpha_filter")
            c.gpu_alpha_filter = val;
        else if (key == "cpu_alpha_sort")
            c.cpu_alpha_sort = val;
        else if (key == "gpu_alpha_sort")
            c.gpu_alpha_sort = val;
    }
    c.valid = true;
    return true;
}

#ifdef USE_CUDF

#include <cuda_runtime.h>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

static double now_seconds() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch())
        .count();
}

/* ---------------- PCIe calibration (no custom kernels) ---------------- */

static void measure_pcie(Calibration &c) {
    const size_t bytes = 64ull * 1024 * 1024;
    void *h_ptr = nullptr;
    void *d_ptr = nullptr;

    cudaMallocHost(&h_ptr, bytes);
    cudaMalloc(&d_ptr, bytes);

    // warmup
    cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    const int iters = 10;
    double t0 = now_seconds();
    for (int i = 0; i < iters; ++i) {
        cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    double t1 = now_seconds();

    double avg = (t1 - t0) / iters;
    c.pcie_bw = bytes / avg;
    c.transfer_overhead = 5e-6;  // conservative floor

    cudaFree(d_ptr);
    cudaFreeHost(h_ptr);
}

/* ---------------- libcudf “launch” overhead via empty op ---------------- */

static void measure_kernel_launch_effective(Calibration &c) {
    // zero-row int32 column
    auto col =
        cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, 0);

    cudf::table_view tv({col->view()});

    // warmup
    auto sorted_warmup = cudf::sort(tv);

    const int iters = 1000;
    double t0 = now_seconds();
    for (int i = 0; i < iters; ++i) {
        auto sorted = cudf::sort(tv);
        (void)sorted;
    }
    cudaDeviceSynchronize();
    double t1 = now_seconds();

    c.kernel_launch = (t1 - t0) / iters;
}

/* ---------------- Filter throughput: CPU vs libcudf ---------------- */

static void measure_filter_cpu_gpu(Calibration &c) {
    const size_t n = 64ull * 1024 * 1024 / sizeof(int32_t);

    // CPU side
    {
        std::vector<int32_t> data(n, 1);
        double t0 = now_seconds();
        volatile int64_t sum = 0;
        for (size_t i = 0; i < n; ++i) {
            if (data[i] > 0)
                sum += data[i];
        }
        double t1 = now_seconds();
        double bytes = n * sizeof(int32_t);
        c.cpu_alpha_filter = (t1 - t0) / bytes;
    }

    // GPU side via libcudf: apply_boolean_mask
    {
        // data column
        auto col =
            cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, n);
        {
            // fill with 1s
            auto view = col->mutable_view();
            std::vector<int32_t> host(n, 1);
            cudaMemcpy(view.head<int32_t>(), host.data(), n * sizeof(int32_t),
                       cudaMemcpyHostToDevice);
        }

        // mask: all true
        auto mask_col =
            cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8}, n);
        {
            auto view = mask_col->mutable_view();
            std::vector<int8_t> host(n, 1);
            cudaMemcpy(view.head<int8_t>(), host.data(), n * sizeof(int8_t),
                       cudaMemcpyHostToDevice);
        }

        cudf::column_view data_view = col->view();
        cudf::column_view mask_view = mask_col->view();

        // warmup
        cudf::table_view tv({data_view});
        auto filtered_warmup = cudf::apply_boolean_mask(tv, mask_view);

        const int iters = 5;
        double t0 = now_seconds();
        for (int i = 0; i < iters; ++i) {
            auto filtered = cudf::apply_boolean_mask(tv, mask_view);
            (void)filtered;
        }
        cudaDeviceSynchronize();
        double t1 = now_seconds();

        double total_bytes = static_cast<double>(n) * sizeof(int32_t);
        double avg = (t1 - t0) / iters;
        c.gpu_alpha_filter = avg / total_bytes;
    }
}

/* ---------------- Sort throughput: CPU vs libcudf ---------------- */

static void measure_sort_cpu_gpu(Calibration &c) {
    const size_t n = 4ull * 1024 * 1024;

    // CPU
    {
        std::vector<int32_t> data(n);
        for (size_t i = 0; i < n; ++i)
            data[i] = (int32_t)(n - i);

        double t0 = now_seconds();
        std::sort(data.begin(), data.end());
        double t1 = now_seconds();

        double bytes = n * sizeof(int32_t);
        c.cpu_alpha_sort = (t1 - t0) / (bytes * std::log2((double)n));
    }

    // GPU via libcudf sort
    {
        auto col =
            cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32}, n);
        {
            auto view = col->mutable_view();
            std::vector<int32_t> host(n);
            for (size_t i = 0; i < n; ++i)
                host[i] = (int32_t)(n - i);
            cudaMemcpy(view.head<int32_t>(), host.data(), n * sizeof(int32_t),
                       cudaMemcpyHostToDevice);
        }

        cudf::table_view tv({col->view()});

        // warmup
        auto sorted_warmup = cudf::sort(tv);

        const int iters = 3;
        double t0 = now_seconds();
        for (int i = 0; i < iters; ++i) {
            auto sorted = cudf::sort(tv);
            (void)sorted;
        }
        cudaDeviceSynchronize();
        double t1 = now_seconds();

        double bytes = n * sizeof(int32_t);
        double avg = (t1 - t0) / iters;
        c.gpu_alpha_sort = avg / (bytes * std::log2((double)n));
    }
}

/* ---------------- Public entry point ---------------- */

void run_cudf_calibration(Calibration &c) {
    measure_pcie(c);
    measure_kernel_launch_effective(c);
    measure_filter_cpu_gpu(c);
    measure_sort_cpu_gpu(c);
    c.valid = true;
}

static void run_calibration(Calibration &c) {
    if (!load_calibration(c)) {
#ifdef DEBUG_GPU_SELECTOR
        std::cout << "Running GPU calibration" << std::endl;
#endif
        run_cudf_calibration(c);
        save_calibration(c);
    }
}

#else

static void run_calibration(Calibration &c) {
    // No CUDA: keep defaults, mark valid.
    c.valid = true;
}

#endif

static bool g_calib_initialized = false;

static void init_cost_model() {
    if (g_calib_initialized)
        return;
    run_calibration(g_calib);
    g_calib_initialized = true;

    PCIe_BW = g_calib.pcie_bw;
    TRANSFER_OVERHEAD = g_calib.transfer_overhead;
    KERNEL_LAUNCH = g_calib.kernel_launch;

    // We only get the time of one slow and one faster operation and
    // compute other slow or faster operations as percentages of those.
    double cpu_f = g_calib.cpu_alpha_filter;
    double gpu_f = g_calib.gpu_alpha_filter;
    double cpu_s = g_calib.cpu_alpha_sort;
    double gpu_s = g_calib.gpu_alpha_sort;

    ALPHA = {
        {duckdb::LogicalOperatorType::LOGICAL_GET, {cpu_f * 0.5, gpu_f * 0.5}},
        {duckdb::LogicalOperatorType::LOGICAL_COPY_TO_FILE,
         {cpu_f * 0.5, gpu_f * 0.5}},
        {duckdb::LogicalOperatorType::LOGICAL_PROJECTION,
         {cpu_f * 0.5, gpu_f * 0.5}},
        {duckdb::LogicalOperatorType::LOGICAL_LIMIT,
         {cpu_f * 0.5, gpu_f * 0.25}},
        {duckdb::LogicalOperatorType::LOGICAL_FILTER, {cpu_f, gpu_f}},
        {duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR, {0, 0}},
        {duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN,
         {cpu_f * 4.0, gpu_f * 5.0}},
        {duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT,
         {cpu_f * 5.0, gpu_f * 8.0}},
        {duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY,
         {cpu_f * 2.5, gpu_f * 3.0}},
        {duckdb::LogicalOperatorType::LOGICAL_ORDER_BY, {cpu_s, gpu_s}},
        {duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE, {1e-10, 1e-10}},
        {duckdb::LogicalOperatorType::LOGICAL_CTE_REF, {1e-10, 1e-10}},
        {duckdb::LogicalOperatorType::LOGICAL_TOP_N,
         {cpu_s * 1.3, gpu_s * 1.3}},
        {duckdb::LogicalOperatorType::LOGICAL_SAMPLE,
         {cpu_f * 1.5, gpu_f * 1.5}},
        {duckdb::LogicalOperatorType::LOGICAL_UNION,
         {cpu_f * 0.75, gpu_f * 0.75}},
        {duckdb::LogicalOperatorType::LOGICAL_DISTINCT,
         {cpu_f * 50.0, gpu_f * 20.0}}};

    GPU_MIN_SIZE = {
        {duckdb::LogicalOperatorType::LOGICAL_GET, 10 * 1024 * 1024},
        {duckdb::LogicalOperatorType::LOGICAL_COPY_TO_FILE, 10 * 1024 * 1024},
        {duckdb::LogicalOperatorType::LOGICAL_PROJECTION, 10 * 1024 * 1024},
        {duckdb::LogicalOperatorType::LOGICAL_LIMIT, 1 * 1024 * 1024},
        {duckdb::LogicalOperatorType::LOGICAL_FILTER, 10 * 1024 * 1024},
        {duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR, 0},
        {duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN, 5 * 1024 * 1024},
        {duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT, 5 * 1024 * 1024},
        {duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY,
         5 * 1024 * 1024},
        {duckdb::LogicalOperatorType::LOGICAL_ORDER_BY, 5 * 1024 * 1024},
        {duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE, 0},
        {duckdb::LogicalOperatorType::LOGICAL_CTE_REF, 0},
        {duckdb::LogicalOperatorType::LOGICAL_TOP_N, 5 * 1024 * 1024},
        {duckdb::LogicalOperatorType::LOGICAL_SAMPLE, 3 * 1024 * 1024},
        {duckdb::LogicalOperatorType::LOGICAL_UNION, 4 * 1024 * 1024},
        {duckdb::LogicalOperatorType::LOGICAL_DISTINCT, 2 * 1024 * 1024}};
}

double compute_time(std::shared_ptr<DevicePlanNode> node, DEVICE device) {
    init_cost_model();  // load or run tests to gen cost model

    auto op = node->getOp().type;

    auto alpha_iter = ALPHA.find(op);
    if (alpha_iter == ALPHA.end()) {
        throw std::runtime_error("compute_time didn't find op in ALPHA.\n" +
                                 node->getOp().ToString());
    }
    double a = alpha_iter->second[device];

    double n_in = std::max((double)node->getRowsIn(), 1.0);
    uint64_t size_in = node->getInputSize();
    uint64_t size_out = node->getOutputSize();

    double t = 0;

    switch (op) {
        case duckdb::LogicalOperatorType::LOGICAL_GET:
            t = a * size_out;
            break;

        case duckdb::LogicalOperatorType::LOGICAL_COPY_TO_FILE:
        case duckdb::LogicalOperatorType::LOGICAL_CTE_REF:
        case duckdb::LogicalOperatorType::LOGICAL_PROJECTION:
        case duckdb::LogicalOperatorType::LOGICAL_FILTER:
        case duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR:
            t = a * size_in;
            break;

        case duckdb::LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY:
            t = a * n_in * std::log2(n_in);
            break;

        case duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE:
            t = 0;
            break;

        case duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
        case duckdb::LogicalOperatorType::LOGICAL_CROSS_PRODUCT:
            t = a * (node->getChildren()[0]->getRowsOut() +
                     node->getChildren()[1]->getRowsOut());
            break;

        case duckdb::LogicalOperatorType::LOGICAL_ORDER_BY:
            t = a * n_in * std::log2(n_in);
            break;

        case duckdb::LogicalOperatorType::LOGICAL_LIMIT:
            t = a * node->getRowsOut();
            break;

        case duckdb::LogicalOperatorType::LOGICAL_TOP_N: {
            auto &topn = node->getOp().Cast<duckdb::LogicalTopN>();
            double k = std::max<double>(topn.limit, 2.0);
            t = a * n_in * std::log2(k);
            break;
        }

        case duckdb::LogicalOperatorType::LOGICAL_SAMPLE:
            t = a * n_in;
            break;

        case duckdb::LogicalOperatorType::LOGICAL_UNION:
            t = a * node->getRowsOut();
            break;

        case duckdb::LogicalOperatorType::LOGICAL_DISTINCT:
            t = a * n_in;  // hashing
            break;

        default:
            throw std::runtime_error("compute_time doesn't support op.");
    }

    if (device == DEVICE::GPU) {
        t += KERNEL_LAUNCH;
        auto gpu_min_size_iter = GPU_MIN_SIZE.find(op);
        if (gpu_min_size_iter == GPU_MIN_SIZE.end()) {
            throw std::runtime_error(
                "compute_time didn't find op in GPU_MIN_SIZE.");
        }
        uint64_t min_size = gpu_min_size_iter->second;
        if (size_in < min_size) {
            t *= 1.5;
        }
    }

#ifdef DEBUG_GPU_SELECTOR
    std::cout << "compute_time " << node->getId() << " "
              << (device == DEVICE::CPU ? "CPU" : "GPU") << " " << t
              << std::endl;
#endif

    return t;
}

/*
 * Compute dynamic programming (DP) costs for all nodes.
 */
DPCost dp_compute(std::shared_ptr<DevicePlanNode> node, NodeCostMap &dp_cache) {
    // If DPCost already computed return it.
    auto dp_iter = dp_cache.find(node->getId());
    if (dp_iter != dp_cache.end()) {
        return dp_iter->second;
    }

#ifdef DEBUG_GPU_SELECTOR
    std::cout << "dp_compute start " << node->getId() << std::endl;
#endif

    // Compute child costs first
    NodeCostMap child_costs;
    std::map<uint64_t, uint64_t> child_sizes;
    for (auto &child : node->getChildren()) {
        child_costs.insert({child->getId(), dp_compute(child, dp_cache)});
        child_sizes.insert({child->getId(), child->getOutputSize()});
    }

    // Determine the node-only time.
    double cpu_total = compute_time(node, DEVICE::CPU);
    double gpu_total = std::numeric_limits<double>::infinity();
    NodeDeviceMap cpu_children_choice;
    NodeDeviceMap gpu_children_choice;

    if (node->isGPUCapable()) {
        gpu_total = compute_time(node, DEVICE::GPU);
    }

#ifdef DEBUG_GPU_SELECTOR
    std::cout << "dp_compute node-only " << node->getId() << " " << cpu_total
              << " " << gpu_total << std::endl;
#endif

    // Add in the cost of the children with any transfer times and
    // pick the best one.
    for (auto &child : node->getChildren()) {
        DPCost &cc = child_costs[child->getId()];
        double cost_cpu = cc.cpu_cost;
        double cost_gpu = std::numeric_limits<double>::infinity();
        if (child->isGPUCapable()) {
            if (node->getOp().type ==
                duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
                cost_gpu = cc.gpu_cost;
            } else {
                cost_gpu =
                    cc.gpu_cost + transfer_time(child_sizes[child->getId()]);
            }
        }

        if (cost_cpu <= cost_gpu) {
            cpu_total += cost_cpu;
            cpu_children_choice.insert({child->getId(), DEVICE::CPU});
        } else {
            cpu_total += cost_gpu;
            cpu_children_choice.insert({child->getId(), DEVICE::GPU});
        }

#ifdef DEBUG_GPU_SELECTOR
        std::cout << "dp_compute child cpu " << node->getId() << " "
                  << child->getId() << " " << cost_cpu << " " << cost_gpu
                  << std::endl;
#endif

        if (node->isGPUCapable()) {
            cost_gpu = cc.gpu_cost;
            if (node->getOp().type ==
                duckdb::LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
                cost_cpu = cc.cpu_cost;
            } else {
                cost_cpu =
                    cc.cpu_cost + transfer_time(child_sizes[child->getId()]);
            }

            if (cost_gpu <= cost_cpu) {
                gpu_total += cost_gpu;
                gpu_children_choice.insert({child->getId(), DEVICE::GPU});
            } else {
                gpu_total += cost_cpu;
                gpu_children_choice.insert({child->getId(), DEVICE::CPU});
            }

#ifdef DEBUG_GPU_SELECTOR
            std::cout << "dp_compute child gpu " << node->getId() << " "
                      << child->getId() << " " << cost_cpu << " " << cost_gpu
                      << std::endl;
#endif
        }
    }

#ifdef DEBUG_GPU_SELECTOR
    std::cout << "dp_compute end " << node->getId() << " " << cpu_total << " "
              << gpu_total << std::endl;
#endif

    DPCost dpc(cpu_total, gpu_total, cpu_children_choice, gpu_children_choice);
    dp_cache.insert({node->getId(), dpc});
    return dpc;
}

void assign_devices(std::shared_ptr<DevicePlanNode> node, NodeCostMap &dp_cache,
                    std::map<void *, bool> &run_on_gpu,
                    std::optional<DEVICE> chosen_device = {}) {
    auto dp_iter = dp_cache.find(node->getId());
    if (dp_iter == dp_cache.end()) {
        throw std::runtime_error(
            "DPCost for node not found in assign_devices.");
    }
    DPCost &dpc = dp_iter->second;

    DEVICE dev;
    if (chosen_device.has_value()) {
        dev = *chosen_device;
    } else {
        dev = (dpc.cpu_cost <= dpc.gpu_cost) ? DEVICE::CPU : DEVICE::GPU;
    }

#ifdef DEBUG_GPU_SELECTOR
    std::cout << "assign_devices " << node->getId() << " will run on "
              << (dev == DEVICE::CPU ? "CPU" : "GPU") << std::endl;
#endif

    run_on_gpu[&(node->getOp())] = (dev == DEVICE::GPU);

    NodeDeviceMap &chosen_map = (dev == DEVICE::GPU) ? dpc.gpu_children_choice
                                                     : dpc.cpu_children_choice;

    for (auto &child : node->getChildren()) {
        auto cmiter = chosen_map.find(child->getId());
        if (cmiter == chosen_map.end()) {
            throw std::runtime_error("Child node ID not found in chosen_map.");
        }
        assign_devices(child, dp_cache, run_on_gpu, cmiter->second);
    }
}

void Executor::partition_internal(duckdb::LogicalOperator &op,
                                  std::map<void *, bool> &run_on_gpu) {
    if (get_use_cudf()) {
        DevicePlanNode::startDeviceAssignment();
        // Converts DuckDB LogicalOperator tree to DevicePlanNode tree.
        std::shared_ptr<DevicePlanNode> root =
            std::make_shared<DevicePlanNode>(op);
        DevicePlanNode::endDeviceAssignment();

        std::map<uint64_t, DPCost> dp_cache;
        dp_compute(root, dp_cache);
        assign_devices(root, dp_cache, run_on_gpu);
    } else {
        for (auto &child : op.children) {
            partition_internal(*child, run_on_gpu);
        }
        // Run on CPU always if CUDF not enabled.
        run_on_gpu[&op] = false;
    }
}
