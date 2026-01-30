#include "_executor.h"
#include <limits.h>
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
    {duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR, {2e-9, 3e-10}},
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
    {duckdb::LogicalOperatorType::LOGICAL_EXTENSION_OPERATOR, 10 * 1024 * 1024},
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
                return false;

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

double compute_time(std::shared_ptr<DevicePlanNode> node, DEVICE device) {
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
