#include "gpu_aggregate.h"

// Definition of the static member
const std::map<std::string, int32_t> PhysicalGPUAggregate::function_to_ftype = {
    {"count", Bodo_FTypes::count},
    {"max", Bodo_FTypes::max},
    {"mean", Bodo_FTypes::mean},
    {"median", Bodo_FTypes::median},
    {"min", Bodo_FTypes::min},
    {"nunique", Bodo_FTypes::nunique},
    {"size", Bodo_FTypes::size},
    {"skew", Bodo_FTypes::skew},
    {"std", Bodo_FTypes::std},
    {"sum", Bodo_FTypes::sum},
    {"var", Bodo_FTypes::var},
    {"std_pop", Bodo_FTypes::std_pop},
    {"var_pop", Bodo_FTypes::var_pop},
    {"first", Bodo_FTypes::first},
    {"last", Bodo_FTypes::last},
    {"boolor_agg", Bodo_FTypes::boolor_agg},
    {"booland_agg", Bodo_FTypes::booland_agg},
    {"boolxor_agg", Bodo_FTypes::boolxor_agg}};
