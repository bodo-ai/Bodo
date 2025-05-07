//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/extra_operator_info.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include "duckdb/common/operator/comparison_operators.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/parser/parsed_data/sample_options.hpp"
// Bodo Change: limit_pushdown to data source not previously supported by DuckDB
#include "duckdb/planner/bound_result_modifier.hpp"

namespace duckdb {

class ExtraOperatorInfo {
public:
	ExtraOperatorInfo() : file_filters(""), sample_options(nullptr), limit_val(nullptr), offset_val(nullptr) {
	}
	ExtraOperatorInfo(ExtraOperatorInfo &extra_info)
	    : file_filters(extra_info.file_filters), sample_options(std::move(extra_info.sample_options)) {
		if (extra_info.total_files.IsValid()) {
			total_files = extra_info.total_files.GetIndex();
		}
		if (extra_info.filtered_files.IsValid()) {
			filtered_files = extra_info.filtered_files.GetIndex();
		}
	}

	//! Filters that have been pushed down into the main file list
	string file_filters;
	//! Total size of file list
	optional_idx total_files;
	//! Size of file list after applying filters
	optional_idx filtered_files;
	//! Sample options that have been pushed down into the table scan
	unique_ptr<SampleOptions> sample_options;
    // Bodo Change: limit_pushdown to data source not previously supported by DuckDB
    //! Limit options that have been pushed down into the table scan
    unique_ptr<BoundLimitNode> limit_val;
    unique_ptr<BoundLimitNode> offset_val;
};

} // namespace duckdb
