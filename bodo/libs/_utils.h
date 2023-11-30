#pragma once
#include <tuple>

/**
 * @brief Get the number of ranks on this node and the
 * current rank's position.
 *
 * We do this by creating sub-communicators based
 * on shared-memory. This is a collective operation
 * and therefore all ranks must call it.
 */
std::tuple<int, int> dist_get_ranks_on_node();
