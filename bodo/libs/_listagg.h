#pragma once

#include "_bodo_common.h"

/**
 * Performs Listagg aggregation on a single column, with the given separator.
 *
 * @param in_table: Table containing the aggregate column, and the order
 * columns. We expect the first num_order_cols, to be the order columns,
 * followed by a single aggregation column.
 * @param separator: the separator to use between values
 * @param num_order_cols: the number of order columns
 * @param ascending: an array of booleans indicating whether each order
 * column is ascending (length == num_order_cols)
 * @param na_position: an array of booleans indicating whether each order
 * column has NA's first (length == num_order_cols)
 * @param output_string_size: a pointer to an int64_t, which will be set to the
 * size of the ouput string by this function
 *
 * @returns: a char* containing the listagg'd column
 */
char *listagg_seq(std::shared_ptr<table_info> in_table,
                  const std::string &separator, int num_order_cols,
                  bool *window_ascending, bool *window_na_position,
                  int64_t *output_string_size_ptr);
