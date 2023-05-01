// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef _BODO_DATAFRAME_BATCHER_H_INCLUDED_
#define _BODO_DATAFRAME_BATCHER_H_INCLUDED_

#include "_bodo_common.h"

namespace bodo {

BodoSqlDataFrameBatcher::BodoSqlDataFrameBatcher(table_info* ti) {
    this->ti = ti;
}

BodoSqlDataFrameBatcher::~BodoSqlDataFrameBatcher() { delete ti; }

table_info::get_next_batch() {
    int length = this.ti->columns.at(0);

    for (int i = 0; i < this.ti->columns.size(); i++) {
        current_column = this.ti->columns.get(i);

        std::vector<int> split_lo(current_column.begin(),
                                  current_column.begin() + DEFAULT_BATCH_SIZE);
        std::vector<int> split_hi(current_column.begin() + DEFAULT_BATCH_SIZE,
                                  current_column.end());
    }
}

}  // namespace bodo

#endif /* _BODO_DATAFRAME_BATCHER_H_INCLUDED_ */
