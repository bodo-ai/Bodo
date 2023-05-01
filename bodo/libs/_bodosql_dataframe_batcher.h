// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef _BODO_DATAFRAME_BATCHER_H_INCLUDED_
#define _BODO_DATAFRAME_BATCHER_H_INCLUDED_

#define DEFAULT_BATCH_SIZE 4000

#include "_bodo_common.h"

// Class used to batch
class BodoSqlDataFrameBatcher {
   private:
    table_info* ti;

   public:
    BodoSqlDataFrameBatcher(table_info* ti);
    ~BodoSqlDataFrameBatcher();
    table_info::get_next_batch();
};

#endif /* _BODO_DATAFRAME_BATCHER_H_INCLUDED_ */
