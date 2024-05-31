# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Tests reading data from a TabularCatalog in a manner that will cause a runtime join filter 
to be pushed down to I/O.
"""


import io

import pandas as pd

import bodosql
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import (
    check_func,
    pytest_tabular,
)

pytestmark = pytest_tabular


def test_simple_join(tabular_catalog, memory_leak_check):
    def impl(bc, query):
        return bc.sql(query)

    bc = bodosql.BodoSQLContext(catalog=tabular_catalog)

    # Joins the locations & yellow tables to look at each zone in
    # staten island and count the number of rides that occurred had
    # the same pickup and dropoff point in location in each such zone.
    # Note: imposes a limit on the yellow table for practical reasons.
    # This should result in the nation table producing a runtime
    # join filter on the pickup location id & dropoff location id to only
    # read the rows where the values are between 5 and 204.
    query = """
    SELECT loc.\"zone_name\" as zone_name, COUNT(*) as n_rides
    FROM \"examples\".\"nyc_taxi_locations\" loc, (SELECT * FROM \"examples\".\"nyc_taxi_yellow\" LIMIT 50000) yel
    WHERE 
        yel.\"pickup_location_id\" = loc.\"location_id\"
        AND yel.\"dropoff_location_id\"
        AND loc.\"borough\" = 'Staten Island'
    GROUP BY zone_name
    """
    py_output = pd.DataFrame(
        {
            "ZONE_NAME": [
                "Arden Heights",
                "Bloomfield/Emerson Hill",
                "Charleston/Tottenville",
                "Eltingville/Annadale/Prince's Bay",
                "Freshkills Park",
                "New Dorp/Midland Beach",
                "Rossville/Woodrow",
            ],
            "N_RIDES": [2, 2, 7, 1, 1, 1, 1],
        }
    )

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        check_func(
            impl,
            (bc, query),
            py_output=py_output,
            only_1DVar=True,
            sort_output=True,
            reset_index=True,
        )
        # TODO: in future, add logger message checking whether the expected
        # runtime join filter occurred during the iceberg read. This should
        # verify that pickup_location_id and dropoff_location_id are both
        # between 5 and 204
        check_logger_msg(
            stream,
            "",
        )
