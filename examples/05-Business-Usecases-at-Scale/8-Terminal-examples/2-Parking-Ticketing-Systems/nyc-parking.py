"""
Source: https://github.com/JBlumstein/NYCParking/blob/master/NYC_Parking_Violations_Mapping_Example.ipynb
NYC parking ticket violations
    Usage:
        python nyc-parking.py
        
    Set the environment variable `BODO_NUM_WORKERS` to limit the number of cores used.

Data for 2016 and 2017 is in S3 bucket (s3://bodo-example-data/nyc-parking-tickets)
or you can get data from https://www.kaggle.com/new-york-city/nyc-parking-tickets

"""
import numpy as np
import pandas as pd
import time
import bodo
import os


@bodo.jit(distributed=["many_year_df"], cache=True)
def load_parking_tickets():
    """
    Load data from S3 bucket and aggregate by day, violation type, and police precinct.
    """

    start = time.time()
    year_2016_df = pd.read_csv(
        "s3://bodo-example-data/nyc-parking-tickets/Parking_Violations_Issued_-_Fiscal_Year_2016.csv",
        parse_dates=["Issue Date"],
    )
    year_2016_df = year_2016_df.groupby(
        ["Issue Date", "Violation County", "Violation Precinct", "Violation Code"],
        as_index=False,
    )["Summons Number"].count()

    year_2017_df = pd.read_csv(
        "s3://bodo-example-data/nyc-parking-tickets/Parking_Violations_Issued_-_Fiscal_Year_2017.csv",
        parse_dates=["Issue Date"],
    )
    year_2017_df = year_2017_df.groupby(
        ["Issue Date", "Violation County", "Violation Precinct", "Violation Code"],
        as_index=False,
    )["Summons Number"].count()

    # concatenate all dataframes into one dataframe
    many_year_df = pd.concat([year_2016_df, year_2017_df])
    end = time.time()
    print("Reading Time: ", end - start)
    print(many_year_df.head())
    return many_year_df


main_df = load_parking_tickets()


@bodo.jit
def load_violation_precincts_codes(dir_path):
    """
    Load violation codes and precincts information.
    """
    start = time.time()
    violation_codes = pd.read_csv(f"{dir_path}/DOF_Parking_Violation_Codes.csv")
    violation_codes.columns = [
        "Violation Code",
        "Definition",
        "manhattan_96_and_below",
        "all_other_areas",
    ]
    nyc_precincts_df = pd.read_csv(f"{dir_path}/nyc_precincts.csv", index_col="index")
    end = time.time()
    print("Violation and precincts load Time: ", end - start)
    return violation_codes, nyc_precincts_df


dir_path = os.path.dirname(os.path.realpath(__file__))
violation_codes, nyc_precincts_df = load_violation_precincts_codes(dir_path)


@bodo.jit(distributed=["main_df"], cache=True)
def elim_code_36(main_df):
    """
    Remove undefined violations (code 36)
    """
    start = time.time()
    main_df = main_df[main_df["Violation Code"] != 36].sort_values(
        "Summons Number", ascending=False
    )
    end = time.time()
    print("Eliminate undefined violations time: ", end - start)
    print(main_df.head())
    return main_df


main_df = elim_code_36(main_df)


@bodo.jit(distributed=["main_df"], cache=True)
def remove_outliers(main_df):
    """
    Delete entries that have dates outside our dataset dates
    """
    start = time.time()
    main_df = main_df[
        (main_df["Issue Date"] >= "2016-01-01")
        & (main_df["Issue Date"] <= "2017-12-31")
    ]
    end = time.time()
    print("Remove outliers time: ", (end - start))
    print(main_df.head())
    return main_df


main_df = remove_outliers(main_df)


@bodo.jit(distributed=["main_df"], cache=True)
def merge_violation_code(main_df, violation_codes):
    """
    Merge violation information in the main_df
    """
    start = time.time()
    # left join main_df and violation_codes df so that there's more info on violation in main_df
    main_df = pd.merge(main_df, violation_codes, on="Violation Code", how="left")
    # cast precincts as integers from floats (inadvertent type change by merge)
    main_df["Violation Precinct"] = main_df["Violation Precinct"].astype(int)
    end = time.time()
    print("Merge time: ", (end - start))
    print(main_df.head())
    return main_df


main_df = merge_violation_code(main_df, violation_codes)


@bodo.jit(distributed=["main_df"], cache=True)
def calculate_total_summons(main_df):
    """
    Calculate the total summonses in dollars for a violation in a precinct on a day
    """
    start = time.time()
    # create column for portion of precinct 96th st. and below
    n = len(main_df)
    portion_manhattan_96_and_below = np.empty(n, np.int64)
    # NOTE: To run pandas, use this loop.
    # for i in range(n):
    for i in bodo.prange(n):
        x = main_df["Violation Precinct"].iat[i]
        if x < 22 or x == 23:
            portion_manhattan_96_and_below[i] = 1.0
        elif x == 22:
            portion_manhattan_96_and_below[i] = 0.75
        elif x == 24:
            portion_manhattan_96_and_below[i] = 0.5
        else:  # other
            portion_manhattan_96_and_below[i] = 0
    main_df["portion_manhattan_96_and_below"] = portion_manhattan_96_and_below

    # create column for average dollar amount of summons based on location
    main_df["average_summons_amount"] = (
        main_df["portion_manhattan_96_and_below"] * main_df["manhattan_96_and_below"]
        + (1 - main_df["portion_manhattan_96_and_below"]) * main_df["all_other_areas"]
    )

    # get total summons dollars by multiplying average dollar amount by number of summons given
    main_df["total_summons_dollars"] = (
        main_df["Summons Number"] * main_df["average_summons_amount"]
    )
    main_df = main_df.sort_values(by=["total_summons_dollars"], ascending=False)
    end = time.time()
    print("Calculate Total Summons Time: ", (end - start))
    print(main_df.head())
    return main_df


main_df = calculate_total_summons(main_df)


@bodo.jit(distributed=["main_df", "precinct_offenses_df"], cache=True)
def aggregate(main_df):
    """function that aggregates and filters data
    e.g. total violations by precinct
    """
    start = time.time()
    filtered_dataset = main_df[
        ["Violation Precinct", "Summons Number", "total_summons_dollars"]
    ]
    precinct_offenses_df = (
        filtered_dataset.groupby(by=["Violation Precinct"])
        .sum()
        .reset_index()
        .fillna(0)
    )
    end = time.time()
    print("Aggregate code time: ", (end - start))
    print(precinct_offenses_df.head())
    return precinct_offenses_df


precinct_offenses_df = aggregate(main_df)
