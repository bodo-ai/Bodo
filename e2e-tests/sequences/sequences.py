import argparse
import math
import zlib

import numba
import numpy as np
import pandas as pd

import bodo

"""
Sequences code translation from Scala Spark into Bodo Pandas.
This code begins with a text input file that seems to encode
a series of coordinate points, and performs analysis on those
points (seems to be computing a neighborhood and then performing
some transformations).

This code tests our support for nested datastructures (i.e. Series
or Dataframe of arrays), as the first few steps correspond to flatmap
operations on arrays of common points.
"""


CHOSEN_VALUE = np.int32(4194304)
CHOSEN_TYPES = (5, 11, 52, 56)


@bodo.jit
def descriptorToLong(x, y, z):
    return np.int64(z) << 59 | np.int64(x) << 29 | np.int64(y)


@bodo.jit
def input01input02ToDescriptorId(input01, input02, z):
    input01_radians = math.radians(input01)
    n = pow(2, z)
    x = np.int32((input02 + 180.0) / 360.0 * n)
    y = np.int32(
        (
            (
                1.0
                - math.log(
                    math.tan(input01_radians) + (1.0 / math.cos(input01_radians))
                )
                / math.pi
            )
            / 2.0
            * n
        )
    )
    return descriptorToLong(x, y, z)


@bodo.jit
def transformationFunction01(sequence, maxSeqLength):
    value_list = sequence.split(",")
    out_seqs = []
    for i in range(0, len(value_list), 5):
        if int(value_list[i + 4]) not in CHOSEN_TYPES:
            continue
        f1 = np.float32(value_list[i])
        f2 = np.float32(value_list[i + 1])
        f3 = np.float32(value_list[i + 2])
        new_seq = (
            input01input02ToDescriptorId(np.float64(f1), np.float64(f2), 24),
            np.float64(f3),
        )
        out_seqs.append(new_seq)

    out_nested_seq = []
    for j in range(0, len(out_seqs), maxSeqLength):
        out_nested_seq.append(out_seqs[j : j + maxSeqLength])
    return out_nested_seq


@bodo.jit
def getXFromDescriptor(descriptorId):
    return np.int32((descriptorId >> 29) & 536870911)


@bodo.jit
def getYFromDescriptor(descriptorId):
    return np.int32(descriptorId & 536870911)


@bodo.jit
def getZFromDescriptor(descriptorId):
    return np.int32((descriptorId >> 59) & 31)


@bodo.jit
def filterFunction(input01, input02):
    # Parse the input
    xInput01 = getXFromDescriptor(input01)
    yInput01 = getYFromDescriptor(input01)
    xInput02 = getXFromDescriptor(input02)
    yInput02 = getYFromDescriptor(input02)
    # Classify to be within neighborhood
    xDifference = np.abs(xInput02 - xInput01)
    yDifference = np.abs(yInput02 - yInput01)
    return (xDifference <= CHOSEN_VALUE) and (yDifference <= CHOSEN_VALUE)


@bodo.jit
def transformationFunction02(sequence):
    n = len(sequence)
    results = []
    for i in range(n):
        elem_i = sequence[i]
        for j in range(n):
            elem_j = sequence[j]
            if (elem_j[1] >= elem_i[1]) and filterFunction(elem_i[0], elem_j[0]):
                results.append((elem_i[0], elem_j[0], np.int8(0), np.float32(1.0)))
                results.append((elem_j[0], elem_i[0], np.int8(1), np.float32(1.0)))

    return results


@bodo.jit
def getLinearHash(input01, input02):
    xInput01 = getXFromDescriptor(input01)
    yInput01 = getYFromDescriptor(input01)
    xInput02 = getXFromDescriptor(input02)
    yInput02 = getYFromDescriptor(input02)
    linHashX = xInput02 - xInput01 + CHOSEN_VALUE
    linHashY = yInput02 - yInput01 + CHOSEN_VALUE
    linHash = (2 * CHOSEN_VALUE + 1) * linHashY + linHashX
    return np.int32(linHash)


@bodo.jit
def groupTransformation03(row):
    byte_value = row["byte_val"]
    hash_value = row["hash_val"]
    float_value = row["float_val"]
    zeros1 = ""
    zeros2 = ""
    ones1 = ""
    ones2 = ""
    if byte_value == 0:
        zeros1 = str(hash_value) + ","
        zeros2 = str(float_value) + ","
    elif byte_value == 1:
        ones1 = str(hash_value) + ","
        ones2 = str(float_value) + ","
    return pd.Series(
        (zeros1, zeros2, ones1, ones2), index=["zeros1", "zeros2", "ones1", "ones2"]
    )


@bodo.jit
def descriptorTransformation02(key):
    return (
        str(getXFromDescriptor(key))
        + ":"
        + str(getYFromDescriptor(key))
        + ":"
        + str(getZFromDescriptor(key))
    )


@bodo.jit
def convertToOutputString(columns):
    output_arr = []
    output_arr.append(descriptorTransformation02(columns[0]))
    output_arr.append("\t")
    output_arr.append("o:")
    # Cut off the trailing comma
    output_arr.append(columns["zeros1"][:-1])
    output_arr.append(":")
    # Cut off the trailing comma
    output_arr.append(columns["zeros2"][:-1])
    output_arr.append("\t")
    output_arr.append("i:")
    # Cut off the trailing comma
    output_arr.append(columns["ones1"][:-1])
    output_arr.append(":")
    # Cut off the trailing comma
    output_arr.append(columns["ones2"][:-1])
    return "".join(output_arr)


@bodo.jit(cache=True)
def sequences(in_file, chosen_type, out_file, maxseqlength):
    df = pd.read_csv(in_file, sep="$", header=None, names=["A"])
    S1 = df.A.apply(transformationFunction01, args=(maxseqlength,))
    S2 = S1.explode().dropna()
    S3 = S2.apply(transformationFunction02)
    S4 = S3.explode()
    df1 = S4.apply(
        lambda x: pd.Series(
            (x[0], x[1], x[2], x[3]), index=["id1", "id2", "byte_val", "float_val"]
        )
    )
    S5 = (
        df1[["id1", "id2"]]
        .apply(lambda x: getLinearHash(x[0], x[1]), axis=1)
        .rename("hash_val")
    )
    df2 = pd.concat([df1[["id1", "byte_val", "float_val"]], S5], axis=1).rename(
        columns={"0": "hash_val"}
    )
    df3 = df2.groupby(["id1", "hash_val", "byte_val"], as_index=False)[
        "float_val"
    ].sum()
    df4 = pd.concat(
        [
            df3["id1"],
            df3[["byte_val", "hash_val", "float_val"]].apply(
                groupTransformation03, axis=1
            ),
        ],
        axis=1,
    ).rename(columns={"0": "id"})
    df5 = df4.groupby("id", as_index=False)[
        ["zeros1", "zeros2", "ones1", "ones2"]
    ].sum()
    S6 = df5.apply(lambda x: convertToOutputString(x), axis=1)
    df6 = pd.DataFrame({"A": S6.values})
    df6.to_csv(out_file, header=False, index=None)
    # Compute a reduction to compute correctness. The strings should always be the same, only the order changes.
    # So any operation that commutes should work

    # TODO: Replace with Bodo code when binary arrays are supported.
    def f(series):
        with bodo.objmode(res="uint32"):
            res = g(series["A"])
        return res

    print(df6.apply(lambda x: f(x), axis=1).sum())


def g(string):
    return zlib.crc32(bytes(string[0], encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, help="input file", default="sample_1_rows.txt"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="output file", default="bodo_out.txt"
    )
    parser.add_argument("--chosentype", type=int, help="", default=0)
    parser.add_argument("--maxseqlength", type=int, help="", default=4194304)
    parser.add_argument("--require_cache", action="store_true", default=False)
    args = parser.parse_args()

    sequences(args.input, args.chosentype, args.output, args.maxseqlength)
    if args.require_cache and isinstance(sequences, numba.core.dispatcher.Dispatcher):
        assert (
            sequences._cache_hits[sequences.signatures[0]] == 1
        ), "ERROR: Bodo did not load from cache"
