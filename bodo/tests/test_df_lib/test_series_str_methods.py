import re

import pandas as pd

from bodo.tests.test_df_lib.series_test_generator import generate_series_test


def _install_series_str_tests():
    """Install Series.str tests."""
    # Tests Series.str methods with arguments
    for method_name in test_map_arg:
        idx = 0
        for frame in exception_dfmap.get(method_name, (df,)):
            test = generate_series_test(
                method_name,
                frame,
                test_map_arg[method_name],
                accessor="str",
            )
            globals()[f"test_{method_name}_df{idx}"] = test
            idx += 1

    # Tests Series.str methods that require no arguments
    for method_name in test_map_no_arg:
        idx = 0
        for frame in exception_dfmap.get(method_name, (df,)):
            test = generate_series_test(
                method_name,
                frame,
                empty_arg,
                accessor="str",
            )
            globals()[f"test_{method_name}_df{idx}"] = test
            idx += 1


# Maps method name to test case for pytest param
# More rigorous testing NEEDED
test_map_arg = {
    "contains": [
        (("A",), {}),
        (("A",), {"na": False}),
        (("A",), {"na": True}),
        (("a",), {"case": False}),
        (("a",), {"case": False, "na": False}),
        (("A",), {"regex": False}),
    ],
    "startswith": [
        (("A",), {}),
        (("A",), {"na": False}),
        (("A",), {"na": None}),
        ((("A", "B"),), {}),
    ],
    "endswith": [
        (("e",), {}),
        (("e",), {"na": False}),
        (("e",), {"na": None}),
        ((("e", "d"),), {}),
    ],
    "find": [
        (("a",), {}),
        (("b",), {"start": 1}),
        (("D",), {"end": 3}),
    ],
    "rfind": [
        (("a",), {}),
        (("b",), {"start": 1}),
        (("D",), {"end": 3}),
    ],
    # TODO [BSE-4842] Below test case should correctly raise ValueError instead of stalling
    "index": [
        # (("e",), {}),
        # (("an",), {"start": 0}),
        # (("l",), {"end": 3}),
        (("",), {"end": 3}),
    ],
    # TODO [BSE-4842] Below test case should correctly raise ValueError instead of stalling
    "rindex": [
        # (("e",), {}),
        # (("an",), {"start": 0}),
        # (("l",), {"end": 3}),
        (("",), {"end": 3}),
    ],
    "replace": [
        (("a", "b"), {}),
        (("a", "b"), {"regex": False}),
        (("a", "b"), {"regex": True}),
    ],
    "match": [
        (("A",), {}),
        (("A",), {"na": False}),
        (("A",), {"na": None}),
    ],
    "fullmatch": [
        (("Apple",), {}),
        (("Apple",), {"na": False}),
    ],
    "get": [((0,), {}), ((-1,), {}), ((2,), {}), ((5,), {})],
    "slice": [
        ((), {"start": 1}),
        ((), {"start": 1, "stop": 3}),
        ((), {"step": 2}),
        ((), {"start": -1}),
    ],
    "slice_replace": [
        ((), {"start": 1}),
        ((), {"start": 1, "stop": 3, "repl": "oi"}),
        ((), {"stop": 4, "repl": "XXX"}),
        ((), {"start": -1}),
    ],
    "repeat": [
        ((2,), {}),
        ((5,), {}),
        ((0,), {}),
    ],
    "pad": [
        ((10,), {}),
        ((10,), {"side": "left"}),
        ((10,), {"side": "right", "fillchar": "-"}),
    ],
    "center": [
        ((10,), {}),
        ((10,), {"fillchar": "*"}),
        ((8,), {"fillchar": "."}),
    ],
    "ljust": [
        ((10,), {}),
        ((10,), {"fillchar": "*"}),
    ],
    "rjust": [
        ((10,), {}),
        ((10,), {"fillchar": "*"}),
    ],
    "zfill": [
        ((10,), {}),
    ],
    "wrap": [
        ((4,), {}),
        ((1,), {"drop_whitespace": False}),
    ],
    "removeprefix": [
        (("A",), {}),
        (("Do",), {}),
        (("B",), {}),
    ],
    "removesuffix": [
        (("anana",), {}),
        (("Dog",), {}),
        (("og",), {}),
        (("Canon",), {}),
    ],
    "translate": [
        ((str.maketrans("abc", "123"),), {}),
    ],
    "count": [
        (("an",), {}),
        (("Ex",), {}),
        (("a",), {}),
    ],
    "findall": [
        (("Banana",), {}),
        (("BANANA",), {"flags": re.IGNORECASE}),
    ],
    "partition": [
        ((), {"sep": "-", "expand": False}),
        ((), {"sep": "-", "expand": True}),
        ((), {"sep": "-"}),
    ],
    "rpartition": [
        ((), {"sep": "-", "expand": False}),
        ((), {"sep": "-", "expand": True}),
        ((), {"sep": "-"}),
    ],
    "normalize": [
        (("NFC",), {}),
        (("NFD",), {}),
    ],
    "join": [(("*",), {}), ((" and ",), {})],
    "decode": [
        (("ascii",), {}),
    ],
    "encode": [
        (("ascii",), {}),
    ],
    "split": [
        ((), {}),
        ((), {"n": 2}),
        ((), {"pat": "/"}),
        ((), {"expand": True}),
        ((), {"pat": "/", "expand": True}),
        ((), {"n": 1, "pat": "/", "expand": True}),
    ],
    "rsplit": [
        ((), {}),
        ((), {"n": 2}),
        ((), {"pat": "/"}),
        ((), {"expand": True}),
        ((), {"pat": "/", "expand": True}),
        ((), {"n": 1, "pat": "/", "expand": True}),
    ],
    "extract": [
        ((r"[ab](\d)",), {"expand": False, "flags": 0}),
        ((r"([ab])(\d)",), {"expand": False, "flags": 0}),
        ((r"(?P<letter>[ab])(?P<digit>\d)",), {"expand": False, "flags": 0}),
        ((r"([a-z]+)-(\d+)",), {"expand": True, "flags": 0}),
        ((r"(?P<letters>[a-z]+)-(?P<numbers>\d+)",), {"expand": True, "flags": 0}),
        ((r"([a-z]+)-(\d+)",), {"expand": True, "flags": re.IGNORECASE}),
        ((r"([a-z]+).(\d+)",), {"expand": True, "flags": re.DOTALL}),
        ((r"([a-z]+)-(\d+)",), {"expand": True, "flags": re.MULTILINE}),
    ],
}

# List of methods that do not take in arguments
test_map_no_arg = [
    "upper",
    "lower",
    "title",
    "swapcase",
    "capitalize",
    "casefold",
    "isalpha",
    "isnumeric",
    "isalnum",
    "isdigit",
    "isdecimal",
    "isspace",
    "islower",
    "isupper",
    "istitle",
    "len",
]

df = pd.DataFrame(
    {
        "A": [
            "Apple",
            "Banana",
            None,
            None,
            "App-le",
            "B-anan-a",
            " E-xc i-ted ",
            "Do-g",
        ],
    }
)

df_normalize = pd.DataFrame(
    {
        "A": ["ñ", "ñ", "n\u0303"],
        "B": ["Amélie", "Am\u00e9lie", "Am\u0065\u0301lie"],
        "C": ["\u00f1", "\u006e\u0303", "ñ"],
        "D": ["ñ", "ñ", None],
    }
)

df_join_tuple = (
    pd.DataFrame(
        {
            "A": [["h"], ["None", "Play"], ["Bad", "News", "A"]],
            "B": [["hoveroverrover"], None, None],
            "C": [["A", "B", "C", "D", "E"], None, ["California", "Sports", "Cars"]],
        }
    ),
    pd.DataFrame(
        {
            "A": ["hi", "my", "name", "0000"],
            "B": ["None", None, None, None],
            "C": ["A", "B", "C", "D"],
            "D": ["hover", "over", "rover", None],
        }
    ),
)

df_decode = pd.DataFrame({"A": [b"hi", b"()", b"hello my name is chris.", None]})

df_split = pd.DataFrame(
    {
        "A": [
            "this is a regular sentence",
            "https://docs.python.org/3/tutorial/index.html",
            None,
        ]
    },
    index=["Apple", "Banana", "Cheese"],
)

# Extract-related DataFrames grouped together
df_extract_tuple = (
    pd.DataFrame({"A": ["a1", "b2", "c3"]}, index=["x", "y", "z"]),
    pd.DataFrame({"A": ["abc-123", "def-456", "ghi-789"]}),
    pd.DataFrame({"A": ["abc-123", "no-match", "ghi-789"]}),
    pd.DataFrame({"A": ["nope", "still nope"]}),
    pd.DataFrame({"A": ["abc-123", "def-456"]}),
    pd.DataFrame({"A": ["foo123", None, "bar456", "789"]}, dtype="string"),
    pd.DataFrame({"A": ["abc123", "def456"]}),
    pd.DataFrame({"A": ["abc-123", "DEF-456", "Ghi-789"]}),
    pd.DataFrame({"A": ["abc\n123", "def\n456"]}),
    pd.DataFrame({"A": ["abc-123", "def-456"]}),
)

# Join-related DataFrames grouped together

# Final exception mapping
exception_dfmap = {
    "normalize": (df_normalize,),
    "join": df_join_tuple,
    "decode": (df_decode,),
    "split": (df_split,),
    "rsplit": (df_split,),
    "extract": df_extract_tuple,
}

empty_arg = [((), {})]

_install_series_str_tests()
