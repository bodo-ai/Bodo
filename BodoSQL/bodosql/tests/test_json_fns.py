import pandas as pd
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture
def json_nested_array_data():
    """JSON data with arrays nested at various levels"""
    return '[["alpha",[["beta","gamma"],["delta"]],[["epsilon","zeta","eta"],[]],"theta","iota"],"kappa",[[["lambdba"],["mu","nu"]],[[],["xi","omicron","pi"]]],["rho",[["sigma"],["tau","upsilon"],"phi"],"chi",[["psi"],[]]],"omega"]'


@pytest.fixture
def json_nested_map_data():
    """JSON data with objects nested at various levels"""
    return '{"California": {\'Capital\': "Sacramento", "Senators": [{"First": "Dianne", \'Last\': "Feinstein"}, {"First": "Alex", \'Last\': "Padilla"}], "Bird": "quail", "Nickname": "golden"},"Georgia": {\'Capital\': "Atlanta", "Senators": [{"First": "Raphael", \'Last\': "Warnock"}, {"First": "Jon", \'Last\': "Ossof"}], "Bird": "thrasher", "Nickname": "peach"},"Texas": {\'Capital\': "Austin", "Senators": [{"First": "Ted", \'Last\': "Cruz"}, {"First": "John", \'Last\': "Cornyn"}], "Bird": "mockingbird", "Nickname": "lone star"}}'


@pytest.fixture
def json_senator_data(datapath):
    """Large sample JSON dataset with an array of objects"""
    with open(datapath("json_data/senators.txt"), "r") as f:
        return f.read()


@pytest.fixture
def json_extract_path_args(
    json_nested_array_data, json_nested_map_data, json_senator_data
):
    data = (
        [json_nested_array_data] * 12
        + [json_nested_map_data] * 12
        + [json_senator_data] * 4
    )
    path = [
        # Nested array tests
        "[0][0]",
        "[0][1][0][0]",
        "[0][1][1][0]",
        "[0][2][0][2]",
        "[0][4]",
        "[1]",
        "[2][1][1][2]",
        "[3][0]",
        "[3][1][0][0]",
        "[3][1][2]",
        "[3][3][0][0]",
        "[4]",
        # Nested map tests
        "California.Capital",
        '["California"]["Bird"]',
        'California["Nickname"]',
        '["Georgia"].Senators[1].First',
        '["Georgia"][\'Senators\'][0]["Last"]',
        "Georgia.Nickname",
        'Texas["Capital"]',
        "Texas.Senators[0].Last",
        "Texas.Bird",
        # Invalid Tests
        'Texas["Tree"]',
        "Alaska.Nickname",
        "Texas.Bird.Genus",
        # Senator data tests
        "objects[12].person.name",
        "objects[99].extra.office",
        "['meta'][\"limit\"]",
        '["objects"][67].person["twitterid"]',
    ]
    answer = [
        # Nested array tests
        "alpha",
        "beta",
        "delta",
        "eta",
        "iota",
        "kappa",
        "pi",
        "rho",
        "sigma",
        "phi",
        "psi",
        "omega",
        # Nested map tests
        "Sacramento",
        "quail",
        "golden",
        "Jon",
        "Warnock",
        "peach",
        "Austin",
        "Cruz",
        "mockingbird",
        # Invalid Tests
        None,
        None,
        None,
        # Senator data tests
        "Sen. Mazie Hirono [D-HI]",
        "B40b Dirksen Senate Office Building",
        "100",
        "JohnBoozman",
    ]
    return (data, path, answer)


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
    ],
)
def test_json_extract_path_text(json_extract_path_args, use_case, memory_leak_check):
    data, path, answer = json_extract_path_args
    if use_case:
        query = "SELECT I, CASE WHEN B THEN NULL ELSE JSON_EXTRACT_PATH_TEXT(D, P) END FROM TABLE1"
    else:
        query = "SELECT I, JSON_EXTRACT_PATH_TEXT(D, P) FROM TABLE1"
    ctx = {
        "table1": pd.DataFrame(
            {
                "I": range(len(data)),
                "D": data,
                "P": path,
                "B": [i % 4 == 3 for i in range(len(data))],
            }
        )
    }
    expected_output = pd.DataFrame({0: range(len(data)), 1: answer})
    if use_case:
        expected_output[1][ctx["table1"].B] = None
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )
