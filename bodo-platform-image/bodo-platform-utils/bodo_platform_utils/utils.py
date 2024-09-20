import json

from .type_convertor import get_value_for_type


def read_sql_query(filename: str):
    with open(filename, "r") as f:
        return f.read()


def read_sql_query_param(filename: str):
    if not filename:
        return tuple()

    with open(filename, "r") as f:
        query_params_dict = json.loads(f.read())

    query_params = []
    for i in range(0, len(query_params_dict) + 1):
        query_param = query_params_dict.get(str(i))

        if query_param is not None:
            query_type = get_value_for_type(query_param)
            query_params.append(query_type)

    return tuple(query_params)
