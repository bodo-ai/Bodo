import json
import argparse


def parse_plan(plan):
    plan = plan.split("\n")

    nodes = {}
    node = {"properties": {}}
    hit_cache_node = False
    for line in plan:
        indent = len(line) - len(line.lstrip())
        line = line.strip()

        is_cache_node = line.startswith("CACHED NODE")
        hit_cache_node |= is_cache_node
        if not is_cache_node and hit_cache_node:
            # Cache nodes are the same indent as their first child so we need to adjust the indent
            # to compensate
            indent += 2

        if line.startswith("#"):
            k, v = line[1:].split(":")
            node["properties"][k.strip()] = v.strip()
        else:
            node["name"] = line.strip()
            if not node["name"]:
                continue
            node["children"] = []

            if indent not in nodes:
                nodes[indent] = []

            if indent != 0:
                nodes[indent - 2][-1]["children"].append(node)
            nodes[indent].append(node)
            node = {"properties": {}}

    root = nodes[0][0]
    cache_nodes = []
    for node in nodes[0][1:]:
        cache_nodes.append(node)
    return {"root": root, "cache_nodes": cache_nodes}


def __main__():
    parser = argparse.ArgumentParser(description="Plan Parser")
    parser.add_argument(
        "-p", "--plan", type=str, help="Path to the input plan", required=True
    )
    parser.add_argument(
        "-o", "--out", type=str, help="Path to the output file", default="plan.json"
    )
    args = parser.parse_args()
    plan_path = args.plan
    out_path = args.out
    with open(plan_path) as f:
        plan = f.read()
    # parse the plan
    parsed_plan = parse_plan(plan)
    json.dump(parsed_plan, open(out_path, "w"))


if __name__ == "__main__":
    __main__()
