import argparse
import json
import jsonschema
from igraph import Graph
import plotly.graph_objects as go
import textwrap


bg_color = "#0e1833"
light_grey = "#abafb9"

plan_json_schema = json.load(open("plan_json_schema.json"))


def add_nodes(graph, node, seen_names):
    if node["name"] in seen_names:
        seen_names[node["name"]] += 1
        node["name"] += f"_{seen_names[node['name']]}"
    else:
        seen_names[node["name"]] = 1

    graph.add_vertex(name=node["name"], properties=node["properties"])
    for child in node["children"]:
        add_nodes(graph, child, seen_names)
        graph.add_edge(node["name"], child["name"])


def wrap_title(title):
    return "<br>".join(textwrap.wrap(title, width=50))


def generate_graph(plan):
    graph = Graph(directed=True)
    seen_nodes = {}
    add_nodes(graph, plan["root"], seen_nodes)
    for node in plan.get("cache_nodes", []):
        add_nodes(graph, node, seen_nodes)
    return graph


def generate_node_coords(layout):
    max_y = max([y for _, y in layout])
    node_x = [layout[i][0] for i in range(len(layout))]
    node_y = [2 * max_y - layout[i][1] for i in range(len(layout))]
    return node_x, node_y


def generate_edge_coords(graph, layout):
    max_y = max([y for _, y in layout])
    edges = [e.tuple for e in graph.es]
    edges_x, edges_y = [], []
    for edge in edges:
        x0, y0 = layout[edge[0]]
        x1, y1 = layout[edge[1]]
        edges_x.append([x0, x1])
        edges_y.append([2 * max_y - y0, 2 * max_y - y1])
    return edges_x, edges_y


def graph_plan(plan):
    # validate the plan
    jsonschema.validate(plan, plan_json_schema)
    # create a graph the from the plan
    graph = generate_graph(plan)

    # Calculate coordinates for the nodes and edges
    layout = graph.layout("rt")
    node_x, node_y = generate_node_coords(layout)
    edge_x, edge_y = generate_edge_coords(graph, layout)
    hover_data = [
        "<br>".join([f"{k}: {v}" for k, v in prop.items()])
        for prop in graph.vs["properties"]
    ]

    # Create the plotly figure
    fig = go.Figure()
    for i in range(len(edge_x)):
        fig.add_trace(
            go.Scatter(
                x=edge_x[i],
                y=edge_y[i],
                mode="lines",
                line=dict(color="#909296", width=3),
                hoverinfo="none",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(
                symbol="circle",
                size=75,
                color="#1fb000",
                line=dict(color=light_grey, width=0),
            ),
            text=[wrap_title(title) for title in graph.vs["name"]],
            textposition="middle left",
            customdata=hover_data,
            hovertemplate="%{customdata}<extra></extra>",
            opacity=1,
        )
    )
    axis = dict(
        showline=False,  # hide axis line, grid, ticklabels and  title
        zeroline=False,
        showgrid=False,
        showticklabels=False,
    )

    fig.update_layout(
        font_size=14,
        font_color=light_grey,
        showlegend=False,
        xaxis=axis,
        yaxis=axis,
        margin=dict(l=40, r=40, b=85, t=100),
        hovermode="closest",
        plot_bgcolor=bg_color,
        hoverlabel=dict(bgcolor=light_grey, font_size=16),
    )
    return fig


def __main__():
    parser = argparse.ArgumentParser(description="Plan Grapher")
    parser.add_argument(
        "-j", "--in_json", type=str, help="Path to the input json file", required=True
    )
    parser.add_argument(
        "-o",
        "--out_file",
        type=str,
        help="Path to the output directory",
        default="plan.html",
    )
    args = parser.parse_args()
    json_path = args.in_json
    out_file = args.out_file
    assert out_file.endswith(".html")

    plan = json.load(open(json_path))
    fig = graph_plan(plan)

    fig.write_html(out_file, include_plotlyjs="cdn")


if __name__ == "__main__":
    __main__()
