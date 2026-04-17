import pandas as pd
import re
import networkx as nx


def load_station_coords(path="station-data.xlsx"):
    stations = pd.read_excel(path)
    return (
        stations.dropna(subset=["Code", "lon", "lat"])
        .drop_duplicates(subset=["Code"])
        .set_index("Code")[["lon", "lat"]]
        .to_dict("index")
    )


def load_route_stops(routes, path="route-stop-data.txt"):
    with open(path, "r") as route_stop_txt:
        txt = route_stop_txt.read()

    route_stop_dict = {route: [] for route in routes}

    for route in routes:
        route_pattern = re.escape(route)
        stop_txts = (
            re.search(
                rf"Train Stations Served by (?:the )?{route_pattern}\n(.*?)(?=\nTrain Stations Served by|\Z)",
                txt,
                flags=re.DOTALL,
            )
            .group(0)
            .split("\n")
        )
        route_stop_dict[route] = [
            re.search(r"\(.*?\)", stop_txt, flags=re.DOTALL)
            .group(0)
            .replace("(", "")
            .replace(")", "")
            for stop_txt in stop_txts[1:]
        ]

    return route_stop_dict


def load_route_weights(routes, path="ridership-data.txt"):
    with open(path, "r") as ridership_txt:
        rows = ridership_txt.read().split("\n")

    route_weight_dict = {route: [] for route in routes}

    for row in rows:
        match = re.match(
            r"^(.*?)\s+((?:\(?[\d,.-]+\)?|-)(?:\s+(?:\(?[\d,.-]+\)?|-))*)$",
            row.strip(),
        )
        if not match:
            continue

        route_name = match.group(1)
        metrics = match.group(2).split()
        route_weight_dict[route_name] = float(metrics[4].replace(",", ""))

    return route_weight_dict


def annotate_graph_with_coords(graph, station_coords):
    missing_nodes = []
    for node in graph.nodes:
        coord = station_coords.get(node)
        if coord:
            graph.nodes[node]["lon"] = float(coord["lon"])
            graph.nodes[node]["lat"] = float(coord["lat"])
        else:
            missing_nodes.append(node)
    return missing_nodes


def build_network():
    routes = pd.read_excel("route-data.xlsx")["name"].tolist()
    station_coords = load_station_coords()
    route_stop_dict = load_route_stops(routes)
    route_weight_dict = load_route_weights(routes)

    graph = nx.Graph()

    for route in routes:
        stops = route_stop_dict[route]
        weight = route_weight_dict[route] / len(stops)
        edges = []

        for i in range(len(stops) - 1):
            edges.append((stops[i], stops[i + 1], weight))

        for u, v, edge_weight in edges:
            if graph.has_edge(u, v):
                graph[u][v]["weight"] += edge_weight
            else:
                graph.add_edge(u, v, weight=edge_weight)

    route_graphs = {}
    for route in routes:
        stops = route_stop_dict[route]
        edges = []

        for i in range(len(stops) - 1):
            edges.append(
                (stops[i], stops[i + 1], graph[stops[i]][stops[i + 1]]["weight"])
            )

        G = nx.Graph()
        G.add_weighted_edges_from(edges)

        route_graphs[route] = G

    missing_nodes = annotate_graph_with_coords(graph, station_coords)
    route_missing_nodes = {}

    for route, route_graph in route_graphs.items():
        route_missing_nodes[route] = annotate_graph_with_coords(
            route_graph, station_coords
        )

    return graph, route_graphs, missing_nodes, route_missing_nodes


G, route_graphs, missing_nodes, route_missing_nodes = build_network()


if __name__ == "__main__":
    from display import display_graph

    if missing_nodes:
        print("Missing coordinates for:", ", ".join(sorted(missing_nodes)))

    display_graph(G, output_path="network-map.png")
