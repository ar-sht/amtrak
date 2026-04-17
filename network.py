import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

routes = pd.read_excel("route-data.xlsx")["name"]
stations = pd.read_excel("station-data.xlsx")
station_coords = (
    stations.dropna(subset=["Code", "lon", "lat"])
    .drop_duplicates(subset=["Code"])
    .set_index("Code")[["lon", "lat"]]
    .to_dict("index")
)

with open("route-stop-data.txt", "r") as route_stop_txt:
    txt = route_stop_txt.read()

    route_stop_dict = dict([(route, []) for route in routes])

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
        stop_codes = [
            re.search(r"\(.*?\)", stop_txt, flags=re.DOTALL)
            .group(0)
            .replace("(", "")
            .replace(")", "")
            for stop_txt in stop_txts[1:]
        ]
        route_stop_dict[route] = stop_codes

with open("ridership-data.txt", "r") as ridership_txt:
    txt = ridership_txt.read().split("\n")

    route_weight_dict = dict([(route, []) for route in routes])

    for row in txt:
        match = re.match(
            r"^(.*?)\s+((?:\(?[\d,.-]+\)?|-)(?:\s+(?:\(?[\d,.-]+\)?|-))*)$",
            row.strip(),
        )
        if not match:
            continue

        route_name = match.group(1)
        metrics = match.group(2).split()

        route_weight_dict[route_name] = float(metrics[4].replace(",", ""))


G = nx.Graph()

route_graphs = dict([(route, None) for route in routes])

for route in routes:
    stops = route_stop_dict[route]
    weight = route_weight_dict[route]
    route_graph = nx.Graph()

    e = []
    for i in range(len(stops) - 1):
        e.append((stops[i], stops[i + 1], weight))

    route_graph.add_weighted_edges_from(e)
    route_graphs[route] = route_graph

    for u, v, weight in e:
        if G.has_edge(u, v):
            G[u][v]["weight"] += weight
        else:
            G.add_edge(u, v, weight=weight)

for node in G.nodes:
    coord = station_coords.get(node)
    if coord:
        G.nodes[node]["lon"] = float(coord["lon"])
        G.nodes[node]["lat"] = float(coord["lat"])

pos = {
    node: (attrs["lon"], attrs["lat"])
    for node, attrs in G.nodes(data=True)
    if "lon" in attrs and "lat" in attrs
}

missing_nodes = sorted(set(G.nodes) - set(pos))
if missing_nodes:
    print("Missing coordinates for:", ", ".join(missing_nodes))

fig = plt.figure(figsize=(16, 10))
ax = plt.axes(projection=ccrs.LambertConformal())
ax.set_extent([-130, -60, 20, 60], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="#f3efe6")
ax.add_feature(cfeature.OCEAN, facecolor="#dbe9f4")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.6)
ax.add_feature(cfeature.STATES, linewidth=0.25, edgecolor="#999999")
ax.gridlines(
    draw_labels=False,
    linewidth=0.3,
    color="#999999",
    alpha=0.4,
    linestyle="--",
)

if pos:
    for u, v, attrs in G.edges(data=True):
        if u in pos and v in pos:
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color="#b03a2e",
                alpha=0.35,
                linewidth=max(attrs["weight"] / 400, 0.4),
                transform=ccrs.PlateCarree(),
            )

    xs = [coords[0] for coords in pos.values()]
    ys = [coords[1] for coords in pos.values()]
    ax.scatter(
        xs,
        ys,
        s=12,
        color="#1f4e79",
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

ax.set_title("Amtrak Network by Station Geography", fontsize=18, pad=14)
plt.tight_layout()
plt.savefig("network-map.png", dpi=300, bbox_inches="tight")
plt.show()
