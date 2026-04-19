import matplotlib.pyplot as plt
from matplotlib import cm, colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def display_graph(
    graph,
    output_path=None,
    title="Amtrak Network by Station Geography",
    extent=None,
    projection=None,
    node_size=12,
    node_color="#1f4e79",
    edge_alpha=0.7,
    min_edge_width=0.6,
    max_edge_width=3.0,
    show=True,
):
    projection = projection or ccrs.LambertConformal()
    pos = {
        node: (attrs["lon"], attrs["lat"])
        for node, attrs in graph.nodes(data=True)
        if "lon" in attrs and "lat" in attrs
    }

    missing_nodes = sorted(set(graph.nodes) - set(pos))
    if missing_nodes:
        print("Missing coordinates for:", ", ".join(missing_nodes))

    if extent is None:
        if pos:
            xs = [coords[0] for coords in pos.values()]
            ys = [coords[1] for coords in pos.values()]
            min_lon, max_lon = min(xs), max(xs)
            min_lat, max_lat = min(ys), max(ys)

            lon_margin = max((max_lon - min_lon) * 0.12, 2.0)
            lat_margin = max((max_lat - min_lat) * 0.12, 1.5)
            extent = (
                min_lon - lon_margin,
                max_lon + lon_margin,
                min_lat - lat_margin,
                max_lat + lat_margin,
            )
        else:
            extent = (-130, -60, 20, 60)

    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=projection)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
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

    edge_data = [
        (u, v, attrs.get("weight", 1))
        for u, v, attrs in graph.edges(data=True)
        if u in pos and v in pos
    ]
    edge_weights = [weight for _, _, weight in edge_data]
    if edge_weights:
        weight_norm = colors.LogNorm(
            vmin=max(min(edge_weights), 1e-6), vmax=max(edge_weights)
        )
        edge_cmap = cm.get_cmap("YlOrRd")
    else:
        weight_norm = None
        edge_cmap = None

    for u, v, weight in sorted(edge_data, key=lambda edge: edge[2]):
        if weight_norm is not None:
            normalized = weight_norm(weight)
            color = edge_cmap(0.35 + 0.6 * normalized)
            width = min_edge_width + (max_edge_width - min_edge_width) * (
                normalized ** 0.7
            )
        else:
            color = "#b03a2e"
            width = min_edge_width

        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=color,
            alpha=edge_alpha,
            linewidth=width,
            solid_capstyle="round",
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

    if pos:
        xs = [coords[0] for coords in pos.values()]
        ys = [coords[1] for coords in pos.values()]
        ax.scatter(
            xs,
            ys,
            s=node_size,
            color=node_color,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

    ax.set_title(title, fontsize=18, pad=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax
