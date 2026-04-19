import os
import random
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import networkx as nx
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

from network import G


# Hub centrality analysis:
# We hypothesize that betweenness centrality is concentrated in a small number of interchange hubs
# (places like Chicago, New York Penn Station, Washington Union Station), and that removing the top ~5 stations
# by betweenness would reduce the size of the largest connected component far more severely than removing a similar number
# of random stations. If true, this would mean that the Amtrak system is vulnerable to coordinated attack,
# but not so much to random failure.


def largest_component_size(graph):
    """Return the number of nodes in the largest connected component."""
    if graph.number_of_nodes() == 0:
        return 0
    return max(len(c) for c in nx.connected_components(graph))


def compute_hub_centrality(graph, top_n=10):
    """Compute betweenness centrality and return the top_n stations sorted descending.

    Edge ``weight`` encodes ridership (higher = busier corridor). NetworkX's
    betweenness treats ``weight`` as a distance/cost for Dijkstra, so we
    invert ridership into an edge cost: busy corridors become "short" and
    shortest paths preferentially route through high-ridership segments.
    """
    for u, v, d in graph.edges(data=True):
        d["cost"] = 1.0 / d["weight"] if d.get("weight", 0) > 0 else float("inf")
    centrality = nx.betweenness_centrality(graph, weight="cost")
    ranked = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n], centrality


def targeted_removal(graph, centrality, max_removals):
    """Remove stations in order of decreasing betweenness centrality.
    Returns list of largest-component sizes after each removal."""
    ranked_nodes = sorted(centrality, key=centrality.get, reverse=True)
    g = graph.copy()
    sizes = [largest_component_size(g)]

    for node in ranked_nodes[:max_removals]:
        g.remove_node(node)
        sizes.append(largest_component_size(g))

    return sizes


def random_removal(graph, max_removals, trials=100):
    """Remove stations at random, averaged over multiple trials.
    Returns list of mean largest-component sizes after each removal."""
    all_nodes = list(graph.nodes)
    cumulative = [0.0] * (max_removals + 1)

    for _ in range(trials):
        g = graph.copy()
        sample = random.sample(all_nodes, min(max_removals, len(all_nodes)))
        cumulative[0] += largest_component_size(g)

        for i, node in enumerate(sample):
            g.remove_node(node)
            cumulative[i + 1] += largest_component_size(g)

    return [s / trials for s in cumulative]


def plot_robustness(targeted_sizes, random_sizes, max_removals, centrality, output_path=None):
    """Plot targeted vs random removal curves with annotations."""
    x = list(range(max_removals + 1))
    ranked_nodes = sorted(centrality, key=centrality.get, reverse=True)[:max_removals]
    initial = targeted_sizes[0]

    fig, ax = plt.subplots(figsize=(12, 7))

    # Shade the gap between curves
    ax.fill_between(x, targeted_sizes, random_sizes, alpha=0.12, color="#e74c3c",
                     label="_Vulnerability gap")

    ax.plot(x, random_sizes, "s--", color="#2980b9", markersize=7, linewidth=2,
            label="Random removal (avg 100 trials)")
    ax.plot(x, targeted_sizes, "o-", color="#c0392b", markersize=8, linewidth=2.5,
            label="Targeted removal (by betweenness)")

    # Annotate each targeted removal step with station code.
    # Positions tuned for new top-10: CHI, CLE, SOB, EKH, WTI, TOL, SKY, ELY, BUF, SDY
    # Ranks 2-8 all sit at y=491 (flat plateau), so alternate above/below.
    # Rank 10 (SDY) is near the bottom edge — push label clearly above.
    station_offsets = [
        (0, 22),     # 1 CHI  — above (95% drop)
        (-8, -20),   # 2 CLE  — below (plateau)
        (0, 22),     # 3 SOB  — above (plateau)
        (0, -20),    # 4 EKH  — below (plateau)
        (0, 22),     # 5 WTI  — above (plateau)
        (0, -20),    # 6 TOL  — below (plateau)
        (0, 22),     # 7 SKY  — above (plateau)
        (8, -20),    # 8 ELY  — below (plateau)
        (-26, 18),   # 9 BUF  — left-above (92% drop)
        (-28, 22),   # 10 SDY — left-above (far from bottom)
    ]
    for i, node in enumerate(ranked_nodes):
        pct = targeted_sizes[i + 1] / initial * 100
        y_val = targeted_sizes[i + 1]
        ox, oy = station_offsets[i] if i < len(station_offsets) else (0, 20)
        ax.annotate(
            f"{node} ({pct:.0f}%)",
            xy=(i + 1, y_val), fontsize=7.5, fontweight="bold",
            ha="center", color="#8b0000",
            textcoords="offset points", xytext=(ox, oy),
            arrowprops=dict(arrowstyle="->", color="#bbbbbb", lw=0.8,
                            connectionstyle="arc3,rad=0.2" if ox != 0 else "arc3,rad=0"),
        )

    # Add a secondary y-axis showing percentage
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim()[0] / initial * 100, ax.get_ylim()[1] / initial * 100)
    ax2.set_ylabel("% of original network", fontsize=12, color="#666666")
    ax2.tick_params(colors="#666666")

    ax.set_xlabel("Number of stations removed", fontsize=13)
    ax.set_ylabel("Largest connected component size", fontsize=13)
    ax.set_title("Amtrak Network Robustness: Targeted vs Random Station Removal", fontsize=15, pad=12)
    ax.legend(fontsize=11, loc="lower left")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def plot_degree_distribution(graph, output_path=None):
    """Bar chart of station degree distribution with count labels."""
    degree_seq = [d for _, d in graph.degree()]
    degree_counts = Counter(degree_seq)
    degrees = sorted(degree_counts)
    frequencies = [degree_counts[d] for d in degrees]
    total = sum(frequencies)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(degrees, frequencies, color="#2c3e50", edgecolor="white", linewidth=0.8, zorder=3)

    # Label each bar with count and percentage
    for bar, freq in zip(bars, frequencies):
        pct = freq / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{freq}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Degree (number of connections)", fontsize=13)
    ax.set_ylabel("Number of stations", fontsize=13)
    ax.set_title("Station Degree Distribution", fontsize=15)
    ax.set_xticks(degrees)
    ax.set_xticklabels([str(d) for d in degrees])
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_axisbelow(True)

    # Add interpretation text
    pct_pass = sum(degree_counts.get(d, 0) for d in [1, 2]) / total * 100
    ax.text(0.97, 0.95, f"{pct_pass:.0f}% of stations have\ndegree \u2264 2 (pass-through)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc"))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def plot_centrality_distribution(centrality, top_n=10, output_path=None):
    """Histogram of betweenness centrality with annotated outliers."""
    values = list(centrality.values())
    ranked = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    threshold = ranked[top_n - 1][1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(values, bins=60, color="#2980b9", edgecolor="white", linewidth=0.5)

    # Mark the top-N threshold
    ax.axvline(threshold, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(threshold + 0.005, ax.get_ylim()[1] * 0.85, f"Top {top_n} threshold",
            fontsize=9, color="#e74c3c", fontstyle="italic")

    # Annotate the top 3 outliers with staggered heights
    outlier_offsets = [(0, 45), (0, 30), (0, 15)]
    for idx, (station, score) in enumerate(ranked[:3]):
        ox, oy = outlier_offsets[idx]
        ax.annotate(
            station, xy=(score, 1), fontsize=9, fontweight="bold", color="#c0392b",
            ha="center", va="bottom",
            textcoords="offset points", xytext=(ox, oy),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
        )

    ax.set_xlabel("Betweenness centrality", fontsize=13)
    ax.set_ylabel("Number of stations", fontsize=13)
    ax.set_title("Betweenness Centrality Distribution", fontsize=15)
    ax.grid(True, alpha=0.2, axis="y")

    # Inset text summarizing the skew
    median_val = np.median(values)
    max_val = max(values)
    ax.text(0.97, 0.95, f"Median: {median_val:.4f}\nMax: {max_val:.3f} ({max_val/median_val:.0f}\u00d7 median)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc"))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def plot_geographic_hubs(graph, centrality, top_n=10, output_path=None):
    """Map stations sized/colored by betweenness centrality, labeling top hubs."""
    if not HAS_CARTOPY:
        print("Skipping geographic hub map (cartopy not installed)")
        return None, None

    pos = {
        node: (attrs["lon"], attrs["lat"])
        for node, attrs in graph.nodes(data=True)
        if "lon" in attrs and "lat" in attrs
    }
    ranked = sorted(centrality, key=centrality.get, reverse=True)
    top_hubs = set(ranked[:top_n])

    projection = ccrs.LambertConformal()
    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=projection)

    xs = [c[0] for c in pos.values()]
    ys = [c[1] for c in pos.values()]
    lon_margin = max((max(xs) - min(xs)) * 0.12, 2.0)
    lat_margin = max((max(ys) - min(ys)) * 0.12, 1.5)
    ax.set_extent(
        (min(xs) - lon_margin, max(xs) + lon_margin,
         min(ys) - lat_margin, max(ys) + lat_margin),
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.LAND, facecolor="#f3efe6")
    ax.add_feature(cfeature.OCEAN, facecolor="#dbe9f4")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.STATES, linewidth=0.25, edgecolor="#999999")

    # Draw edges with slight weight visibility
    edge_weights = [graph[u][v].get("weight", 1) for u, v in graph.edges() if u in pos and v in pos]
    max_ew = max(edge_weights) if edge_weights else 1
    for u, v in graph.edges():
        if u in pos and v in pos:
            w = graph[u][v].get("weight", 1)
            lw = 0.3 + 1.5 * (w / max_ew)
            ax.plot(
                [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#888888", linewidth=lw, alpha=0.4,
                transform=ccrs.PlateCarree(), zorder=1,
            )

    # Draw all non-hub stations
    non_hub_nodes = [n for n in pos if n not in top_hubs]
    if non_hub_nodes:
        ax.scatter(
            [pos[n][0] for n in non_hub_nodes],
            [pos[n][1] for n in non_hub_nodes],
            s=10, color="#bbbbbb", linewidths=0.3, edgecolors="#888888",
            transform=ccrs.PlateCarree(), zorder=2,
        )

    # Draw hub stations scaled by centrality
    hub_nodes = [n for n in ranked[:top_n] if n in pos]
    if hub_nodes:
        c_values = [centrality[n] for n in hub_nodes]
        max_c = max(c_values)
        sizes = [100 + 500 * (c / max_c) for c in c_values]
        hub_colors = [plt.get_cmap("YlOrRd")(0.35 + 0.6 * (c / max_c)) for c in c_values]

        ax.scatter(
            [pos[n][0] for n in hub_nodes],
            [pos[n][1] for n in hub_nodes],
            s=sizes, c=hub_colors, edgecolors="black", linewidths=1.0,
            transform=ccrs.PlateCarree(), zorder=4,
        )

        # Offset labels with leader lines. 8 of the top 10 are packed into
        # a ~6° longitude band along the IN/OH corridor (lat ~41.5). Fan
        # labels above and below the corridor, staggered by rank, so
        # leaders separate clearly.
        label_offsets = {
            0: (-6.0, 3.0),   # CHI  (lon=-87.6) — NW anchor, up-left
            1: (3.5, -5.5),   # CLE  (lon=-81.7) — down-right (east end of cluster)
            2: (-3.0, 5.5),   # SOB  (lon=-86.3) — up
            3: (-1.5, -5.5),  # EKH  (lon=-86.0) — down
            4: (0.5, 6.5),    # WTI  (lon=-85.0) — up (higher)
            5: (1.0, -6.5),   # TOL  (lon=-83.5) — down (lower)
            6: (3.0, 7.0),    # SKY  (lon=-82.7) — up-right (highest)
            7: (0.0, -7.5),   # ELY  (lon=-82.1) — down (lowest)
            8: (2.0, 3.5),    # BUF  (lon=-78.7) — NE
            9: (3.0, -4.5),   # SDY  (lon=-73.9) — down-right
        }
        for i, n in enumerate(hub_nodes):
            dx, dy = label_offsets.get(i, (3.0, 1.5))
            rank = i + 1
            ax.annotate(
                f"#{rank} {n}",
                xy=(pos[n][0], pos[n][1]),
                xytext=(pos[n][0] + dx, pos[n][1] + dy),
                fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0,
                                connectionstyle="arc3,rad=0.2"),
                transform=ccrs.PlateCarree(), zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#333333",
                          alpha=0.9),
            )

    ax.set_title("Amtrak Hub Stations by Betweenness Centrality", fontsize=18, pad=14)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def plot_centrality_vs_degree(graph, centrality, top_n=10, output_path=None):
    """Scatter of betweenness centrality vs degree with jitter — reveals bottleneck stations."""
    nodes = list(graph.nodes())
    degrees = np.array([graph.degree(n) for n in nodes], dtype=float)
    cent_values = np.array([centrality[n] for n in nodes])

    ranked = sorted(centrality, key=centrality.get, reverse=True)
    top_hubs = set(ranked[:top_n])

    # Add horizontal jitter for discrete degree values
    rng = np.random.default_rng(42)
    jitter = rng.uniform(-0.25, 0.25, size=len(degrees))
    jittered_degrees = degrees + jitter

    fig, ax = plt.subplots(figsize=(10, 7))

    # Shade the "bottleneck" quadrant (low degree, high centrality)
    median_deg = np.median(degrees)
    cent_75 = np.percentile(cent_values, 90)
    ax.axhspan(cent_75, ax.get_ylim()[1] if ax.get_ylim()[1] > cent_75 else 0.6,
               xmax=0.35, alpha=0.06, color="#e74c3c")
    ax.text(1.3, cent_75 + 0.01, "Bottleneck zone\n(low degree, high centrality)",
            fontsize=8, color="#c0392b", fontstyle="italic", alpha=0.7)

    # Plot all non-hub nodes
    non_hub_mask = np.array([n not in top_hubs for n in nodes])
    ax.scatter(jittered_degrees[non_hub_mask], cent_values[non_hub_mask],
              s=20, alpha=0.4, color="#7f8c8d", zorder=2)

    # Plot and label top hubs. New top-10 degrees: CHI=9, CLE=3, SOB=2,
    # EKH=2, WTI=3, TOL=3, SKY=2, ELY=2, BUF=3, SDY=4. Most land in the
    # degree=2–3 column around centrality ~0.43, so spread labels radially.
    hub_offsets = {
        0: (-35, 5),    # CHI  — left of marker (it's at far right, degree 9)
        1: (28, 14),    # CLE  — up-right (degree 3 cluster)
        2: (-34, 14),   # SOB  — up-left (degree 2 cluster)
        3: (-34, -14),  # EKH  — down-left (degree 2 cluster)
        4: (-22, -24),  # WTI  — down-left (degree 3 cluster)
        5: (34, -8),    # TOL  — right (degree 3 cluster)
        6: (-22, -22),  # SKY  — down-left (degree 2 cluster)
        7: (-38, -5),   # ELY  — far left (degree 2 cluster)
        8: (20, 22),    # BUF  — up-right (degree 3 cluster)
        9: (24, -18),   # SDY  — down-right (degree 4, more isolated)
    }
    for i, hub in enumerate(ranked[:top_n]):
        idx = nodes.index(hub)
        jx = jittered_degrees[idx]
        cy = cent_values[idx]
        ax.scatter(jx, cy, s=100, color="#e74c3c", edgecolors="black",
                   linewidths=0.8, zorder=4)
        ox, oy = hub_offsets.get(i, (14, 10))
        ax.annotate(
            hub, (jx, cy), textcoords="offset points", xytext=(ox, oy),
            fontsize=8.5, fontweight="bold", color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8,
                            connectionstyle="arc3,rad=0.15"),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#dddddd", alpha=0.9),
        )

    ax.set_xlabel("Degree (number of connections)", fontsize=13)
    ax.set_ylabel("Betweenness centrality", fontsize=13)
    ax.set_title("Betweenness Centrality vs Degree", fontsize=15)
    ax.grid(True, alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def plot_edge_weight_distribution(graph, output_path=None):
    """Histogram of edge weights (ridership) with log scale to reveal the long tail."""
    edge_data = [(u, v, d["weight"]) for u, v, d in graph.edges(data=True)]
    weights = [w for _, _, w in edge_data]
    edge_data_sorted = sorted(edge_data, key=lambda x: x[2], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    log_weights = np.log10(np.array(weights) + 1)
    ax.hist(log_weights, bins=50, color="#e67e22", edgecolor="white", linewidth=0.5)

    # Custom x-axis labels showing original values
    tick_vals = [0, 0.5, 1.0, 1.5, 2.0, 2.5]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"{10**v:.0f}" for v in tick_vals])

    # Annotate top 5 heaviest corridors
    top_edges = edge_data_sorted[:5]
    y_max = ax.get_ylim()[1]
    for i, (u, v, w) in enumerate(top_edges):
        ax.annotate(
            f"{u}\u2194{v}: {w:.0f}",
            xy=(np.log10(w + 1), 1), fontsize=7.5, fontweight="bold",
            color="#8b4513", ha="center", va="bottom",
            textcoords="offset points", xytext=(0, 5 + i * 14),
            arrowprops=dict(arrowstyle="->", color="#8b4513", lw=0.8) if i == 0 else None,
        )

    ax.set_xlabel("Ridership per segment (log scale)", fontsize=13)
    ax.set_ylabel("Number of edges", fontsize=13)
    ax.set_title("Edge Weight (Ridership) Distribution", fontsize=15)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_axisbelow(True)

    # Summary stat
    median_w = np.median(weights)
    max_w = max(weights)
    ax.text(0.97, 0.95, f"Median: {median_w:.1f}\nMax: {max_w:.0f} ({max_w/median_w:.0f}\u00d7 median)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc"))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def print_top_hubs(top_hubs):
    """Print a ranked table of the most central stations."""
    print(f"\n{'Rank':<6}{'Station':<12}{'Betweenness Centrality':<24}")
    print("-" * 42)
    for i, (station, score) in enumerate(top_hubs, 1):
        print(f"{i:<6}{station:<12}{score:<24.6f}")


if __name__ == "__main__":
    MAX_REMOVALS = 10

    print(f"Network: {G.number_of_nodes()} stations, {G.number_of_edges()} edges")
    print(f"Initial largest component: {largest_component_size(G)} stations\n")

    top_hubs, centrality = compute_hub_centrality(G, top_n=MAX_REMOVALS)
    print_top_hubs(top_hubs)

    print(f"\nRunning targeted removal (top {MAX_REMOVALS} by betweenness)...")
    targeted_sizes = targeted_removal(G, centrality, MAX_REMOVALS)

    print("Running random removal (100 trials)...")
    random_sizes = random_removal(G, MAX_REMOVALS, trials=100)

    print(f"\nAfter removing {MAX_REMOVALS} targeted stations: "
          f"largest component = {targeted_sizes[-1]} "
          f"({targeted_sizes[-1] / targeted_sizes[0] * 100:.1f}% of original)")
    print(f"After removing {MAX_REMOVALS} random stations:   "
          f"largest component = {random_sizes[-1]:.0f} "
          f"({random_sizes[-1] / random_sizes[0] * 100:.1f}% of original)")

    out = "graphs"
    os.makedirs(out, exist_ok=True)

    print("\nGenerating plots...")
    plot_robustness(targeted_sizes, random_sizes, MAX_REMOVALS, centrality, output_path=f"{out}/hub-robustness.png")
    plot_degree_distribution(G, output_path=f"{out}/degree-distribution.png")
    plot_centrality_distribution(centrality, top_n=MAX_REMOVALS, output_path=f"{out}/centrality-distribution.png")
    plot_geographic_hubs(G, centrality, top_n=MAX_REMOVALS, output_path=f"{out}/geographic-hubs.png")
    plot_centrality_vs_degree(G, centrality, top_n=MAX_REMOVALS, output_path=f"{out}/centrality-vs-degree.png")
    plot_edge_weight_distribution(G, output_path=f"{out}/edge-weight-distribution.png")
    print(f"All plots saved to {out}/.")


