from __future__ import annotations

import math
import random
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

from network import G


def load_station_state_lookup(path="station-data.xlsx"):
    station_data = pd.read_excel(path)
    return (
        station_data.dropna(subset=["Code", "State"])
        .drop_duplicates(subset=["Code"])
        .set_index("Code")["State"]
        .to_dict()
    )


def haversine_miles(lat1, lon1, lat2, lon2):
    radius_miles = 3958.7613
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * radius_miles * math.asin(min(1.0, math.sqrt(a)))


def detect_communities(graph, weight="weight", seed=42, resolution=1):
    """Detect graph communities without using geographic metadata."""
    if hasattr(nx.community, "louvain_communities"):
        communities = nx.community.louvain_communities(
            graph,
            weight=weight,
            seed=seed,
            resolution=resolution,
        )
        method = "louvain"
    else:
        communities = nx.community.greedy_modularity_communities(graph, weight=weight)
        method = "greedy_modularity"

    communities = sorted((set(c) for c in communities), key=len, reverse=True)
    modularity = nx.community.modularity(graph, communities, weight=weight)
    return communities, modularity, method


def _community_distance_stats(nodes, coords):
    if len(nodes) < 2:
        return {
            "centroid_lat": np.nan,
            "centroid_lon": np.nan,
            "mean_radius_miles": 0.0,
            "max_radius_miles": 0.0,
            "mean_pairwise_miles": 0.0,
        }

    community_coords = np.array([coords[node] for node in nodes], dtype=float)
    centroid_lat = float(community_coords[:, 0].mean())
    centroid_lon = float(community_coords[:, 1].mean())

    radii = [
        haversine_miles(lat, lon, centroid_lat, centroid_lon)
        for lat, lon in community_coords
    ]

    pairwise_distances = []
    for i in range(len(community_coords)):
        lat1, lon1 = community_coords[i]
        for j in range(i + 1, len(community_coords)):
            lat2, lon2 = community_coords[j]
            pairwise_distances.append(haversine_miles(lat1, lon1, lat2, lon2))

    return {
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "mean_radius_miles": float(np.mean(radii)),
        "max_radius_miles": float(np.max(radii)),
        "mean_pairwise_miles": float(np.mean(pairwise_distances)),
    }


def summarize_communities(graph, communities, state_lookup):
    """Return per-community summaries and aggregate geography metrics."""
    coords = {
        node: (attrs["lat"], attrs["lon"])
        for node, attrs in graph.nodes(data=True)
        if "lat" in attrs and "lon" in attrs
    }
    edge_weight_sum = graph.size(weight="weight")
    rows = []

    for community_id, community in enumerate(communities, start=1):
        nodes = sorted(community)
        size = len(nodes)
        subgraph = graph.subgraph(nodes)
        internal_weight = subgraph.size(weight="weight")
        total_strength = sum(dict(graph.degree(nodes, weight="weight")).values())
        external_weight = total_strength - 2 * internal_weight

        state_counts = Counter(
            state_lookup[node] for node in nodes if node in state_lookup
        ).most_common()
        dominant_state, dominant_count = (
            state_counts[0] if state_counts else ("Unknown", 0)
        )
        distance_stats = _community_distance_stats(nodes, coords)

        rows.append(
            {
                "community_id": community_id,
                "size": size,
                "internal_edges": subgraph.number_of_edges(),
                "internal_weight": float(internal_weight),
                "external_weight": float(external_weight),
                "internal_weight_share": (
                    float(internal_weight / edge_weight_sum) if edge_weight_sum else 0.0
                ),
                "dominant_state": dominant_state,
                "dominant_state_share": dominant_count / size if size else 0.0,
                "state_count": len(state_counts),
                "top_states": ", ".join(
                    f"{state} ({count})" for state, count in state_counts[:4]
                ),
                "sample_stations": ", ".join(nodes[:8]),
                **distance_stats,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["size", "internal_weight"], ascending=[False, False]
    )

    total_nodes = summary["size"].sum()
    aggregate = {
        "n_communities": int(len(summary)),
        "mean_community_size": float(summary["size"].mean()),
        "median_community_size": float(summary["size"].median()),
        "weighted_mean_radius_miles": float(
            np.average(summary["mean_radius_miles"], weights=summary["size"])
        ),
        "weighted_mean_pairwise_miles": float(
            np.average(summary["mean_pairwise_miles"], weights=summary["size"])
        ),
        "weighted_dominant_state_share": float(
            np.average(summary["dominant_state_share"], weights=summary["size"])
        ),
        "share_in_single_state": float((summary["state_count"] == 1).mean()),
        "share_small_communities": float((summary["size"] <= 10).mean()),
        "total_nodes": int(total_nodes),
    }
    return summary, aggregate


def random_partition_by_sizes(nodes, sizes, rng):
    shuffled = list(nodes)
    rng.shuffle(shuffled)

    communities = []
    start = 0
    for size in sizes:
        communities.append(set(shuffled[start : start + size]))
        start += size
    return communities


def permutation_test_geographic_coherence(
    graph,
    observed_communities,
    state_lookup,
    n_trials=500,
    seed=42,
):
    """Compare observed community geography to random partitions of the same sizes."""
    nodes = list(graph.nodes())
    sizes = [len(c) for c in observed_communities]
    rng = random.Random(seed)

    observed_summary, observed_metrics = summarize_communities(
        graph, observed_communities, state_lookup
    )

    null_radius = []
    null_pairwise = []
    null_state_share = []
    null_single_state = []

    for _ in range(n_trials):
        communities = random_partition_by_sizes(nodes, sizes, rng)
        _, metrics = summarize_communities(graph, communities, state_lookup)
        null_radius.append(metrics["weighted_mean_radius_miles"])
        null_pairwise.append(metrics["weighted_mean_pairwise_miles"])
        null_state_share.append(metrics["weighted_dominant_state_share"])
        null_single_state.append(metrics["share_in_single_state"])

    null_radius = np.array(null_radius)
    null_pairwise = np.array(null_pairwise)
    null_state_share = np.array(null_state_share)
    null_single_state = np.array(null_single_state)

    results = {
        "observed_summary": observed_summary,
        "observed_metrics": observed_metrics,
        "null_radius_miles_mean": float(null_radius.mean()),
        "null_radius_miles_std": float(null_radius.std(ddof=1)),
        "null_pairwise_miles_mean": float(null_pairwise.mean()),
        "null_pairwise_miles_std": float(null_pairwise.std(ddof=1)),
        "null_dominant_state_share_mean": float(null_state_share.mean()),
        "null_dominant_state_share_std": float(null_state_share.std(ddof=1)),
        "null_single_state_share_mean": float(null_single_state.mean()),
        "null_single_state_share_std": float(null_single_state.std(ddof=1)),
        "radius_p_value": float(
            (np.count_nonzero(null_radius <= observed_metrics["weighted_mean_radius_miles"]) + 1)
            / (n_trials + 1)
        ),
        "pairwise_p_value": float(
            (
                np.count_nonzero(
                    null_pairwise <= observed_metrics["weighted_mean_pairwise_miles"]
                )
                + 1
            )
            / (n_trials + 1)
        ),
        "state_share_p_value": float(
            (np.count_nonzero(null_state_share >= observed_metrics["weighted_dominant_state_share"]) + 1)
            / (n_trials + 1)
        ),
        "single_state_p_value": float(
            (np.count_nonzero(null_single_state >= observed_metrics["share_in_single_state"]) + 1)
            / (n_trials + 1)
        ),
        "radius_z_score": _z_score(
            observed_metrics["weighted_mean_radius_miles"], null_radius.mean(), null_radius.std(ddof=1)
        ),
        "pairwise_z_score": _z_score(
            observed_metrics["weighted_mean_pairwise_miles"],
            null_pairwise.mean(),
            null_pairwise.std(ddof=1),
        ),
        "state_share_z_score": _z_score(
            observed_metrics["weighted_dominant_state_share"],
            null_state_share.mean(),
            null_state_share.std(ddof=1),
        ),
        "single_state_z_score": _z_score(
            observed_metrics["share_in_single_state"],
            null_single_state.mean(),
            null_single_state.std(ddof=1),
        ),
        "n_trials": int(n_trials),
    }
    return results


def _z_score(observed, mean, std):
    if std == 0:
        return np.nan
    return float((observed - mean) / std)


def print_community_report(summary, modularity, method, test_results, top_n=10):
    observed = test_results["observed_metrics"]
    print(
        f"Detected {observed['n_communities']} communities via {method} "
        f"(weighted modularity = {modularity:.4f})."
    )
    print(
        f"Weighted mean community radius: {observed['weighted_mean_radius_miles']:.1f} miles "
        f"vs random {test_results['null_radius_miles_mean']:.1f} "
        f"(z = {test_results['radius_z_score']:.1f}, p = {test_results['radius_p_value']:.4f})"
    )
    print(
        f"Weighted mean pairwise distance: {observed['weighted_mean_pairwise_miles']:.1f} miles "
        f"vs random {test_results['null_pairwise_miles_mean']:.1f} "
        f"(z = {test_results['pairwise_z_score']:.1f}, p = {test_results['pairwise_p_value']:.4f})"
    )
    print(
        f"Weighted dominant-state share: {observed['weighted_dominant_state_share']:.3f} "
        f"vs random {test_results['null_dominant_state_share_mean']:.3f} "
        f"(z = {test_results['state_share_z_score']:.1f}, p = {test_results['state_share_p_value']:.4f})"
    )
    print(
        f"Single-state community share: {observed['share_in_single_state']:.3f} "
        f"vs random {test_results['null_single_state_share_mean']:.3f} "
        f"(z = {test_results['single_state_z_score']:.1f}, p = {test_results['single_state_p_value']:.4f})"
    )

    columns = [
        "community_id",
        "size",
        "dominant_state",
        "dominant_state_share",
        "state_count",
        "mean_radius_miles",
        "mean_pairwise_miles",
        "top_states",
        "sample_stations",
    ]
    display_summary = summary.loc[:, columns].copy()
    print("\nLargest communities:")
    print(display_summary.head(top_n).to_string(index=False))


def plot_community_size_distribution(summary, output_path="community-size-distribution.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        summary["community_id"].astype(str),
        summary["size"],
        color="#1f4e79",
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_xlabel("Community ID", fontsize=12)
    ax.set_ylabel("Stations", fontsize=12)
    ax.set_title("Community Size Distribution", fontsize=14)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def plot_community_geographic_coherence(
    summary, test_results, output_path="community-geographic-coherence.png"
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].scatter(
        summary["size"],
        summary["mean_radius_miles"],
        s=70,
        color="#c0392b",
        alpha=0.8,
    )
    axes[0].axhline(
        test_results["null_radius_miles_mean"],
        color="#34495e",
        linestyle="--",
        linewidth=1.6,
        label="Random-size-matched mean",
    )
    axes[0].set_xlabel("Community size", fontsize=12)
    axes[0].set_ylabel("Mean radius from centroid (miles)", fontsize=12)
    axes[0].set_title("Spatial Compactness", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.2)

    axes[1].scatter(
        summary["size"],
        summary["dominant_state_share"],
        s=70,
        color="#1f77b4",
        alpha=0.8,
    )
    axes[1].axhline(
        test_results["null_dominant_state_share_mean"],
        color="#34495e",
        linestyle="--",
        linewidth=1.6,
        label="Random-size-matched mean",
    )
    axes[1].set_xlabel("Community size", fontsize=12)
    axes[1].set_ylabel("Dominant-state share", fontsize=12)
    axes[1].set_title("State Concentration", fontsize=13)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return fig, axes


def plot_communities_map(graph, communities, output_path="community-map.png"):
    if not HAS_CARTOPY:
        print("Skipping community map (cartopy not installed)")
        return None, None

    pos = {
        node: (attrs["lon"], attrs["lat"])
        for node, attrs in graph.nodes(data=True)
        if "lon" in attrs and "lat" in attrs
    }
    if not pos:
        return None, None

    community_by_node = {}
    for idx, community in enumerate(communities):
        for node in community:
            community_by_node[node] = idx

    xs = [coord[0] for coord in pos.values()]
    ys = [coord[1] for coord in pos.values()]
    lon_margin = max((max(xs) - min(xs)) * 0.12, 2.0)
    lat_margin = max((max(ys) - min(ys)) * 0.12, 1.5)

    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes(projection=ccrs.LambertConformal())
    ax.set_extent(
        (
            min(xs) - lon_margin,
            max(xs) + lon_margin,
            min(ys) - lat_margin,
            max(ys) + lat_margin,
        ),
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.LAND, facecolor="#f3efe6")
    ax.add_feature(cfeature.OCEAN, facecolor="#dbe9f4")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.STATES, linewidth=0.25, edgecolor="#999999")

    cmap = plt.get_cmap("tab20")
    for u, v, attrs in graph.edges(data=True):
        if u not in pos or v not in pos:
            continue
        same_community = community_by_node.get(u) == community_by_node.get(v)
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=cmap(community_by_node.get(u, 0) % 20) if same_community else "#b0b0b0",
            alpha=0.55 if same_community else 0.15,
            linewidth=0.8 if same_community else 0.45,
            transform=ccrs.PlateCarree(),
            zorder=1,
        )

    for idx, community in enumerate(communities):
        plotted_nodes = [node for node in community if node in pos]
        if not plotted_nodes:
            continue
        ax.scatter(
            [pos[node][0] for node in plotted_nodes],
            [pos[node][1] for node in plotted_nodes],
            s=18,
            color=cmap(idx % 20),
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

    ax.set_title("Amtrak Communities Detected Without Geography", fontsize=17, pad=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def analyze_community_structure(graph, n_trials=500, seed=42):
    state_lookup = load_station_state_lookup()
    communities, modularity, method = detect_communities(graph, seed=seed)
    test_results = permutation_test_geographic_coherence(
        graph,
        communities,
        state_lookup,
        n_trials=n_trials,
        seed=seed,
    )
    summary = test_results["observed_summary"]
    return {
        "communities": communities,
        "modularity": modularity,
        "method": method,
        "summary": summary,
        "test_results": test_results,
    }


if __name__ == "__main__":
    analysis = analyze_community_structure(G, n_trials=500, seed=42)
    print(f"Network: {G.number_of_nodes()} stations, {G.number_of_edges()} edges")
    print_community_report(
        analysis["summary"],
        analysis["modularity"],
        analysis["method"],
        analysis["test_results"],
        top_n=10,
    )
    plot_community_size_distribution(analysis["summary"])
    plot_community_geographic_coherence(
        analysis["summary"],
        analysis["test_results"],
    )
    plot_communities_map(G, analysis["communities"])
