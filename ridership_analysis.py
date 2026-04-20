import numpy as np
import pandas as pd


STATE_ABBREVIATIONS = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}


STATION_NAME_ALIASES = {
    "BBY": "Boston-Back Bay, Massachusetts",
    "BON": "Boston-North Station, Massachusetts",
    "BOS": "Boston-South Station, Massachusetts",
    "BFX": "Buffalo-Exchange Street, New York",
    "BUF": "Buffalo-Depew, New York",
    "BWI": "BWI Thurgood Marshall Airport, Maryland",
    "GPK": "East Glacier (summer only), Montana",
    "LKL": "Lakeland, Florida",
    "LAK": "Lakeland, Florida",
    "MKA": "Milwaukee Airport, Wisconsin",
    "MKE": "Milwaukee, Wisconsin",
    "MPR": "Montpelier, Vermont",
    "NYP": "New York City (Penn Station), New York",
    "NFS": "Niagara Falls, New York",
    "OLW": "Olympia/Lacey, Washington",
    "PHL": "Philadelphia William H. Gray III 30th Street, Pennsylvania",
    "PHN": "Philadelphia-North, Pennsylvania",
    "PIH": "Pinehurst, North Carolina (Special Stop)",
    "RVM": "Richmond - Main Street, Virginia",
    "RVR": "Richmond - Staples Mill, Virginia",
    "RTE": "Route 128 (Boston), Massachusetts",
    "SCC": "Santa Clara (University), California",
    "SFA": "Sanford (Auto Train Station), Florida",
    "SKN": "Stockton (San Joaquin St.), California",
    "SKT": "Stockton (Downtown), California",
    "WAS": "Washington Union Station, District of Columbia",
    "WAB": "Waterbury, Vermont",
    "WIP": "Winter Park-Fraser, Colorado",
    "WPR": "Winter Park Resort (seasonal), Colorado",
}


def normalize_station_name(name):
    if pd.isna(name):
        return None

    city, _, state = str(name).rpartition(", ")
    if not city:
        return str(name).strip()

    return f"{city}, {STATE_ABBREVIATIONS.get(state, state)}"


def load_station_ridership_by_code(
    station_data_path="station-data.xlsx",
    station_ridership_path="station-ridership.xlsx",
):
    station_data = pd.read_excel(station_data_path).copy()
    station_ridership = pd.read_excel(station_ridership_path).copy()

    station_data["normalized_name"] = station_data["Name"].map(normalize_station_name)
    station_data["lookup_name"] = station_data["Code"].map(STATION_NAME_ALIASES)
    station_data["lookup_name"] = station_data["lookup_name"].fillna(
        station_data["normalized_name"]
    )

    ridership_lookup = (
        station_ridership.dropna(subset=["Station", "Ridership"])
        .drop_duplicates(subset=["Station"], keep="first")
        .set_index("Station")["Ridership"]
    )

    matched = station_data.dropna(subset=["Code"]).copy()
    matched["Ridership"] = matched["lookup_name"].map(ridership_lookup)

    ridership_by_code = (
        matched.dropna(subset=["Ridership"]).set_index("Code")["Ridership"].to_dict()
    )

    unmatched = matched.loc[
        matched["Ridership"].isna(), ["Code", "Name", "lookup_name"]
    ].sort_values("Code")

    return ridership_by_code, unmatched


def build_degree_ridership_comparison(graph, ridership_lookup):
    station_data = pd.read_excel("station-data.xlsx").copy()
    station_data["normalized_name"] = station_data["Name"].map(normalize_station_name)
    station_data["lookup_name"] = station_data["Code"].map(STATION_NAME_ALIASES)
    station_data["lookup_name"] = station_data["lookup_name"].fillna(
        station_data["normalized_name"]
    )

    comparison = station_data.loc[
        station_data["Code"].isin(ridership_lookup), ["Code", "Name", "lookup_name"]
    ].copy()
    comparison["Ridership"] = comparison["Code"].map(ridership_lookup)

    # Calculate both weighted and unweighted degree
    comparison["Weighted Degree"] = comparison["Code"].map(
        dict(graph.degree(weight="weight"))
    )
    comparison["Unweighted Degree"] = comparison["Code"].map(dict(graph.degree()))
    comparison = comparison.dropna(
        subset=["Weighted Degree", "Unweighted Degree"]
    ).copy()
    comparison["Weighted Degree"] = comparison["Weighted Degree"].astype(float)
    comparison["Unweighted Degree"] = comparison["Unweighted Degree"].astype(float)
    comparison["Ridership"] = comparison["Ridership"].astype(float)

    # Process weighted degree
    comparison["Scaled Weighted Degree"] = comparison["Weighted Degree"] * 1000.0
    comparison["Log Weighted Degree"] = np.log1p(comparison["Scaled Weighted Degree"])
    comparison["Normalized Weighted Degree"] = (
        comparison["Log Weighted Degree"] - comparison["Log Weighted Degree"].min()
    ) / (
        comparison["Log Weighted Degree"].max()
        - comparison["Log Weighted Degree"].min()
    )

    # Process unweighted degree
    comparison["Scaled Unweighted Degree"] = comparison["Unweighted Degree"] * 1000.0
    comparison["Log Unweighted Degree"] = np.log1p(
        comparison["Scaled Unweighted Degree"]
    )
    comparison["Normalized Unweighted Degree"] = (
        comparison["Log Unweighted Degree"] - comparison["Log Unweighted Degree"].min()
    ) / (
        comparison["Log Unweighted Degree"].max()
        - comparison["Log Unweighted Degree"].min()
    )

    # Process ridership (same for both)
    comparison["Log Ridership"] = np.log1p(comparison["Ridership"])
    comparison["Normalized Ridership"] = (
        comparison["Log Ridership"] - comparison["Log Ridership"].min()
    ) / (comparison["Log Ridership"].max() - comparison["Log Ridership"].min())

    # Keep legacy "Degree" and "Normalized Degree" for backward compatibility with existing code
    comparison["Degree"] = comparison["Weighted Degree"]
    comparison["Scaled Degree"] = comparison["Scaled Weighted Degree"]
    comparison["Log Degree"] = comparison["Log Weighted Degree"]
    comparison["Normalized Degree"] = comparison["Normalized Weighted Degree"]

    return comparison


def fit_ridership_regression(comparison):
    x = comparison["Normalized Degree"].to_numpy()
    y = comparison["Normalized Ridership"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residuals = y - predicted
    absolute_residuals = np.abs(residuals)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot else np.nan

    fitted = comparison.copy()
    fitted["Predicted Normalized Ridership"] = predicted
    fitted["Residual"] = residuals
    fitted["Absolute Residual"] = absolute_residuals

    return fitted, slope, intercept, r_squared


def identify_outliers(comparison, max_outliers=12):
    return comparison.sort_values("Absolute Residual", ascending=False).head(
        max_outliers
    )


def fit_weighted_vs_unweighted_models(comparison):
    """
    Fit linear regressions for both weighted and unweighted degree models
    to test the hypothesis that weighted degree is a better predictor of ridership.

    Returns dict with results for both models including R², slope, intercept, and residuals.
    """
    y = comparison["Normalized Ridership"].to_numpy()

    # Weighted degree model
    x_weighted = comparison["Normalized Weighted Degree"].to_numpy()
    slope_weighted, intercept_weighted = np.polyfit(x_weighted, y, 1)
    predicted_weighted = slope_weighted * x_weighted + intercept_weighted
    residuals_weighted = y - predicted_weighted
    ss_res_weighted = np.sum(residuals_weighted**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared_weighted = 1 - (ss_res_weighted / ss_tot) if ss_tot else np.nan

    weighted_fitted = comparison.copy()
    weighted_fitted["Predicted Normalized Ridership"] = predicted_weighted
    weighted_fitted["Residual"] = residuals_weighted
    weighted_fitted["Absolute Residual"] = np.abs(residuals_weighted)

    # Unweighted degree model
    x_unweighted = comparison["Normalized Unweighted Degree"].to_numpy()
    slope_unweighted, intercept_unweighted = np.polyfit(x_unweighted, y, 1)
    predicted_unweighted = slope_unweighted * x_unweighted + intercept_unweighted
    residuals_unweighted = y - predicted_unweighted
    ss_res_unweighted = np.sum(residuals_unweighted**2)
    r_squared_unweighted = 1 - (ss_res_unweighted / ss_tot) if ss_tot else np.nan

    unweighted_fitted = comparison.copy()
    unweighted_fitted["Predicted Normalized Ridership"] = predicted_unweighted
    unweighted_fitted["Residual"] = residuals_unweighted
    unweighted_fitted["Absolute Residual"] = np.abs(residuals_unweighted)

    return {
        "weighted": {
            "fitted": weighted_fitted,
            "slope": slope_weighted,
            "intercept": intercept_weighted,
            "r_squared": r_squared_weighted,
            "rmse": np.sqrt(np.mean(residuals_weighted**2)),
            "mae": np.mean(np.abs(residuals_weighted)),
        },
        "unweighted": {
            "fitted": unweighted_fitted,
            "slope": slope_unweighted,
            "intercept": intercept_unweighted,
            "r_squared": r_squared_unweighted,
            "rmse": np.sqrt(np.mean(residuals_unweighted**2)),
            "mae": np.mean(np.abs(residuals_unweighted)),
        },
    }


def plot_degree_vs_ridership_scatter(
    comparison,
    outliers,
    slope,
    intercept,
    r_squared,
    output_path="degree-ridership-scatter.png",
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(
        comparison["Normalized Degree"],
        comparison["Normalized Ridership"],
        alpha=0.55,
        s=28,
        color="#1f77b4",
        label="Stations",
    )

    x_min = comparison["Normalized Degree"].min()
    x_max = comparison["Normalized Degree"].max()
    x_line = np.linspace(x_min, x_max, 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#444444", linewidth=2, label="Best fit")

    if not outliers.empty:
        ax.scatter(
            outliers["Normalized Degree"],
            outliers["Normalized Ridership"],
            color="#d62728",
            s=42,
            label="Outliers",
            zorder=3,
        )

        for _, row in outliers.iterrows():
            ax.annotate(
                row["Code"],
                (row["Normalized Degree"], row["Normalized Ridership"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#d62728",
            )

    ax.set_title(f"Normalized Log Degree vs Ridership (R^2 = {r_squared:.4f})")
    ax.set_xlabel("Normalized log weighted degree")
    ax.set_ylabel("Normalized log ridership")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_degree_vs_ridership_bars(
    comparison, output_path="degree-ridership-bars.png", max_stations=None
):
    import matplotlib.pyplot as plt

    ordered = comparison.sort_values("Normalized Ridership", ascending=False).copy()
    if max_stations is not None:
        ordered = ordered.head(max_stations).copy()

    positions = np.arange(len(ordered))
    width = 0.45

    fig_width = max(14, len(ordered) * 0.16)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    ax.bar(
        positions - width / 2,
        ordered["Normalized Ridership"],
        width=width,
        color="#ff7f0e",
        alpha=0.65,
        label="Normalized log ridership",
    )
    ax.bar(
        positions + width / 2,
        ordered["Normalized Degree"],
        width=width,
        color="#1f77b4",
        alpha=0.65,
        label="Normalized log weighted degree",
    )

    ax.set_title("Normalized Ridership and Degree by Station")
    ax.set_xlabel("Station")
    ax.set_ylabel("Normalized value")
    ax.set_xticks(positions)
    ax.set_xticklabels(ordered["Code"], rotation=90, fontsize=7)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_degree_models_comparison(
    comparison, models_results, output_path="weighted-vs-unweighted-comparison.png"
):
    """
    Plot side-by-side comparison of weighted vs unweighted degree models.
    This visualization tests the hypothesis that weighted degree is a better predictor.
    """
    import matplotlib.pyplot as plt

    weighted_results = models_results["weighted"]
    unweighted_results = models_results["unweighted"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Weighted degree plot
    ax_weighted = axes[0]
    ax_weighted.scatter(
        comparison["Normalized Weighted Degree"],
        comparison["Normalized Ridership"],
        alpha=0.55,
        s=28,
        color="#1f77b4",
        label="Stations",
    )

    x_min = comparison["Normalized Weighted Degree"].min()
    x_max = comparison["Normalized Weighted Degree"].max()
    x_line = np.linspace(x_min, x_max, 200)
    y_line = weighted_results["slope"] * x_line + weighted_results["intercept"]
    ax_weighted.plot(x_line, y_line, color="#444444", linewidth=2, label="Best fit")

    ax_weighted.set_title(
        f"Weighted Degree Model\n(R² = {weighted_results['r_squared']:.4f}, "
        f"RMSE = {weighted_results['rmse']:.4f})"
    )
    ax_weighted.set_xlabel("Normalized log weighted degree")
    ax_weighted.set_ylabel("Normalized log ridership")
    ax_weighted.legend()
    ax_weighted.grid(True, alpha=0.3)

    # Unweighted degree plot
    ax_unweighted = axes[1]
    ax_unweighted.scatter(
        comparison["Normalized Unweighted Degree"],
        comparison["Normalized Ridership"],
        alpha=0.55,
        s=28,
        color="#ff7f0e",
        label="Stations",
    )

    x_min = comparison["Normalized Unweighted Degree"].min()
    x_max = comparison["Normalized Unweighted Degree"].max()
    x_line = np.linspace(x_min, x_max, 200)
    y_line = unweighted_results["slope"] * x_line + unweighted_results["intercept"]
    ax_unweighted.plot(x_line, y_line, color="#444444", linewidth=2, label="Best fit")

    ax_unweighted.set_title(
        f"Unweighted Degree Model\n(R² = {unweighted_results['r_squared']:.4f}, "
        f"RMSE = {unweighted_results['rmse']:.4f})"
    )
    ax_unweighted.set_xlabel("Normalized log unweighted degree")
    ax_unweighted.set_ylabel("Normalized log ridership")
    ax_unweighted.legend()
    ax_unweighted.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


ridership_by_code, unmatched_stations = load_station_ridership_by_code()

from network import G

comparison_df = build_degree_ridership_comparison(G, ridership_by_code)
comparison_df, slope, intercept, r_squared = fit_ridership_regression(comparison_df)
average_error = comparison_df["Absolute Residual"].mean()
outliers_df = identify_outliers(comparison_df)

# Compare weighted vs unweighted degree models
models_results = fit_weighted_vs_unweighted_models(comparison_df)


if __name__ == "__main__":
    plot_degree_vs_ridership_scatter(
        comparison_df, outliers_df, slope, intercept, r_squared
    )
    plot_degree_vs_ridership_bars(comparison_df)
    plot_degree_models_comparison(comparison_df, models_results)

    print("=" * 80)
    print("HYPOTHESIS TEST: Weighted vs Unweighted Degree Centrality")
    print("=" * 80)
    print("\nHypothesis: Weighted degree centrality predicts annual boardings more")
    print("accurately than unweighted degree centrality.")
    print("\nRationale: Edge weights are defined by annual passenger volume (total")
    print("route ridership). If the weighted model is more predictive, it suggests")
    print("that this simplification was not disastrous.\n")

    weighted_r2 = models_results["weighted"]["r_squared"]
    unweighted_r2 = models_results["unweighted"]["r_squared"]
    r2_improvement = weighted_r2 - unweighted_r2
    r2_pct_better = (
        (r2_improvement / abs(unweighted_r2)) * 100 if unweighted_r2 != 0 else 0
    )

    weighted_rmse = models_results["weighted"]["rmse"]
    unweighted_rmse = models_results["unweighted"]["rmse"]
    rmse_improvement = unweighted_rmse - weighted_rmse
    rmse_pct_better = (
        (rmse_improvement / unweighted_rmse) * 100 if unweighted_rmse != 0 else 0
    )

    print("MODEL COMPARISON RESULTS:")
    print("-" * 80)
    print(f"\nWeighted Degree Model:")
    print(f"  R²:                              {weighted_r2:.4f}")
    print(f"  RMSE:                            {weighted_rmse:.4f}")
    print(f"  MAE (Mean Absolute Error):       {models_results['weighted']['mae']:.4f}")
    print(
        f"  Slope:                           {models_results['weighted']['slope']:.4f}"
    )
    print(
        f"  Intercept:                       {models_results['weighted']['intercept']:.4f}"
    )

    print(f"\nUnweighted Degree Model:")
    print(f"  R²:                              {unweighted_r2:.4f}")
    print(f"  RMSE:                            {unweighted_rmse:.4f}")
    print(
        f"  MAE (Mean Absolute Error):       {models_results['unweighted']['mae']:.4f}"
    )
    print(
        f"  Slope:                           {models_results['unweighted']['slope']:.4f}"
    )
    print(
        f"  Intercept:                       {models_results['unweighted']['intercept']:.4f}"
    )

    print(f"\nModel Performance Difference:")
    print(
        f"  R² improvement (weighted):       {r2_improvement:+.4f} ({r2_pct_better:+.2f}%)"
    )
    print(
        f"  RMSE improvement (weighted):     {rmse_improvement:+.4f} ({rmse_pct_better:+.2f}%)"
    )

    if weighted_r2 > unweighted_r2:
        print(f"\n✓ HYPOTHESIS SUPPORTED: Weighted degree is a BETTER predictor")
        print(f"  The weighted model explains {abs(r2_pct_better):.1f}% more variance")
        print(f"  in annual boardings than the unweighted model.")
    else:
        print(f"\n✗ HYPOTHESIS NOT SUPPORTED: Unweighted degree is EQUALLY or BETTER")
        print(f"  The unweighted model explains at least as much variance.")

    print("\n" + "=" * 80)
    print("ORIGINAL WEIGHTED DEGREE ANALYSIS (for reference):")
    print("=" * 80)
    plot_degree_vs_ridership_bars(comparison_df)

    print(f"Compared stations: {len(comparison_df)}")
    print(f"Average absolute residual: {average_error:.4f}")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R^2: {r_squared:.4f}")
    print("\nWorst outliers:")
    print(
        outliers_df[
            [
                "Code",
                "Name",
                "lookup_name",
                "Degree",
                "Scaled Degree",
                "Ridership",
                "Normalized Degree",
                "Normalized Ridership",
                "Predicted Normalized Ridership",
                "Residual",
                "Absolute Residual",
            ]
        ].to_string(index=False)
    )

    if not unmatched_stations.empty:
        print("\nUnmatched stations:")
        print(unmatched_stations.to_string(index=False))
