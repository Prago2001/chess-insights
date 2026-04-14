"""
ChessInsight Streamlit Dashboard
Team 029 - CSE6242 Spring 2026

Run locally with:
    streamlit run streamlit_app.py

This app visualizes the outputs of the analysis pipeline in `run_analysis.py` using
precomputed artifacts in `data/processed/` and `models/`.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PROCESSED_DATA_DIR, MODELS_DIR

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ChessInsight Streamlit Dashboard",
    page_icon="♟",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SKILL_TIERS = ["Beginner", "Intermediate", "Advanced", "Expert"]
TIER_COLORS = {
    "Beginner": "#e74c3c",
    "Intermediate": "#f39c12",
    "Advanced": "#3498db",
    "Expert": "#2ecc71",
}
CLUSTER_COLORS = px.colors.qualitative.Set2


# ---------------------------------------------------------------------------
# Data loading helpers (cached for fast local dev)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_player_data() -> pd.DataFrame:
    """Load player-level features and clustering embeddings."""
    pf = pd.read_parquet(PROCESSED_DATA_DIR / "player_features.parquet")
    emb = pd.read_parquet(PROCESSED_DATA_DIR / "player_clustering_embeddings.parquet")
    pf = pd.concat([pf.reset_index(drop=True), emb.reset_index(drop=True)], axis=1)
    return pf


@st.cache_data(show_spinner=False)
def load_game_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DATA_DIR / "game_features.parquet")


@st.cache_data(show_spinner=False)
def load_games_processed() -> pd.DataFrame:
    """Load game-level data used for opening pattern analysis."""
    return pd.read_parquet(PROCESSED_DATA_DIR / "games_processed.parquet")


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    with open(MODELS_DIR / name) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / name)


# ---------------------------------------------------------------------------
# Load all core artifacts once per session with basic error handling
# ---------------------------------------------------------------------------
try:
    player_df = load_player_data()
    game_df = load_game_features()
    analysis = load_json("analysis_summary.json")
    clustering_results = load_json("player_clustering_results.json")
    classifier_metrics = load_json("skill_classifier_metrics.json")
    feature_importance = load_csv("skill_classifier_feature_importance.csv")
    confusion_csv = load_csv("skill_classifier_confusion_matrix.csv")
    cluster_stats = load_csv("player_clustering_statistics.csv")
    method_comparison = load_csv("clustering_method_comparison.csv")
except Exception as e:  # pragma: no cover - runtime safeguard
    st.error(
        "Failed to load processed data and model artifacts. "
        "Please run `python run_analysis.py` first so that `data/processed/` "
        "and `models/` are populated.\n\nDetails: " + str(e)
    )
    st.stop()

# Build cluster name map and attach to player_df
cluster_name_map = {int(k): v["name"] for k, v in clustering_results["cluster_names"].items()}
player_df["cluster_name"] = player_df["cluster"].map(cluster_name_map)

# Confusion matrix as numpy array
cm_labels = list(confusion_csv.columns[1:])
cm_array = confusion_csv.iloc[:, 1:].values.astype(int)

# Precompute aggregates for time heatmap (game-level)
time_phase_cols = {
    "Opening": "white_avg_time_opening",
    "Middlegame": "white_avg_time_middlegame",
    "Endgame": "white_avg_time_endgame",
}

time_heatmap_data = []
for tier in SKILL_TIERS:
    tier_games = game_df[game_df["white_skill_tier"] == tier]
    row = {"Skill Tier": tier}
    for phase, col in time_phase_cols.items():
        row[phase] = round(tier_games[col].mean(), 2) if col in tier_games.columns else 0.0
    time_heatmap_data.append(row)

time_heatmap_df = pd.DataFrame(time_heatmap_data).set_index("Skill Tier")

# Precompute accuracy-related aggregates (game-level)
accuracy_by_tier = []
for tier in SKILL_TIERS:
    t = game_df[game_df["white_skill_tier"] == tier]
    accuracy_by_tier.append(
        {
            "Skill Tier": tier,
            "Blunder Rate (%)": round(t["white_blunder_rate"].mean() * 100, 2)
            if "white_blunder_rate" in t.columns
            else 0.0,
            "Mistake Rate (%)": round(t["white_mistake_rate"].mean() * 100, 2)
            if "white_mistake_rate" in t.columns
            else 0.0,
            "Avg CPL": round(t["white_avg_centipawn_loss"].mean(), 1)
            if "white_avg_centipawn_loss" in t.columns
            else 0.0,
            "Accuracy (%)": round(t["white_accuracy_percentage"].mean(), 1)
            if "white_accuracy_percentage" in t.columns
            else 0.0,
        }
    )
accuracy_df = pd.DataFrame(accuracy_by_tier)


# ---------------------------------------------------------------------------
# Helper to render a numeric KPI-style block
# ---------------------------------------------------------------------------
def render_kpi(label: str, value: str, help_text: Optional[str] = None):
    st.metric(label=label, value=value, help=help_text)


# ---------------------------------------------------------------------------
# Overview Tab
# ---------------------------------------------------------------------------
def render_overview_tab():
    st.subheader("Overview")

    ds = analysis["dataset"]
    cl = analysis["classification"]
    clu = analysis["clustering"]

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        render_kpi("Total Games", f"{ds['total_games']:,}", ds["data_source"])
    with kpi_cols[1]:
        render_kpi("Unique Players", f"{ds['unique_players']:,}", f"Rating {ds['rating_range']}")
    with kpi_cols[2]:
        render_kpi("Test Accuracy", cl["test_accuracy"], "Exact tier classification accuracy")
    with kpi_cols[3]:
        render_kpi("Clusters Found", str(clu["n_clusters"]), f"Silhouette: {clu['silhouette_score']}")

    # Skill tier distribution
    tier_counts = pd.DataFrame(
        [
            {"Tier": t, "Count": int(classifier_metrics["class_distribution"].get(t, 0))}
            for t in SKILL_TIERS
        ]
    )
    fig_dist = px.bar(
        tier_counts,
        x="Tier",
        y="Count",
        color="Tier",
        color_discrete_map=TIER_COLORS,
        title="Skill Tier Distribution (Classification Dataset)",
    )
    fig_dist.update_layout(showlegend=False, height=350)

    # Rating histogram from player data
    fig_rating = px.histogram(
        player_df,
        x="avg_elo",
        nbins=60,
        title="Player Rating Distribution",
        labels={"avg_elo": "Average Elo"},
        color_discrete_sequence=["#3498db"],
    )
    fig_rating.update_layout(height=350)

    # Accuracy metrics by tier
    acc_melted = accuracy_df.melt(id_vars="Skill Tier", var_name="Metric", value_name="Value")
    fig_acc = px.bar(
        acc_melted,
        x="Skill Tier",
        y="Value",
        color="Metric",
        barmode="group",
        title="Accuracy Metrics by Skill Tier",
    )
    fig_acc.update_layout(height=380)

    col_top = st.columns(2)
    with col_top[0]:
        st.plotly_chart(fig_dist, use_container_width=True)
    with col_top[1]:
        st.plotly_chart(fig_rating, use_container_width=True)

    st.plotly_chart(fig_acc, use_container_width=True)


# ---------------------------------------------------------------------------
# Player Cluster Map Tab (with per-player drill-down)
# ---------------------------------------------------------------------------
def render_cluster_tab():
    st.subheader("Player Cluster Map")

    col_controls, col_plot = st.columns([1, 3])

    with col_controls:
        color_by = st.radio(
            "Color By",
            options=["Cluster Archetype", "Skill Tier"],
            index=0,
        )

        tiers_selected = st.multiselect(
            "Filter Skill Tiers",
            options=SKILL_TIERS,
            default=SKILL_TIERS,
            key="cluster_tiers_multiselect",
        )

        min_elo = int(player_df["avg_elo"].min())
        max_elo = int(player_df["avg_elo"].max())
        rating_min, rating_max = st.slider(
            "Rating Range",
            min_value=min_elo,
            max_value=max_elo,
            value=(min_elo, max_elo),
            step=50,
        )

        st.markdown("**Cluster Summary**")
        summary_rows = []
        for cid in sorted(cluster_name_map.keys()):
            info = clustering_results["cluster_names"][str(cid)]
            summary_rows.append(
                {
                    "Archetype": info["name"],
                    "Size": info["size"],
                    "Avg Elo": round(info["avg_elo"]),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("**Player Drill-Down**")
        player_query = st.text_input(
            "Enter player handle (exact match)",
            "",
            help="Type a specific Lichess player handle present in the dataset.",
        )
        selected_player = None
        if player_query:
            matches = player_df[player_df["player"] == player_query]
            if not matches.empty:
                selected_player = matches.iloc[0]
                st.markdown(
                    f"**Selected player:** `{player_query}`  |  Elo: {selected_player['avg_elo']:.0f}  "
                    f"|  Tier: {selected_player['skill_tier']}  |  Archetype: {selected_player['cluster_name']}"
                )
                st.markdown(
                    f"Games analyzed: **{int(selected_player.get('num_games', 0))}**"
                )
            else:
                st.info("No player found with that handle in the processed dataset.")
        else:
            st.caption("Tip: paste a handle from the scatterplot tooltip to inspect a player.")

    with col_plot:
        df = player_df[
            (player_df["skill_tier"].isin(tiers_selected))
            & (player_df["avg_elo"] >= rating_min)
            & (player_df["avg_elo"] <= rating_max)
        ].copy()

        if color_by == "Skill Tier":
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="skill_tier",
                color_discrete_map=TIER_COLORS,
                hover_data=["player", "avg_elo", "num_games", "cluster_name"],
                title=f"Player Embedding Map — {len(df):,} players",
                labels={
                    "x": "t-SNE Dim 1",
                    "y": "t-SNE Dim 2",
                    "skill_tier": "Skill Tier",
                },
                category_orders={"skill_tier": SKILL_TIERS},
            )
        else:
            fig = px.scatter(
                df,
                x="x",
                y="y",
                color="cluster_name",
                color_discrete_sequence=CLUSTER_COLORS,
                hover_data=["player", "avg_elo", "num_games", "skill_tier"],
                title=f"Player Embedding Map — {len(df):,} players",
                labels={
                    "x": "t-SNE Dim 1",
                    "y": "t-SNE Dim 2",
                    "cluster_name": "Archetype",
                },
            )

        # Highlight selected player if present
        if "selected_player" in locals() and selected_player is not None:
            fig.add_trace(
                go.Scatter(
                    x=[selected_player["x"]],
                    y=[selected_player["y"]],
                    mode="markers",
                    marker=dict(
                        size=12,
                        color="black",
                        symbol="x",
                        line=dict(width=2, color="white"),
                    ),
                    name="Selected Player",
                    hovertext=[f"Selected: {selected_player['player']}`"],
                    hoverinfo="text",
                )
            )

        fig.update_traces(marker=dict(size=4, opacity=0.6, line=dict(width=0)), selector=dict(mode="markers"))
        fig.update_layout(
            margin=dict(l=10, r=10, t=60, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Time Analysis Tab
# ---------------------------------------------------------------------------
def render_time_tab():
    st.subheader("Time Usage and Time Pressure")

    # Heatmap of average time per move
    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=time_heatmap_df.values,
            x=time_heatmap_df.columns.tolist(),
            y=time_heatmap_df.index.tolist(),
            colorscale="YlOrRd",
            text=time_heatmap_df.values.round(2),
            texttemplate="%{text}s",
            hovertemplate=(
                "Tier: %{y}<br>Phase: %{x}<br>Avg Time: %{z:.2f}s<extra></extra>"
            ),
        )
    )
    fig_heatmap.update_layout(
        title="Average Time per Move (seconds) by Skill Tier & Game Phase",
        xaxis_title="Game Phase",
        yaxis_title="Skill Tier",
        height=380,
    )

    # Time variance by tier using player-level data
    tv_data = []
    for phase, col in [
        ("Opening", "time_variance_opening_mean"),
        ("Middlegame", "time_variance_middlegame_mean"),
        ("Endgame", "time_variance_endgame_mean"),
    ]:
        if col in player_df.columns:
            for tier in SKILL_TIERS:
                vals = player_df[player_df["skill_tier"] == tier][col]
                tv_data.append(
                    {
                        "Phase": phase,
                        "Skill Tier": tier,
                        "Time Variance": vals.mean(),
                    }
                )
    tv_df = pd.DataFrame(tv_data)
    fig_variance = px.bar(
        tv_df,
        x="Phase",
        y="Time Variance",
        color="Skill Tier",
        barmode="group",
        color_discrete_map=TIER_COLORS,
        title="Average Time Variance by Skill Tier & Game Phase",
        category_orders={"Skill Tier": SKILL_TIERS},
    )
    fig_variance.update_layout(height=380)

    # Time trouble frequency
    if "time_trouble_frequency_mean" in player_df.columns:
        tt_data = []
        for tier in SKILL_TIERS:
            vals = player_df[player_df["skill_tier"] == tier][
                "time_trouble_frequency_mean"
            ]
            tt_data.append(
                {"Skill Tier": tier, "Time Trouble Freq": vals.mean()}
            )
        tt_df = pd.DataFrame(tt_data)
        fig_tt = px.bar(
            tt_df,
            x="Skill Tier",
            y="Time Trouble Freq",
            color="Skill Tier",
            color_discrete_map=TIER_COLORS,
            title="Average Time Trouble Frequency by Skill Tier",
        )
        fig_tt.update_layout(showlegend=False, height=370)
    else:
        fig_tt = go.Figure()

    col_top = st.columns(2)
    with col_top[0]:
        st.plotly_chart(fig_heatmap, use_container_width=True)
    with col_top[1]:
        st.plotly_chart(fig_variance, use_container_width=True)

    col_bottom = st.columns(2)
    with col_bottom[0]:
        st.plotly_chart(fig_tt, use_container_width=True)
    with col_bottom[1]:
        st.markdown(
            """
            **Key time-management insights**

            - Beginners spend more time per move and show higher variance, especially in the opening.
            - Intermediate and Advanced players allocate time more consistently across phases.
            - Time-trouble frequency decreases as skill level increases.
            - Stronger players appear to rely more on fast pattern recognition in routine positions.
            """
        )


# ---------------------------------------------------------------------------
# Classification Tab
# ---------------------------------------------------------------------------
def render_classification_tab():
    st.subheader("Skill-Tier Classification Performance")

    # Confusion matrix (normalized row-wise)
    cm_norm = cm_array.astype(float) / cm_array.sum(axis=1, keepdims=True)
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm_norm,
            x=cm_labels,
            y=cm_labels,
            colorscale="Blues",
            text=cm_array,
            texttemplate="%{text}",
            hovertemplate=(
                "True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.2%}<extra></extra>"
            ),
        )
    )
    fig_cm.update_layout(
        title="Skill Tier Classification — Confusion Matrix",
        xaxis_title="Predicted Tier",
        yaxis_title="Actual Tier",
        height=450,
    )

    # Feature importance
    fi = feature_importance.head(15).copy()
    fi["feature_clean"] = (
        fi["feature"].str.replace("white_", "", regex=False)
        .str.replace("_", " ")
        .str.title()
    )
    fig_fi = px.bar(
        fi,
        x="importance",
        y="feature_clean",
        orientation="h",
        color="importance",
        color_continuous_scale="Viridis",
        title="Top 15 Feature Importances (Random Forest)",
        labels={"importance": "Importance", "feature_clean": "Feature"},
    )
    fig_fi.update_layout(
        yaxis=dict(autorange="reversed"),
        height=450,
        coloraxis_showscale=False,
    )

    m = classifier_metrics["metrics"]
    metrics_df = pd.DataFrame(
        [
            ["Test Accuracy", f"{m['test_accuracy']:.1%}"],
            ["Adjacent Accuracy (±1 tier)", f"{m['adjacent_accuracy']:.1%}"],
            ["Macro Precision", f"{m['macro_precision']:.3f}"],
            ["Macro Recall", f"{m['macro_recall']:.3f}"],
            ["Macro F1", f"{m['macro_f1']:.3f}"],
            [
                "Train / Val / Test",
                f"{classifier_metrics['train_size']:,} / {classifier_metrics['val_size']:,} / {classifier_metrics['test_size']:,}",
            ],
        ],
        columns=["Metric", "Value"],
    )

    col_top = st.columns([3, 2])
    with col_top[0]:
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_top[1]:
        st.markdown("**Classification Metrics**")
        st.table(metrics_df)
        st.markdown(
            """
            - Model: Random Forest using 18 behavioral features.
            - "Adjacent accuracy" treats predictions within ±1 tier as correct.
            - Most mistakes occur between neighboring tiers (e.g., Intermediate vs. Advanced).
            """
        )

    st.plotly_chart(fig_fi, use_container_width=True)


# ---------------------------------------------------------------------------
# Cluster Analysis Tab
# ---------------------------------------------------------------------------
def render_cluster_analysis_tab():
    st.subheader("Behavioral Archetypes and Clustering Quality")

    # Cluster size pie chart
    clu_sizes = []
    for cid, info in clustering_results["cluster_names"].items():
        clu_sizes.append(
            {
                "Archetype": info["name"],
                "Size": info["size"],
                "Avg Elo": round(info["avg_elo"]),
            }
        )
    clu_df = pd.DataFrame(clu_sizes)
    fig_pie = px.pie(
        clu_df,
        values="Size",
        names="Archetype",
        title="Cluster Size Distribution",
        color_discrete_sequence=CLUSTER_COLORS,
        hole=0.35,
    )
    fig_pie.update_layout(height=400)

    # Skill composition per cluster (stacked bar)
    comp_data = []
    for _, row in cluster_stats.iterrows():
        cid = int(row["cluster"])
        name = cluster_name_map.get(cid, f"Cluster {cid}")
        for tier in SKILL_TIERS:
            col = f"pct_{tier.lower()}"
            if col in row:
                comp_data.append(
                    {
                        "Cluster": name,
                        "Skill Tier": tier,
                        "Percentage": row[col],
                    }
                )
    comp_df = pd.DataFrame(comp_data)
    fig_comp = px.bar(
        comp_df,
        x="Cluster",
        y="Percentage",
        color="Skill Tier",
        color_discrete_map=TIER_COLORS,
        title="Skill Tier Composition per Cluster",
        category_orders={"Skill Tier": SKILL_TIERS},
    )
    fig_comp.update_layout(height=400, barmode="stack")

    # Avg Elo by cluster
    fig_elo = px.bar(
        clu_df,
        x="Archetype",
        y="Avg Elo",
        color="Archetype",
        color_discrete_sequence=CLUSTER_COLORS,
        title="Average Elo by Cluster",
    )
    fig_elo.update_layout(showlegend=False, height=370)

    # Clustering methods comparison table
    mc = method_comparison.copy()
    mc["silhouette_score"] = mc["silhouette_score"].round(3)
    mc["calinski_harabasz_index"] = mc["calinski_harabasz_index"].round(1)
    mc["davies_bouldin_index"] = mc["davies_bouldin_index"].round(3)
    mc = mc[[
        "method",
        "n_clusters",
        "silhouette_score",
        "calinski_harabasz_index",
        "davies_bouldin_index",
    ]]
    mc.columns = [
        "Method",
        "k",
        "Silhouette",
        "Calinski-Harabasz",
        "Davies-Bouldin",
    ]

    col_top = st.columns([2, 3])
    with col_top[0]:
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_top[1]:
        st.plotly_chart(fig_comp, use_container_width=True)

    col_bottom = st.columns([3, 2])
    with col_bottom[0]:
        st.plotly_chart(fig_elo, use_container_width=True)
    with col_bottom[1]:
        st.markdown("**Clustering Method Comparison**")
        st.table(mc)

        primary_method = clustering_results.get("method", "kmeans")
        primary_k = clustering_results.get("n_clusters", len(cluster_name_map))
        st.markdown(
            f"- Primary clustering: **{primary_method}** with **k = {primary_k}** as used in the dashboard."
        )
        st.markdown(
            "- Alternative methods in the table may achieve better internal metrics "
            "(e.g., higher silhouette, lower Davies–Bouldin) and can guide future refinements."
        )


# ---------------------------------------------------------------------------
# Opening Patterns Tab
# ---------------------------------------------------------------------------
def render_openings_tab():
    st.subheader("Opening Patterns by Skill Tier")

    try:
        games = load_games_processed()
    except Exception as e:  # pragma: no cover - runtime safeguard
        st.warning(
            "Opening pattern analysis requires `games_processed.parquet`. "
            "Please rerun `python run_analysis.py` to regenerate processed games.\n\n"
            f"Details: {e}"
        )
        return

    if games is None or games.empty:
        st.info("No games available for opening analysis.")
        return

    if "white_skill_tier" not in games.columns:
        st.warning(
            "Could not find `white_skill_tier` in processed games. "
            "Opening patterns by tier cannot be computed."
        )
        return

    tiers_selected = st.multiselect(
        "Filter Skill Tiers",
        options=SKILL_TIERS,
        default=SKILL_TIERS,
        key="openings_tiers_multiselect",
    )
    games = games[games["white_skill_tier"].isin(tiers_selected)]

    if games.empty:
        st.info("No games remain after applying the selected skill-tier filters.")
        return

    if "opening_name" in games.columns:
        opening_col = "opening_name"
        opening_label = "Opening Name"
    elif "opening_eco" in games.columns:
        opening_col = "opening_eco"
        opening_label = "ECO Code"
    else:
        st.warning(
            "Could not find `opening_name` or `opening_eco` in processed games. "
            "Opening pattern visualization is not available."
        )
        return

    top_n = st.slider(
        "Number of top openings to show per tier",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    counts = (
        games.groupby(["white_skill_tier", opening_col])
        .size()
        .reset_index(name="count")
    )

    # Keep top-N openings per tier
    top_frames = []
    for tier in tiers_selected:
        tier_counts = (
            counts[counts["white_skill_tier"] == tier]
            .sort_values("count", ascending=False)
            .head(top_n)
        )
        top_frames.append(tier_counts)

    if not top_frames:
        st.info("No openings found for the selected configuration.")
        return

    top_counts = pd.concat(top_frames, ignore_index=True)

    fig = px.bar(
        top_counts,
        x="count",
        y=opening_col,
        color="white_skill_tier",
        orientation="h",
        title=f"Top {top_n} openings by skill tier",
        labels={
            "count": "Games",
            opening_col: opening_label,
            "white_skill_tier": "Skill Tier",
        },
        color_discrete_map=TIER_COLORS,
        category_orders={"white_skill_tier": SKILL_TIERS},
    )
    fig.update_layout(
        height=550,
        yaxis={"categoryorder": "total ascending"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "This view highlights which openings are most frequently played at each skill tier "
        "based on the processed Lichess games."
    )


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
def main():
    st.title("ChessInsight Dashboard")
    st.caption("Team 029 · CSE6242 · Spring 2026")

    st.sidebar.header("How to use this dashboard")
    st.sidebar.markdown(
        """
        1. Run `python run_analysis.py` once to generate processed data and model artifacts.
        2. Start the app with `streamlit run streamlit_app.py`.
        3. Use the tabs above to explore:
           - High-level dataset and model KPIs
           - Player behavior map and archetypes
           - Time management patterns across skill tiers
           - Classification performance and feature importances
           - Cluster compositions and method quality
           - Opening patterns across skill tiers
        """
    )

    tabs = st.tabs(
        [
            "Overview",
            "Player Cluster Map",
            "Time Analysis",
            "Classification",
            "Cluster Analysis",
            "Opening Patterns",
        ]
    )

    with tabs[0]:
        render_overview_tab()
    with tabs[1]:
        render_cluster_tab()
    with tabs[2]:
        render_time_tab()
    with tabs[3]:
        render_classification_tab()
    with tabs[4]:
        render_cluster_analysis_tab()
    with tabs[5]:
        render_openings_tab()


if __name__ == "__main__":
    main()
